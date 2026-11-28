"""Run the AUD/USD experiment and save the diagnostics and figures."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew

from pmh import particle_loglik, sample_pmh


def aud_returns(path):
    fx = pd.read_csv(path, index_col=0, parse_dates=True)["DEXUSAL"]
    fx = pd.to_numeric(fx, errors="coerce").dropna().loc["1994":"2003"]
    returns = np.log(fx.resample("W-TUE").last().dropna()).diff().dropna()
    return (returns - returns.mean()) / returns.std()


def chain_ess(draws):
    """Univariate ESS using the initial positive pairs of autocorrelations."""
    centered = draws - draws.mean(axis=0)
    n = len(centered)
    spectrum = np.fft.rfft(centered, n=2*n, axis=0)
    acov = np.fft.irfft(abs(spectrum)**2, n=2*n, axis=0)[:n]
    sizes = []
    for covariance in acov.T:
        if covariance[0] <= 0:
            sizes.append(0.0)
            continue
        rho = covariance / covariance[0]
        pairs = rho[:n//2*2].reshape(-1, 2).sum(axis=1)
        end = np.flatnonzero(pairs <= 0)
        positive = pairs[:end[0]] if len(end) else pairs
        tau = max(1, 2 * np.minimum.accumulate(positive).sum() - 1)
        sizes.append(n / tau)
    return np.array(sizes)


def split_rhat(draws):
    """Classical split R-hat on the original parameter scale."""
    half = draws.shape[1] // 2
    split = np.concatenate([draws[:, :half], draws[:, -half:]])
    within = split.var(axis=1, ddof=1).mean(axis=0)
    between = half * split.mean(axis=1).var(axis=0, ddof=1)
    return np.sqrt(((half - 1) * within + between) / (half * within))


def calibrate(y, reps, ig_scale):
    points = [(0.9, 0.85, 0.20), (0.9, 0.95, 0.24), (0.85, 0.98, 0.30)]
    grid = [64, 128, 256, 512, 1024]
    records = []
    for guided in [True, False]:
        for n in grid:
            rng = np.random.default_rng(543)
            for index, theta in enumerate(points):
                start = perf_counter()
                values = [particle_loglik(y, theta, n, rng, guided)
                          for _ in range(reps)]
                records.append(dict(guided=guided, particles=n, point=index + 1,
                                    variance=np.var(values, ddof=1),
                                    ms=1000 * (perf_counter() - start) / reps))
    diagnostic = pd.DataFrame(records)
    worst = diagnostic[diagnostic.guided].groupby("particles").variance.max()
    eligible = worst[worst <= 1]
    n = int(eligible.index[0]) if len(eligible) else grid[-1]
    pilot, _, _ = sample_pmh(y, points[1], np.diag([.065, .04, .04]), n, 1400, 12, ig_scale)
    base = np.linalg.cholesky(np.cov(pilot[400:].T) + np.eye(3) * 1e-6)
    scales = []
    for scale in [0.65, 1.0, 1.45, 2.0]:
        start = perf_counter()
        draws, accepted, failures = sample_pmh(y, pilot[-1], scale * base, n, 1400, 42, ig_scale)
        seconds = perf_counter() - start
        scales.append(dict(scale=scale, acceptance=accepted[400:].mean(),
                           min_ess_per_second=chain_ess(draws[400:]).min() / seconds,
                           failures=failures))
    trials = pd.DataFrame(scales)
    eligible = trials[trials.acceptance.between(.2, .4)]
    candidates = eligible if len(eligible) else trials
    scale = candidates.loc[candidates.min_ess_per_second.idxmax(), "scale"]
    return n, scale * base, diagnostic, trials


def figures(output, draws, calibration):
    labels = [r"$\beta$", r"$\varphi$", r"$\sigma_\eta$"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), layout="constrained")
    for j, label in enumerate(labels):
        for chain in draws[:4]:
            axes[0, j].plot(chain[:, j], alpha=.52, lw=.6)
        axes[0, j].set(title=label, xlabel="Itération après chauffe")
        axes[1, j].hist(draws[:, :, j].ravel(), bins=38, density=True,
                        color="#b43e6a", alpha=.82)
        axes[1, j].set(xlabel=label, ylabel="Densité")
    fig.suptitle("AUD/USD hebdomadaire · PMH")
    fig.savefig(output / "posterieurs.png", dpi=165)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3.4), layout="constrained", sharey=True)
    for point, axis in enumerate(axes, 1):
        for guided, group in calibration[calibration.point == point].groupby("guided"):
            axis.loglog(group.particles, group.variance, "o-",
                        label="Guidé" if guided else "Bootstrap")
        axis.axhline(1, color="gray", linestyle="--", linewidth=1)
        axis.set(title=f"Point {point}", xlabel="Particules")
    axes[0].set_ylabel("Variance estimée de log-vraisemblance")
    axes[-1].legend()
    fig.savefig(output / "calibration.png", dpi=165)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--burn", type=int, default=1000)
    parser.add_argument("--chains", type=int, default=10)
    parser.add_argument("--pf-reps", type=int, default=60)
    parser.add_argument("--ig-scale", type=float, choices=[.25, .0125], default=.25)
    parser.add_argument("--data", type=Path, default=Path(__file__).with_name("aud_usd_fred.csv"))
    parser.add_argument("--output", type=Path, default=Path("resultats_pmh"))
    args = parser.parse_args()
    if args.iterations - args.burn < 4 or args.burn < 0 or args.chains < 2 or args.pf_reps < 2:
        parser.error("Need at least 4 retained draws, 2 chains and 2 filter replicates.")
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    returns = aud_returns(args.data)
    y = returns.to_numpy()
    # Exclude one-time JIT compilation from the calibration timings.
    for guided in [True, False]:
        particle_loglik(y[:2], (0.9, .93, .23), 4, np.random.default_rng(0), guided)
    particle_loglik(y[:2], (0.9, .93, .23), 4, np.random.default_rng(0))
    n, proposal, calibration, trials = calibrate(y, args.pf_reps, args.ig_scale)
    calibration.to_csv(output / "calibration.csv", index=False)
    trials.to_csv(output / "propositions.csv", index=False)
    print(f"Calibration terminée. N={n}\n{trials.to_string(index=False)}", flush=True)
    chains, diagnostics = [], []
    for index in range(args.chains):
        fraction = index / (args.chains - 1)
        initial = [.84 + .22*fraction, .83 + .15*fraction, .30 - .12*fraction]
        start = perf_counter()
        draws, accepted, failures = sample_pmh(y, initial, proposal, n, args.iterations,
                                               100 + index, args.ig_scale)
        retained = draws[args.burn:]
        chains.append(retained)
        diagnostics.append([100 + index, accepted[args.burn:].mean(), failures,
                            perf_counter() - start, *retained.mean(axis=0), *chain_ess(retained)])
        print(f"Chaîne {index+1}/{args.chains} · acceptation {diagnostics[-1][1]:.1%}", flush=True)
    draws = np.array(chains)
    np.save(output / "chaines.npy", draws)
    names = ["beta", "phi", "sigma"]
    columns = ["seed", "acceptance", "failures", "seconds"]
    columns += [f"mean_{name}" for name in names] + [f"ess_{name}" for name in names]
    pd.DataFrame(diagnostics, columns=columns).to_csv(output / "diagnostics.csv", index=False)
    flat = draws.reshape(-1, 3)
    summary = pd.DataFrame({"parameter": names, "mean": flat.mean(axis=0),
                            "q025": np.quantile(flat, .025, axis=0),
                            "q975": np.quantile(flat, .975, axis=0),
                            "sd_chain_means": draws.mean(axis=1).std(axis=0, ddof=1),
                            "split_rhat": split_rhat(draws)})
    summary.to_csv(output / "posterieurs.csv", index=False)
    settings = dict(n_obs=len(y), first=str(returns.index[0].date()),
                    last=str(returns.index[-1].date()), skew=float(skew(y)),
                    pearson_kurtosis=float(kurtosis(y, fisher=False)), particles=n,
                    proposal_covariance=(proposal @ proposal.T).tolist(),
                    iterations=args.iterations, burn=args.burn, chains=args.chains,
                    pf_reps=args.pf_reps, ig_scale=args.ig_scale)
    (output / "protocole.json").write_text(json.dumps(settings, indent=2), encoding="utf-8")
    figures(output, draws, calibration)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
