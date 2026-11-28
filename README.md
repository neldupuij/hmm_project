# Volatilité stochastique hebdomadaire de l’AUD/USD par PMH

Le projet applique le modèle univarié et la marche gaussienne PMH de [Shapovalova (2021)](https://www.mdpi.com/1099-4300/23/4/466). Les notebooks [Bootstrap](HMM_data_%26_BPF.ipynb) et [Bootstrap et proposition guidée](HMM_data_%26_BPF_APF.ipynb) préparent les données et construisent les filtres. Le module [pmh.py](pmh/pmh.py) reprend le filtre guidé de `HMM_data_&_BPF_APF.ipynb` et ajoute l’échantillonnage de la marginale des paramètres. Le filtre n’enregistre pas les trajectoires latentes, qui sont nécessaires si l’on veut aussi produire des tirages de la postérieure jointe de l’algorithme 2.

## Refaire les expériences

Depuis le dépôt, avec Python 3.12. Le dossier [pmh/](pmh/) regroupe le code, les données et les résultats.

```sh
cd pmh
python -m pip install -r requirements-pmh.txt
python run_pmh.py --output resultats_pmh
python run_pmh.py --ig-scale .0125 --output resultats_sensibilite
```

Le CSV contient les cotations quotidiennes de la série [FRED DEXUSAL](https://fred.stlouisfed.org/series/DEXUSAL) sur 1994–2003. On prend la dernière cotation disponible dans chaque semaine se terminant le mardi, comme dans le notebook, puis la différence des logarithmes. On centre et réduit les rendements pour travailler sur l’échelle voisine de celle de la figure 7 et du tableau A2. Il reste 522 observations, contre 519 dans l’article. On n’a pas retrouvé les observations originales exactes.

Le filtre reprend l’équation 26 et les étapes de l’algorithme A1. Il utilise la densité normale de variance `beta**2 * exp(h)`, le rendement courant et le rééchantillonnage systématique si l’ESS particulaire descend sous `N/2`. L’implémentation compilée retrouve les log-vraisemblances du notebook à 10⁻⁹ près sur cinq graines de validation. Pour une observation, une intégration numérique donne 0,213751 et le filtre à 100 000 particules donne 0,213813 avec la graine de la vérification.

On choisit les priors de l’équation 75 pour l’expérience principale. Le prior d’échelle de β est proportionnel à 1/β, celui de (φ+1)/2 est Beta(20, 1.5) et celui de σ² est IG(5, 0.25), avec une densité proportionnelle à `v**(-6) * exp(-0.25/v)`. La seconde expérience garde les données fixes et remplace le paramètre d’échelle de l’IG par 0.0125, convention alternative discutée dans le cours. Le PMH marche sur (β, φ, σ) et rejette les propositions hors support. Le facteur de changement de variable `2*sigma` donne le terme `-11*log(sigma)` du prior sur σ.

La calibration répète le filtre 60 fois pour chacun des couples (point, N), avec trois points de paramètres et cinq valeurs de N, pour les deux filtres. On sélectionne le plus petit N pour lequel toutes les variances estimées de log-vraisemblance du filtre guidé sont au plus 1. Le seuil et les points sont des choix de calibration. Une chaîne pilote estime la forme de la covariance. Parmi quatre multiplicateurs appliqués au facteur de Cholesky, on choisit le meilleur ESS minimal par seconde dans la plage de 20 à 40 % d’acceptation, ou sur toute la grille si aucun essai ne s’y trouve. Le temps mesuré peut modifier ce choix en relançant l’expérience. Les dix chaînes finales démarrent en des points dispersés et utilisent des graines distinctes ainsi qu’une proposition fixe. Chaque chaîne conserve 4 000 tirages sur 5 000.

Les dossiers [expérience principale](pmh/resultats_pmh/) et [analyse de sensibilité](pmh/resultats_sensibilite/) contiennent les traces, les histogrammes, les courbes de calibration et les CSV. `diagnostics.csv` donne le taux d’acceptation, le temps et l’ESS par chaîne. `posterieurs.csv` donne la moyenne, un intervalle de crédibilité à quantiles égaux, l’écart-type des dix moyennes et le R-hat classique calculé sur les demi-chaînes. L’écart-type inter-chaînes mesure la variabilité numérique des moyennes estimées. La largeur de l’intervalle mesure l’incertitude a posteriori sur le paramètre.

## Résultats de l’expérience

À N = 128, la variance de log-vraisemblance du filtre guidé varie de 0,33 à 0,71 aux trois points testés, contre 0,46 à 0,74 avec le bootstrap. Le classement s’inverse au troisième point. Le multiplicateur retenu vaut 1 dans les deux expériences. Le taux d’acceptation des dix chaînes principales va de 32,3 à 35,4 %. Aucun appel au filtre avec un candidat valide n’a donné de log-vraisemblance non finie dans ces chaînes.

| Prior sur σ² | Moyenne β | Moyenne φ | Moyenne σ | Intervalle à 95 % de σ | Écart-type des moyennes de σ |
| --- | ---: | ---: | ---: | ---: | ---: |
| IG(5, 0.25) | 0,938 | 0,942 | 0,192 | [0,141 ; 0,260] | 0,0020 |
| IG(5, 0.0125) | 0,951 | 0,984 | 0,071 | [0,045 ; 0,112] | 0,0017 |

Dans la figure 7, le PMH donne visuellement un β autour de 0,9, un φ proche de 0,98 et un pic de σ près de 0,10. Notre expérience principale retrouve l’ordre de grandeur de β mais place σ plus haut. La seconde convention porte φ près de 0,98 et déplace σ sous le pic publié. La différence de 0,12 entre les deux moyennes de σ dépasse largement la dispersion des moyennes obtenues avec un même prior. Cette expérience montre la sensibilité au choix du prior. Le prior effectivement utilisé pour la courbe PMH publiée et l’effet du changement de série restent inconnus. [Figure 7](https://www.mdpi.com/1099-4300/23/4/466#fig_body_display_entropy-23-00466-f007)

Le R-hat classique scindé vaut 1,005 à 1,006 pour l’expérience principale. Avec le second prior, il monte à 1,017 pour β et à 1,012 pour σ. Les traces et ces valeurs invitent à la prudence sur les queues de la seconde expérience. La dernière semaine porte l’étiquette 6 janvier 2004, mais sa dernière cotation retenue date de décembre 2003.

## Au tableau

**Minutes 0–3 · Modèle et cible.** Dessine trois états latents qui se suivent et trois observations, chacune reliée à son état. Écris `h_t = φ h_{t-1} + σ_η η_t` et `y_t = β exp(h_t/2) ε_t`, avec deux bruits normaux standards indépendants. Dis que β règle l’échelle du rendement, φ la persistance et σ₍η₎ l’amplitude des innovations de l’état. En prenant la variance de l’équation AR(1), montre `V = φ² V + σ_η²`. La variance stationnaire vaut donc `σ_η² / (1 − φ²)` sous `abs(phi) < 1`. Écris ensuite `p(theta | y) ∝ prior(theta) L(theta)` et explique que calculer L demande d’intégrer tous les états cachés.

**Minutes 3–6 · Filtre.** Factorise `L(theta) = produit_t p(y_t | y_1:t-1, theta)`. Pour chaque observation, pars des particules pondérées de l’instant précédent. Tire les nouveaux états sous q, calcule `w = g * f/q`, puis `ell_hat = somme(W_precedent * w)`. Normalise les poids et rééchantillonne si leur ESS `1/somme(W**2)` tombe sous N/2. Dans le bootstrap, q=f et le ratio disparaît. Dans le filtre guidé d’Eq. 26, `m = φ h_{t-1}` et la moyenne de q vaut `m + σ_η² ∂_h log g(y_t | m)`. Pour la retrouver, linéarise `log g` autour de m, ajoute la log-densité de transition et complète le carré. La correction f/q conserve le modèle lorsque la proposition utilise `y_t`.

**Minutes 6–9 · PMH et exactitude.** Écris `theta* = theta + A z` puis `alpha = min(1, prior(theta*) L_hat* / [prior(theta) L_hat])`. La symétrie de la marche fait disparaître le ratio des propositions de paramètres. À gauche du tableau, garde le temps t du filtre. À droite, garde l’itération i de la chaîne. En cas de rejet, `theta, score` gardent leur ancienne valeur. En notant u les tirages du filtre, écris la cible étendue `prior(theta) L_hat(theta,u) m_theta(u)`. Intègre u et utilise `E[L_hat | theta] = L(theta)`. L’exactitude concerne la cible marginale. La chaîne finie comporte une erreur Monte Carlo, et le logarithme de l’estimateur est généralement biaisé par Jensen.

**Minutes 9–12 · Calibration.** Ouvre `calibration.png` puis `propositions.csv`. À θ fixe, la dispersion de `log L_hat` mesure le bruit du filtre. À N fixe, l’échelle de la marche change la distance des propositions. Une marche avec des pas minuscules peut accepter souvent et explorer lentement. Un filtre trop bruité peut donner une surestimation exceptionnelle de L, puis une série de rejets. Associe chaque réglage à sa mesure et explique pourquoi l’acceptation s’interprète avec l’ESS et les traces.

**Minutes 12–15 · Résultats.** Montre les marginales et les traces, puis compare visuellement aux courbes roses de la figure 7. Donne les limites liées aux données reconstruites et aux priors. Montre la variabilité sur dix chaînes. Si on te demande si les résultats prouvent la convergence, explique que les traces, le R-hat et l’ESS permettent surtout de repérer des signes de mauvais mélange.

### Questions à travailler à voix haute

- Pourquoi `h_t/2` dans l’équation du rendement ? `exp(h_t/2)` multiplie son écart-type. En le mettant au carré, on obtient la variance `beta**2 * exp(h_t)`.
- Pourquoi additionner les logarithmes ? Le filtre estime une vraisemblance qui est un produit de densités prédictives. On calcule le logarithme de chaque incrément par log-sum-exp, puis on additionne.
- Quel est le rôle des deux propositions ? La gaussienne de l’état latent guide le filtre à θ fixé et nécessite les poids f/q. La gaussienne des paramètres propose un nouveau θ et passe par le test d’acceptation.
- Que mesure exactement l’ESS ? Dans le filtre, la concentration des poids à un instant t. Dans le PMH, l’information d’une chaîne autocorrélée par rapport à des tirages indépendants.
- Pourquoi conserver l’estimation courante après un rejet ? Elle fait partie de l’état de Metropolis–Hastings sur la cible étendue. Recalculer seulement le dénominateur revient à changer l’algorithme.
- Que se passe-t-il si on multiplie tous les rendements par c > 0 ? La même équation d’observation s’obtient avec `beta_new = c*beta`. La comparaison des marginales de β exige des données sur la même échelle.

Dans le PDF du cours, travaille en priorité les chapitres 5–6 (modèle), 7 (filtre), 8–9 (exactitude et calibration), puis 15 (entraînement à l’oral). Les chapitres 2 et 4 servent à vérifier les priors et Metropolis–Hastings.
