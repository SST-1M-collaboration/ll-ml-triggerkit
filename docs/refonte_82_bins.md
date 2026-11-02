# Refonte de ll-ml-triggerkit : plusieurs particules, alignement, données à 82 échantillons

Plan de travail rédigé le 2026-11-02. Aucune ligne de code n'est écrite : tous les noms ci-dessous (`Simulation`, `NsbSet`, `Watch`, `window=`, ...) sont des propositions. Seuls `TriggerDataset`, `TriggerChain`, `NpeWeighting`, `compute_statistics` et les fenêtres de `cascade.py` (`Full`, `Fixed`, `AlignedClamped`, `AlignedStrict`) existent.

## 1. Objectifs

1. Un événement a un type (gamma, proton, muon, NSB). Les statistiques donnent, pour chaque type, la part captée du total simulé, une fois le seuil de l'algorithme fixé.
2. L'alignement temporel devient un choix explicite de la chaîne, pas seulement de la cascade, et il est défini par source de données.
3. On travaille sur les données à 82 échantillons. Les statistiques de TDSCAN et patch7 sont à refaire dessus.
4. La compatibilité avec les fichiers à 50 échantillons est conservée.

## 2. Décisions déjà prises

| Sujet | Décision |
|---|---|
| Taux à 50 kHz | Le seuil de l'algorithme est fixé une fois (sur le NSB choisi). Pour chaque type : déclenchés / total simulé, et déclenchés / écrits. |
| Alignement | Choisi par l'utilisateur, source par source (`"none"`, `"given"`, `"on_the_fly"`). Le jeu de données et l'alignement se choisissent ensemble. |
| NSB aligné `given` | On fait confiance à l'indice donné par sim_telarray (22), même quand le TDSCAN python n'y franchit pas le seuil. L'écart est compté dans le journal. |
| Nombre d'événements simulés du NSB | Donné à la main, en paramètre (ici environ 12,5 M pour `tdscan_50khz`). On ne le lit pas dans les fichiers. |
| Muons | Peu d'événements, pas de choix. L'augmentation de données est permise pour tous les types. |
| Protons et muons | Pas dans l'entraînement pour l'instant. Ils sont évalués (`watch`), et prévus comme point d'accroche pour une perte ou une métrique plus tard. |
| Fenêtre du TDSCAN | Le TDSCAN ne voit que les échantillons 16 à 65 : `Fixed(16, 50)`. Il ne peut pas déclencher à l'échantillon 10. La durée du calcul de taux vient de cette fenêtre, pas de la longueur du cube. |
| Coordonnées | `t_trig` est en indice absolu du cube de 82 échantillons (0 à 81). |
| Compatibilité | Conservée : `Full()` sur les anciens fichiers redonne le comportement actuel. |
| Python | 3.12, environnement Poetry (voir `README.md`). dl1-data-handler est la source de la grille hexagonale de la caméra entière. |

## 3. Données disponibles

| Jeu | Où | Contenu | Remarque |
|---|---|---|---|
| Gammas (Novotný) | PC GPU, `/media/ssd/NSB_med4_82bins_hdf5/gammas/` | 200 fichiers, 346 029 événements écrits | `gammas/test1` : 10 fichiers, 117 441 événements. |
| Protons (Novotný) | `.../protons/` | 200 fichiers, 949 462 événements écrits | |
| Muons (Novotný) | `.../muons/` | 100 fichiers, 4 169 événements écrits | Environ 47 par fichier : efficacités bruitées. |
| NSB sans déclenchement (Novotný) | `.../nsb/dummy100000.corsika.gz.NSBmed4_dummy.hdf5` | 100 000 événements | Sert au réglage du seuil du TDSCAN. |
| NSB aligné `tdscan_50khz` (produit) | `simtelFileData/NSB/NewSim_82samples/medium/sim_telarray_tdscan_50khz/` | 100 982 événements déclenchés, environ 12,5 M simulés (125 graines) | Premier franchissement à l'échantillon 22 pour 99,86 %. Seul NSB du CNN en cascade. |
| NSB aligné `tdscan_200khz` (en cours) | Bamboo `~/SST1M/tdscan200k/` | environ 100 000 événements attendus | |
| patch7 à 50 kHz | `.../sim_telarray_patch7_50khz/` | graines 50201-50225 | Non aligné sur le TDSCAN. |

Totaux simulés des particules : `n_showers × shower_reuse` du tableau `configuration/simulation/run` de chaque fichier. Vérifié sur tous les fichiers (2026-11-02, `Source.simulated()`) : gammas 200 × 60 000 = 12 000 000 (`3000 × 20`), `gammas/test1` 10 × 60 000 = 600 000, protons 200 × 40 000 = 8 000 000 (`2000 × 20`), muons 100 × 1 000 = 100 000 (`1000 × 1`). Chaque fichier d'un même type a le même total.

## 4. Interface proposée

```python
D = "/media/ssd/NSB_med4_82bins_hdf5"
tdscan = load_chain("tdscan_82.keras")          # frozen: threshold 8.109374, trig_time="rising"

sim = Simulation(
    gamma=f"{D}/gammas/*.hdf5", proton=f"{D}/protons/*.hdf5", muon=f"{D}/muons/*.hdf5",
    nsb=NsbSet(".../tdscan_50khz/*.hdf5", align="given", at=22, n_simulated=12_500_000,
               made_by=dict(trigger="tdscan", threshold=8.109374, trig_time="rising",
                            observe=(16, 50), rate_hz=50_000)),
)
window = Window(length=16, offset=4)            # [t_trig - 4, t_trig + 12)

ds = TriggerDataset(sim, window=window, align_with=tdscan,
                    train_on=("gamma", "nsb"),
                    watch=[Watch("proton"), Watch("muon"),
                           Watch("proton", align="none", window=Fixed(18, 16), name="proton_raw")],
                    watch_max=20_000, npe_weighting=NpeWeighting(30, 50))
chain.compute_statistics(sim, window=window, align_with=tdscan, folder="stats_cnn_82")
```

Alignement par source :
- `"none"` : rien n'est aligné (entraînement du TDSCAN, fenêtre `Fixed(16, 50)`).
- `"given"` : déjà aligné dans le fichier, à l'indice `at`. Aucun calcul.
- `"on_the_fly"` : on passe l'événement dans l'étage `align_with`, on jette ceux qu'il ne déclenche pas, on découpe la fenêtre autour de `t_trig`. C'est le défaut des gammas, protons et muons.

Vérification de cohérence : la définition `made_by` d'un NSB aligné est comparée à celle de `align_with` (seuil, `trig_time`, fenêtre d'observation). Une différence est une erreur.

Anciens fichiers : `Simulation(gamma=GAMMAS_50, nsb=NSB_50)` avec `Full()` redonne le comportement actuel.

## 5. Règles

**Fenêtre d'observation et taux.** Le TDSCAN lit `Fixed(16, 50)`. La durée du calcul de taux est le nombre d'échantillons de cette fenêtre multiplié par la période d'échantillonnage. Pour une cascade, les taux sont normalisés sur la fenêtre du premier étage (déjà le cas dans `cascade.py`).

**Bornes de `AlignedStrict`.** Le premier étage ne déclenche qu'entre 16 et 65, donc il faut `offset <= 16` et `length - offset <= 17` pour ne jamais sortir du cube de 82 échantillons. Les bornes sont prouvées à la construction.

**NSB aligné.** Le taux final d'une chaîne qui suit le TDSCAN est `rate_hz × fraction des événements alignés que la chaîne garde`. Le nombre d'événements simulés (`n_simulated`) sert à vérifier le taux de 50 kHz de l'étage 1.

**Dénominateurs.**
- Gammas et protons sont diffus (portée du cœur 928 m et 1032 m, cône de 10°), les muons ont un cœur à moins de 3 m et un cône de 6°. « Déclenchés / simulés » n'a donc pas le même sens pour chaque type, et l'indice spectral simulé (-2) n'est pas le spectre physique : ce ne sont pas des taux physiques. On l'écrit dans le rapport et on ajoute l'efficacité par tranche d'énergie.
- Si on sous-échantillonne (`watch_max`, protons), le dénominateur est corrigé de la même fraction.
- Deux chaînes qui filtrent d'abord par le TDSCAN (seuils différents) ne gardent pas les mêmes événements : on les compare sur le même total simulé.

**Augmentation.** Rotations de 120° et 240° et décalage de pixels du NSB, pour tous les types. Pas de décalage temporel du NSB.

**NSB d'entraînement et de statistiques.** Le CNN en cascade n'a que `tdscan_50khz` comme bruit (100 982 événements) : l'entraînement, la validation et les statistiques partagent ces événements. Le sur-apprentissage côté NSB reste donc non testable, comme avant. Option plus tard : générer du NSB sans déclenchement à la volée (prototype numpy du 30/10, hors dépôt), puis le passer dans le TDSCAN python (environ 1 % passe) pour obtenir de nouveaux événements alignés.

**Début de fenêtre du TDSCAN.** Pour retrouver les résultats des 50 échantillons, le noyau temporel (`eps_t=2`) ne lit rien avant l'échantillon 16. On pourra distinguer plus tard ce que le TDSCAN lit de ce où il peut déclencher.

## 6. Étapes

Une branche : `refonte-82-bins`, créée à partir de `main`. Tu commites et pousses. Les calculs lourds sur le PC GPU ne partent que sur ta demande.

| # | Étape | Fichiers | Tests et critères |
|---|---|---|---|
| 1 | Spécification | ce document | Ta relecture. |
| 2 | **Fait (non commité)** : lecture de plusieurs types | `FileIO/FileOpenerCTAOHDF5.py`, `data/dataset.py` (`Simulation`, `NsbSet`) | Chaque événement porte type, énergie, n_pe et total simulé de son fichier ; anciens fichiers inchangés. |
| 3 | **Fait (non commité, voir section 10)** : format de statistiques v2, particules comme entrées du fichier | `Statistics/H5StatsWriter.py` (colonnes `particle`, `passed_stage1`, `t_trig`, `label` inchangé), `Statistics/StatPlotter.py`, `augment.py` | Sur des fichiers à 50 échantillons, les résultats actuels sont retrouvés à l'identique ; le tableau par type est correct. |
| 4 | **Fait (non commité, voir section 10)** : alignement dans la chaîne | `TriggerChain.py`, `cascade.py`, `data/dataset.py` (`align`, `made_by`, `watch`) | Tests des fenêtres ; vérification de `made_by` ; bornes de `AlignedStrict`. |
| 5 | **Fait (non commité, voir section 10)** : points d'accroche des pertes et métriques | `training/` (callback des taux par type) | Comportement par défaut inchangé. |
| 6 | Nouvelles statistiques TDSCAN et patch7 | scripts de `python-reference/` | Voir la section 7. |

## 7. Validations (à passer avant de se fier aux nouveaux chiffres)

1. **Non-régression à 50 échantillons :** mêmes résultats qu'aujourd'hui avec `Full()`.
2. **TDSCAN sur 82 échantillons, fenêtre `Fixed(16, 50)`, NSB sans déclenchement de Novotný :** le seuil 8,109374 donne environ 50 kHz, et les décisions et le taux sont ceux de l'ancien jeu à 50 échantillons.
3. **TDSCAN python contre sim_telarray sur `tdscan_50khz` :** au moins 99,8 % des événements franchissent à 22 avec la fenêtre `Fixed(16, 50)`.
4. **Dénominateurs :** un test avec sous-échantillonnage donne la même efficacité qu'avec tous les événements (à l'erreur statistique près).
5. **Bout en bout :** un petit CNN en cascade (entraînement court, statistiques courtes), sans erreur d'alignement.

## 8. Risques et points ouverts

- **Muons :** 47 événements écrits par fichier : efficacités très bruitées.
- **Configurations de simulation différentes :** les gammas de Novotný viennent de sim_telarray 2021 (déclenchement patch7, fenêtre positionnée par ce déclenchement), le NSB de ta production de 2025.51 avec le TDSCAN intégré. Le TDSCAN déclenche environ 2 échantillons plus tôt que patch7 : l'alignement à la volée devrait absorber l'écart, à surveiller sur la position de l'impulsion dans la fenêtre.
- **Taille des données :** 82 × 432 octets par événement. Les 949 462 protons ne tiennent pas en RAM : `watch_max` et lecture en flux.
- **Matériel (hors code) :** une fenêtre qui commence 4 échantillons avant le déclenchement et finit 12 après demande un tampon dans le FPGA et ajoute au moins 12 échantillons (48 ns) avant la décision.
- **Poids de flux physiques :** hors périmètre.

## 9. Hors périmètre

- Les poids de flux physiques (taux de rayons cosmiques).
- Les protons ou muons dans la perte d'entraînement (seulement le point d'accroche).
- Toute modification de keras-hexagdly.

## 10. Avancement

- **Étape 2 (2026-11-02), branche `refonte-82-bins`, non commitée.**
  - `src/triggerkit/data/simulation.py` (nouveau) : `Simulation`, `Source`, `NsbSet`, `Watch`, `PARTICLES`, `read_n_simulated`. Les fichiers se donnent par chemin, motif glob ou liste ; un motif sans fichier est une erreur. `NsbSet(align="given", at=22, n_simulated=..., made_by=...)` : le nombre simulé d'un NSB aligné est donné à la main, jamais lu.
  - `TriggerDataset` accepte une `Simulation` à la place des deux listes (les deux listes marchent toujours), plus `train_on` (seul `("gamma", "nsb")` pour l'instant), `watch` et `watch_max`. Chaque lot porte `features["particle"]` (0 NSB, 1 gamma, 2 proton, 3 muon). `watch_datasets()` lit les sources observées à part, comme les gammas de validation. `Watch(align=..., window=...)` lève `NotImplementedError` jusqu'à l'étape 4.
  - Tests : `tests/test_simulation.py` (20). Suite complète : 224 tests passés (204 avant).
  - Pas encore fait : la correction du dénominateur quand `watch_max` sous-échantillonne (étape 3), les statistiques par type, l'alignement.
- **Étape 3 (2026-11-02), non commitée : format de statistiques v2.** Décision prise avec toi : les particules sont des entrées du même fichier que les conditions medium / low, et on repart de zéro sur le format pour supprimer les doublons (pas de compatibilité avec les anciens fichiers de statistiques ; le module séparé `particles.py` est supprimé).
  - **Format `stats_v2`** (`Statistics/H5StatsWriter.py`) : `/events` une ligne par événement physique, une seule fois (`particle` 0 NSB / 1 gamma / 2 proton / 3 muon, `condition`, `file`, `event_id`, `tel_id`, `n_pe`, `energy`, ...) ; `/scores` une ligne par (fold, événement) : `score` (ou `decision` pour une chaîne sans score), et des colonnes supplémentaires (la cascade y met ce que fait chaque étage) ; `/folds` le nom, la condition, la config en texte et le budget demandé par type ; `/sources` les dénominateurs par (condition, type) : écrits et simulés (le simulé est celui de tous les fichiers ; un fold qui a un budget lit un préfixe et son dénominateur est mis à l'échelle à la lecture) ; `/files`, `/conditions`. Supprimés : `label`, `fold`, `triggered`, `p_trig`, les compteurs / taux / efficacités de `/folds` et les attributs du fold 0, qui se déduisent des événements.
  - **Lecture** : `Statistics/stats_file.py`, `StatsFile(path).fold(nom)` joint les événements et les scores du fold ; `counts`, `efficiency(fold, type, threshold=, of="simulated"|"read")`, `nsb_rate_hz`, `simulated`. Le seuil se choisit après coup quand la chaîne a un score. Un ancien fichier est refusé avec un message.
  - **Écriture** : `TriggerChain.compute_statistics(sim=, folds=, events=)` lit tous les types de particules (lecteur généralisé : étiquettes 2 et 3, un plafond par type, `file_index`), écrit les événements une fois et les scores par fold. La rotation et le décalage temporel des gerbes (`shower_deg`, `shower_time_shift`) s'appliquent aux gammas, protons et muons, celui du NSB au NSB seul. Les budgets sont `events={type: n ou None}` ; les folds qui lisent les mêmes fichiers partagent une lecture.
  - **Folds** (`augment.py`) : `Fold(sources=Simulation, events=...)`, `make_rotation_folds(geom, specs, {condition: Simulation})`, `make_condition_folds(geom, {nom: (Simulation, events)})` ; les clés `gamma_events` / `nsb_events` / `gamma_deg` / `gamma_time_shift` n'existent plus (`events`, `shower_deg`, `shower_time_shift`).
  - **`StatPlotter`** et `Statistics/folds.py` lisent le nouveau format (via `StatsFile`, sans lecture par morceaux) ; les tracés eux-mêmes n'ont pas changé. Pas encore de courbes par type de particule dans `StatPlotter` (les tableaux par type se font avec `StatsFile`). La cascade écrit ses fichiers par étage dans ce format.
  - **Code mort supprimé** : lecture des anciens fichiers (pickle `allow_legacy_pickle`, layouts `/gamma` / `/nsb`, `n_clusters`, `p_trig`), `TauRateController`, `ScheduleCallback`, `Stages/DBSCAN.py`, `parse_qformat` et 3 fonctions `_fixed_point_*` de `Helper/Quantize.py`. À vérifier encore (non touchés) : `RateConstrainedBCE`, `NSBRateHz`, `TauMetric`, `TDSCANMetric` (enregistrés pour recharger d'anciens modèles), `load_legacy_npy_history`, `compile_chain_autoencoder`.
  - **Exemples et scripts adaptés** : `examples/stats_*.py`, `stats_common.py` (triggerkit) ; `stats_tdscan.py`, `stats_patch7.py`, `stats_hexcnn.py`, `stats_patch7_Q.py`, `stats_report_mix_hillas.py`, `dataset_config.py` (`get_simulation`), `position_baseline/make_stats_*.py`, `plot_eff_vs_npe*.py`, `example_laurent/make_stats.py`, `run_tdscan_baseline.py` (digicam). Pas adaptés : `evaluate_perf_*.py`, `pso_search*.py`, `quantize_grid_search_tdscan.py`, `sweep_*.py` (ils utilisent l'ancien paquet `TriggerChainTensorFlowPy`).
  - **Tests** : `test_h5writer.py` (13), `test_compute_statistics.py` (8, plusieurs types, folds, deux conditions, rotation des protons, chaîne sans score), `test_file_reading.py`, `test_augment.py`, `test_simulation.py`, `test_statplotter.py`, `test_folds.py`, `test_cascade_windows.py` adaptés.
  - **Pas encore fait** : `/events` ne garde que les colonnes que `compute_statistics` donne (je n'ai pas encore supprimé celles que `StatPlotter` ne lit pas) ; les courbes par type dans `StatPlotter` ; le test sur les fichiers à 82 échantillons et les nouvelles statistiques TDSCAN / patch7 (étape 6, sur ta demande) ; l'alignement (étape 4).
- **Étape 4 (2026-11-02/03), non commitée : alignement.**
  - **`triggerkit.alignment`** (nouveau) : `Window(length, offset, clamp=False)` (`length` échantillons à partir de `offset` avant le déclenchement), `Aligner(upstream, window, observe=, trig_time=)`, `crop_given`, `crop_fixed`, `plan_sources`. L'`Aligner` fait passer la fenêtre `observe` (par exemple `Fixed(16, 50)`) dans la chaîne amont figée, garde les événements qu'elle déclenche, lit l'instant de déclenchement (premier franchissement montant par défaut), et coupe la fenêtre ; les bornes sont prouvées à la construction (`Window(16, 4)` sur un déclenchement entre 16 et 65 donne des fenêtres entre 12 et 76, dans les 82 échantillons). `check_made_by` compare la définition du NSB aligné (`trigger`, `threshold`, `trig_time`, `observe`) à celle de la chaîne amont et refuse un désaccord. Il s'appuie sur `TriggerCascade` (désormais acceptée avec un seul étage et `time_refs=(0,)`).
  - **Par source** (`Source.align` : `none` / `given` / `on_the_fly`, vu à l'étape 2) : `given` coupe `[at - offset, at - offset + length)` sans rien calculer ; `on_the_fly` utilise l'aligneur ; `none` n'accepte qu'une fenêtre `Fixed` / `Full`. Par défaut, avec `align_with`, les gammas, protons et muons sont alignés à la volée.
  - **`TriggerDataset(window=, align_with=, observe=, trig_time=)`** : les lots portent des waveforms de la longueur de la fenêtre, plus `features["t_trig"]` et `features["window_start"]` ; les événements non déclenchés par la chaîne amont sont écartés, à chaque époque. `Watch(align=, window=)` donne une règle propre à une source observée (par exemple un proton non aligné sur `Fixed(18, 16)` pour tester la chaîne seule).
  - **`compute_statistics(window=, align_with=, observe=, trig_time=)`** : la fenêtre est coupée après la transformation du fold (rotation, décalage). Les événements non déclenchés par l'amont sont lus et comptés mais ne passent pas dans la chaîne : score plancher `GATED_OUT_SCORE` (-1e30), `passed_gate` 0 ; ils comptent donc dans tous les taux, seuils et AUC. Colonnes supplémentaires de `/scores` : `passed_gate`, `t_trig`, `window_start`. Taux de NSB : normalisé sur la fenêtre observée de la chaîne amont (`align_with.window_size`), ou, pour un NSB fait par un déclencheur (`made_by` avec `rate_hz`), `rate_hz` fois la fraction d'événements que la chaîne garde (`window_sec` du fichier reste la fenêtre physique de la chaîne ; `rate_hz` est dans `/sources` et c'est le lecteur qui en tire la fenêtre d'un événement de NSB, par condition : `StatsFile.nsb_window_sec`).
  - **`StatPlotter`** : les histogrammes de scores laissent de côté les événements écartés (`drop_gated_out`), tout le reste les garde.
  - **Tests** : `test_alignment.py` (19), 4 nouveaux dans `test_compute_statistics.py`, 1 dans `test_statplotter.py`.
  - **Contrôles sur de vraies données à 82 échantillons** (local, `simtelFileData/NSB_med4_82bins_hdf5` et `NSB/NewSim_82samples/medium/sim_telarray_tdscan_50khz/hdf5`, TDSCAN python au seuil 8,109374, `observe=Fixed(16, 50)`, `Window(16, 4)`) : (1) sur 3000 événements du NSB aligné par le TDSCAN de sim_telarray, le TDSCAN python tire sur 100 % d'entre eux et trouve l'instant 22 pour 99,93 % (2 événements à 16) : validation 3 du plan réussie ; (2) premiers gammas / protons / muons (3000 lus par type pour les deux premiers, 1187 muons) : le TDSCAN tire sur 29,3 % des gammas (instants les plus fréquents 22, 23, 21 comme dans les statistiques du 01/11), 24,0 % des protons, 69,0 % des muons. La chaîne en aval de ce test est un patch7 de remplacement (rien de physique à en tirer).
  - **Pas encore fait** : le test sur les nouvelles statistiques complètes TDSCAN / patch7 à 82 échantillons (étape 6, sur ta demande), la validation 2 du plan (TDSCAN limité à `Fixed(16, 50)` sur le NSB sans déclenchement de Novotný : taux et décisions comparés à l'ancien jeu), les pertes et métriques par type (étape 5), et la possibilité de prendre `observe` / `trig_time` d'une description du NSB au lieu de les redonner.
- **Étape 5 (2026-11-03), non commitée : protons et muons dans la perte et les métriques.**
  - **Comme négatifs de l'entraînement** : `TriggerDataset(train_on=("gamma", "nsb", "proton", "muon"), negative_weights={"proton": 0.25}, max_extra_samples_train=...)`. Le label binaire reste gamma = 1 contre tout le reste = 0 ; chaque événement porte `features["particle"]` (son code) et `features["particle_weight"]` (poids du proton ou du muon comme négatif, 1 pour un gamma ou du NSB). Avec `npe_weighting`, ce poids multiplie le poids négatif de `features["npe_channels"]`, donc la BCE pondérée et la perte AUC par paires le prennent en compte sans autre changement ; sans `npe_weighting`, on l'utilise comme poids d'échantillon. Les fichiers de chaque type supplémentaire sont répartis entre entraînement et validation comme les fichiers gamma (`split_extra_files()`), et alignés comme eux (fenêtre, `align_with`). Par défaut (`train_on=("gamma", "nsb")`), rien ne change.
  - **Comme métrique** : `ParticleRates` (`triggerkit.training`), un callback Keras qui, à chaque époque, fixe la coupure sur le NSB du flux de validation (fraction gardée `nsb_fraction`, 1 % par défaut) et affiche la fraction de chaque type au-dessus, pour le flux de validation et pour les sources `watch` : `PARTICLES epoch=3 cut=-8.12 nsb=0.0100 gamma=0.3612 proton=0.0214 muon=0.4033` ; les valeurs vont aussi dans `logs` (`val_<type>_at_nsb`).
  - **Tests** : `tests/test_extra_negatives.py` (7), `test_simulation.py` adapté.
- **Validation 2 du plan (2026-11-03), faite.** TDSCAN python (poids de puissances de 2, seuil 8,109374) limité à `Fixed(16, 50)` par `compute_statistics(window=Fixed(16, 50))`, sur le NSB sans déclenchement de Novotný (30 000 événements lus sur 100 000, pas plus : l'erreur statistique est déjà d'environ ±3 kHz) : 292 déclenchés, soit un taux de **48,7 kHz** (48,9 kHz avec la correction -ln(1-f)), pour 50 kHz visés et 49,5 kHz mesurés dans sim_telarray ; la durée de calcul du taux vient de la fenêtre observée (200 ns = 50 échantillons), pas du cube (328 ns : le taux apparaîtrait environ 30 kHz plus bas, donc faux). Gammas : 30,4 % déclenchés sur 3 000 (29,8 % des gammas écrits dans la note du 01/11). Les événements ne sont pas ceux de l'ancien jeu à 50 échantillons, la comparaison porte donc sur le taux et l'efficacité, pas sur les décisions une à une.
- **Données NSB déclenchées (2026-11-03)** : `simtelFileData/NSB_med4_82bins_hdf5/nsb/tdscan50k` (143 fichiers : 25 `nsb120_patch7_50khz_*` et 118 `nsb120_tdscan50khz_*`, à ne pas mélanger : utiliser le motif `tdscan50k/nsb120_tdscan50khz_*.hdf5` pour un NSB aligné par le TDSCAN) et `nsb/tdscan200k` (31 fichiers, environ 102 000 événements) ; copiés aussi sur le PC GPU, `/media/ssd/NSB_med4_82bins_hdf5/nsb/`. Chaque fichier a une table `run` avec `n_showers` 100 000 (un total simulé par graine).
- **Audit et corrections (2026-11-03), non commités.** Quatre défauts reproduits et corrigés : un chemin littéral avec des crochets n'est plus pris pour un motif glob (`expand_files` garde un chemin qui existe) ; `ParticleRates` ignore un lot vide ; un fichier dont la chaîne a un score mais aucun seuil s'ouvre et se lit (voir ci-dessous) ; les scripts `make_stats_*` construisent leurs `Simulation` dans `main()` (le message clair du contrôle des fichiers revient). Risques de conception traités : le taux de NSB n'est plus un attribut du fichier (la fenêtre d'un événement de NSB se calcule par condition à la lecture) ; `StatsFile` garde des comptes en cache, a `clear_cache()`, lit seulement des colonnes légères à l'ouverture dans `StatPlotter` ; une seule fonction de Wilson (`metrics.wilson`, qui renvoie les erreurs) ; `Fold` refuse l'ancienne paire `(gamma_files, nsb_files)` avec un message clair ; Ctrl+C est pris en compte même sur des lots vides ; les sources sans condition correspondent aux folds sans condition (plus de condition « all » parasite). Dénominateur d'un fold plafonné : l'hypothèse « les premiers événements sont représentatifs » est écrite dans `StatsFile.simulated`. Code mort que j'avais introduit : supprimé.
- **Décisions de l'utilisateur (2026-11-03) appliquées au format.** (1) **Aucun seuil et aucun taux dans le fichier** : les scores sont sauvegardés avant le seuil, qui se choisit à la lecture (`StatsFile.efficiency(..., threshold=)`, `nsb_rate_hz(threshold=)`, `threshold_for_nsb_rate(fold, 50_000)`) ; sans seuil donné, on lit au seuil dont la chaîne était décrite (le JSON de la chaîne, qui reste l'identité de la chaîne) ; une chaîne décrite sans seuil se lit avec un seuil donné. Les attributs `threshold` et `comparison` n'existent plus. Un fichier de chaîne sans score (OR, cascade) garde sa décision. (2) **Chaque run a un UUID** (`StatsFile.uuid`, `compute_statistics` le garde dans `chain.last_stats_uuid` et le renvoie dans `res["uuid"]`) et (3) **un onglet `/description`** : texte libre (`compute_statistics(description=...)`, `H5StatsWriter.describe`), et toujours remplis : UUID, date et heure UTC et locale, machine, utilisateur, ligne de commande, versions de Python / triggerkit / TensorFlow / Keras / numpy et commit git (avec `+dirty` s'il y a des modifications). Les scripts `stats_*.py` et `make_stats_*.py` passent une description.
