# Système de décision de trading GBP/USD

Projet fil rouge — M2 Data Science

## Objectif

Concevoir un système de décision algorithmique sur la paire GBP/USD.
- Fréquence brute : 1 minute (M1)
- Fréquence de décision : 15 minutes (M15)
- Actions : BUY / SELL / HOLD

## Structure du projet

```
project/
├── data/           # Données M1 et M15
├── features/       # Features calculées par année
├── models/
│   ├── v1/         # Meilleur modèle ML
│   └── v2/         # Meilleur modèle RL
├── training/       # Scripts d'entraînement ML et RL
├── evaluation/     # Analyses, baselines, backtests
├── api/            # API FastAPI
└── docker/         # Dockerfile et docker-compose
```

## Pipeline

1. **T01** — Import M1 : fusion timestamp, vérification régularité
2. **T02** — Agrégation M1 → M15 (OHLCV)
3. **T03** — Nettoyage M15 : bougies incomplètes, gaps
4. **T04** — Analyse exploratoire (ADF, ACF, volatilité)
5. **T05** — Feature Engineering V2 (29 features)
6. **T06** — Baselines (random, buy&hold, EMA cross)
7. **T07** — Machine Learning (LogReg, RF, XGB, LGBM, GB)
8. **T08** — Reinforcement Learning (PPO)
9. **T09** — Évaluation finale + backtest 2024
10. **T10** — API FastAPI
11. **T11** — Versioning modèle (registry)
12. **T12** — Docker

## Donnees

- 2022 : Entrainement
- 2023 : Validation
- 2024 : Test final (jamais utilise pour entrainer)

## Demarche et reflexion

Je trouve ma solution pertinente parce qu'elle n'est pas le fruit d'un seul essai reussi, mais d'une vraie demarche iterative ou chaque echec m'a pousse a mieux comprendre le probleme.

Au depart, j'ai construit un pipeline propre (import M1, agregation M15, nettoyage) et j'ai lance mes premiers modeles de Machine Learning (Logistic Regression, Random Forest, XGBoost, LightGBM, Gradient Boosting) avec les 20 features obligatoires du cahier des charges. Le premier test a ete catastrophique : tous les modeles perdaient de l'argent sur les donnees de validation 2023. Le meilleur modele affichait un Sharpe negatif autour de -30, ce qui signifie qu'il aurait ete plus rentable de ne rien faire du tout.

Je me suis alors rendu compte de deux choses. Premierement, forcer un modele a prendre position (BUY ou SELL) a chaque bougie de 15 minutes etait irrealiste : un vrai trader ne prend pas position en permanence. J'ai donc introduit un mecanisme de HOLD avec un seuil de confiance optimise. Si le modele n'est pas suffisamment sur de sa prediction, il s'abstient. Deuxiemement, 20 features ne suffisaient pas a capturer la complexite du marche. J'ai enrichi le feature engineering en passant de 20 a 29 features (V2) avec des indicateurs temporels (heure, jour, sessions Londres/New York, encodage cyclique), puis a 56 features (V3) en ajoutant des blocs Price Action (momentum multi-horizon, ratio body/range, bougies consecutives), Regime (Bollinger Bands, Stochastique, divergence MACD, alignement de tendance) et Micro-structure (ratio de volume, proxy VWAP, RSI multi-timeframe).

Pour le Machine Learning, j'ai aussi remplace l'evaluation simple par une walk-forward validation sur 4 fenetres glissantes, ce qui est beaucoup plus realiste pour des donnees financieres ou le regime de marche change dans le temps. Malgre ces ameliorations, le meilleur modele ML (LightGBM) restait en territoire negatif sur le test 2024, avec un Sharpe de -51.67 et seulement -74.1 pips de perte. C'etait deja un progres enorme par rapport aux premieres versions qui perdaient des milliers de pips, mais ce n'etait pas suffisant.

C'est le Reinforcement Learning qui a fait la difference. J'ai concu un environnement de trading personnalise (TradingEnv) avec un espace d'observation de 58 dimensions (56 features + position courante + PnL non realise), des couts de transaction de 2 pips de spread, et une fonction de recompense normalisee en pips avec penalite de drawdown. Mon premier entrainement PPO a 100 000 timesteps donnait des resultats mediocres. J'ai donc augmente a 500 000 timesteps, elargi le reseau de politique a [128, 64] neurones, et adouci la penalite de drawdown pour ne pas paralyser l'agent.

Le resultat sur les donnees de test 2024 (jamais vues pendant l'entrainement) parle de lui-meme :

| Strategie | Sharpe | Profit (pips) | Max Drawdown | Trades | Win Rate |
|-----------|--------|---------------|--------------|--------|----------|
| Random | -43.57 | -31 304.7 | 31 311.9 | 16 454 | 26.55% |
| Regles (EMA Cross) | -3.64 | -3 147.8 | 3 217.3 | 516 | 48.48% |
| ML (LightGBM + HOLD) | -16.88 | -7 674.0 | 7 687.8 | 3 409 | 32.61% |
| **RL (PPO)** | **0.55** | **+467.9** | **849.4** | **418** | **50.37%** |

Le modele RL est le seul a avoir un Sharpe positif et a generer du profit sur 2024. Il prend moins de trades (418 contre 16 454 pour le random), ce qui montre qu'il a appris a etre selectif. Son profit factor de 1.01 confirme que les gains depassent les pertes, meme apres couts de transaction. J'ai aussi realise des stress tests en faisant varier le spread de 1 a 5 pips pour verifier que la strategie ne s'effondre pas dans des conditions de marche moins favorables.

Ce que je retiens de cette experience, c'est que la performance brute sur les donnees d'entrainement ne veut rien dire. Mon meilleur modele ML avait de bons scores en 2022, mais il s'effondrait des qu'il rencontrait un regime de marche different. Le RL, en revanche, a appris une politique de decision plus robuste parce qu'il a ete expose a des milliers de scenarios differents et qu'il optimise directement le profit cumule plutot qu'une metrique statistique deconnectee de la realite financiere.

## Backlog des taches

| ID  | Tache                                  | Responsable | Branche Git                   |
|-----|----------------------------------------|-------------|-------------------------------|
| T01 | Import M1 + controle regularite        | dany        | dany__T01__import_m1          |
| T02 | Agregation M1 -> M15                   | dany        | dany__T02__m15_agg            |
| T03 | Nettoyage M15 + rapport qualite        | dany        | dany__T03__cleaning_m15       |
| T04 | Analyse exploratoire + ADF/ACF         | dany        | dany__T04__eda                |
| T05 | Feature Pack V2 (court terme + regime) | dany        | dany__T05__features           |
| T06 | Baseline regles + backtest simple      | dany        | dany__T06__baselines          |
| T07 | ML (split temporel + modeles + eval)   | dany        | dany__T07__ml_training        |
| T08 | RL (env + reward + entrainement)       | dany        | dany__T08__rl_env             |
| T09 | Evaluation robuste (benchmarks + 2024) | dany        | dany__T09__evaluation         |
| T10 | API (contrat + endpoints + modele)     | dany        | dany__T10__api_predict        |
| T11 | Versioning modele (v1/v2 + registry)   | dany        | dany__T11__model_versioning   |
| T12 | Docker + execution reproductible       | dany        | dany__T12__docker             |
