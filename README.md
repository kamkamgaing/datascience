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

## Données

- 2022 : Entraînement
- 2023 : Validation
- 2024 : Test final (jamais utilisé pour entraîner)
