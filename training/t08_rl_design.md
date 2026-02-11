# T08 — Conception du système de Reinforcement Learning pour Trading GBP/USD M15

## 1. Problème métier

### Objectif
Concevoir un agent de Reinforcement Learning capable de prendre des décisions de trading (BUY / SELL / HOLD) sur la paire GBP/USD en timeframe M15 (15 minutes), afin de **maximiser le Profit & Loss (PnL) cumulé** tout en **contrôlant le drawdown maximal**.

### Horizon temporel
- **Fréquence de décision** : toutes les 15 minutes (1 bougie M15 = 1 step)
- **Données d'entraînement** : 2022 (24 617 steps)
- **Données de validation** : 2023 (21 434 steps)
- **Données de test final** : 2024 (24 680 steps)

### Contraintes
- Drawdown maximal à surveiller (pénalité dans le reward)
- Coûts de transaction (spread = 2 pips ≈ 0.0002)
- Pas de slippage modélisé (slippage = 0)
- Position unique : l'agent ne peut détenir qu'une seule position à la fois (flat, long ou short)

---

## 2. Modélisation MDP (Markov Decision Process)

### 2.1 State (Observation)

L'espace d'observation est un vecteur continu de **22 dimensions** :

| # | Feature | Description |
|---|---------|-------------|
| 1 | `return_1` | Rendement sur 1 période (normalisé) |
| 2 | `return_4` | Rendement sur 4 périodes (normalisé) |
| 3 | `ema_20` | EMA 20 périodes (normalisée) |
| 4 | `ema_50` | EMA 50 périodes (normalisée) |
| 5 | `ema_diff` | Différence EMA20 - EMA50 (normalisée) |
| 6 | `rsi_14` | RSI 14 périodes (normalisé) |
| 7 | `rolling_std_20` | Écart-type roulant 20 périodes (normalisé) |
| 8 | `range_15m` | Range de la bougie M15 (normalisé) |
| 9 | `body` | Corps de la bougie (normalisé) |
| 10 | `upper_wick` | Mèche haute (normalisée) |
| 11 | `lower_wick` | Mèche basse (normalisée) |
| 12 | `ema_200` | EMA 200 périodes (normalisée) |
| 13 | `distance_to_ema200` | Distance au EMA200 (normalisée) |
| 14 | `slope_ema50` | Pente de l'EMA50 (normalisée) |
| 15 | `atr_14` | ATR 14 périodes (normalisé) |
| 16 | `rolling_std_100` | Écart-type roulant 100 périodes (normalisé) |
| 17 | `volatility_ratio` | Ratio de volatilité (normalisé) |
| 18 | `adx_14` | ADX 14 périodes (normalisé) |
| 19 | `macd` | MACD (normalisé) |
| 20 | `macd_signal` | Signal MACD (normalisé) |
| 21 | `position` | Position courante : 0=flat, 1=long, -1=short |
| 22 | `unrealized_pnl` | PnL non réalisé de la position ouverte (normalisé) |

**Normalisation** : StandardScaler (mean=0, std=1) ajusté sur les données 2022 uniquement (éviter le data leakage).

### 2.2 Action

Espace d'action **discret** à 3 valeurs :

| Action | Code | Description |
|--------|------|-------------|
| HOLD | 0 | Ne rien faire, conserver la position actuelle |
| BUY | 1 | Ouvrir/maintenir une position longue |
| SELL | 2 | Ouvrir/maintenir une position courte |

**Logique de transition de position** :
- Si position=flat et action=BUY → ouvrir long (coût de transaction appliqué)
- Si position=flat et action=SELL → ouvrir short (coût de transaction appliqué)
- Si position=long et action=SELL → fermer long + ouvrir short (double coût)
- Si position=short et action=BUY → fermer short + ouvrir long (double coût)
- Si position=long et action=HOLD → conserver long
- Si position=short et action=HOLD → conserver short
- Si position=flat et action=HOLD → rester flat

### 2.3 Reward

La fonction de reward est conçue pour encourager la rentabilité tout en pénalisant le risque excessif :

```
reward_t = pnl_step - transaction_penalty - drawdown_penalty
```

Où :
- **pnl_step** = `(close[t+1] - close[t]) * position_t` → PnL réalisé sur le step
- **transaction_penalty** = `transaction_cost * |changement_position|`
  - `transaction_cost = 0.0002` (2 pips de spread)
  - Appliqué uniquement lors d'un changement de position
- **drawdown_penalty** = `max(0, drawdown - 0.05) * 0.1`
  - Pénalité progressive quand le drawdown dépasse 5%
  - Encourage l'agent à couper les pertes

### 2.4 Environnement

- **Type** : Simulateur basé sur données historiques (backtesting)
- **Parcours** : Séquentiel (pas de random reset) — chaque épisode = un parcours complet du dataset
- **Slippage** : 0 (non modélisé)
- **Transaction cost** : 0.0002 (2 pips)
- **Position sizing** : Toujours 1 lot (normalisation du PnL en pourcentage)

**Info dict retourné à chaque step** :
- `cumulative_pnl` : PnL cumulé depuis le début de l'épisode
- `max_drawdown` : Drawdown maximal atteint
- `num_trades` : Nombre de trades exécutés
- `current_position` : Position courante

---

## 3. Algorithme choisi : PPO (Proximal Policy Optimization)

### Justification du choix

| Critère | PPO | DQN | A2C | SAC |
|---------|-----|-----|-----|-----|
| Espace d'action discret | Oui, natif | Oui, natif | Oui, natif | Non, continu |
| Stabilite d'entrainement | Excellente | Moyenne | Moyenne | Bonne |
| Sample efficiency | Moyenne | Moyenne | Faible | Bonne |
| Facilite d'implementation | Oui (SB3) | Oui (SB3) | Oui (SB3) | Non (pas discret) |
| Robustesse aux hyperparams | Tres bonne | Sensible | Moyenne | Moyenne |

**PPO est retenu** car :
1. **Stabilité** : Le clipping du ratio de politique empêche les mises à jour trop agressives, crucial pour les données financières bruitées
2. **Action discrète** : Support natif de `Discrete(3)` sans adaptation
3. **Robustesse** : Moins sensible aux hyperparamètres que DQN (pas de replay buffer à tuner)
4. **Implémentation** : Disponible dans `stable-baselines3` avec une API mature

### Hyperparamètres

| Paramètre | Valeur | Justification |
|-----------|--------|---------------|
| `gamma` | 0.99 | Discount factor élevé — l'agent valorise les gains futurs |
| `learning_rate` | 3e-4 | Valeur par défaut de PPO, bon point de départ |
| `batch_size` | 64 | Compromis entre stabilité et vitesse |
| `n_steps` | 2048 | Nombre de steps par rollout — capture environ 1 journée de trading |
| `n_epochs` | 10 | Nombre de passes sur chaque batch — valeur standard |
| `clip_range` | 0.2 | Clipping par défaut — empêche les mises à jour extrêmes |
| `ent_coef` | 0.01 | Encourage l'exploration initiale |
| `vf_coef` | 0.5 | Poids de la value function loss |
| `max_grad_norm` | 0.5 | Gradient clipping pour la stabilité |
| `seed` | 42 | Reproductibilité |
| `total_timesteps` | 100 000 | Environ 4 passes sur les données 2022 |

### Architecture du réseau (policy)

- **Type** : `MlpPolicy` (Multi-Layer Perceptron)
- **Architecture** : 2 couches cachées de 64 neurones chacune
- **Activation** : Tanh
- **Réseaux séparés** pour la politique (actor) et la valeur (critic)

---

## 4. Pipeline d'entraînement et d'évaluation

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  features_2022  │────▶│  TradingEnv  │────▶│  PPO Training   │
│  (TRAIN)        │     │  (Gym)       │     │  100K timesteps  │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
┌─────────────────┐     ┌──────────────┐     ┌─────────────────┐
│  features_2023  │────▶│  TradingEnv  │────▶│  Évaluation     │
│  (VALIDATION)   │     │  (Gym)       │     │  Métriques      │
└─────────────────┘     └──────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │  Sauvegarde     │
                                              │  models/v2/     │
                                              └─────────────────┘
```

### Métriques d'évaluation

1. **Profit cumulé** : PnL total sur la période de validation
2. **Drawdown maximal** : Plus grande perte depuis un pic de PnL
3. **Ratio de Sharpe** : `mean(returns) / std(returns) * sqrt(252*24*4)` (annualisé M15)
4. **Nombre de trades** : Nombre total de changements de position
5. **Win rate** : Pourcentage de trades gagnants

---

## 5. Risques et limitations

| Risque | Mitigation |
|--------|-----------|
| Overfitting sur 2022 | Validation sur 2023, early stopping potentiel |
| Données non-stationnaires | Features normalisées, RL s'adapte mieux que ML supervisé |
| Coûts de transaction ignorés/sous-estimés | Spread de 2 pips inclus, mais pas de slippage |
| Reward hacking | Pénalité de drawdown et coût de transaction |
| Instabilité de l'entraînement | PPO avec clipping, seed fixe pour reproductibilité |

---

## 6. Fichiers de sortie

| Fichier | Contenu |
|---------|---------|
| `models/v2/best_rl_model.zip` | Modèle PPO entraîné (weights + architecture) |
| `models/v2/model_info.json` | Métadonnées : algo, métriques, date, paramètres |
| `training/t08_rl_design.md` | Ce document de conception |
| `training/t08_rl_training.py` | Code d'entraînement et d'évaluation |
