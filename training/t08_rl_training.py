"""
T08 — Reinforcement Learning : Entraînement PPO pour Trading GBP/USD M15
=========================================================================
Ce script contient :
1. TradingEnv : Environnement Gym custom pour le trading
2. Entraînement PPO via stable-baselines3 sur données 2022
3. Évaluation sur données 2023 (validation)
4. Sauvegarde du modèle et des métadonnées
"""

import os
import json
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

# =============================================================================
# Configuration
# =============================================================================
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR = PROJECT_ROOT / "models" / "v2"

# Colonnes des 20 features utilisées comme state
FEATURE_COLUMNS = [
    "return_1", "return_4", "ema_20", "ema_50", "ema_diff",
    "rsi_14", "rolling_std_20", "range_15m", "body", "upper_wick",
    "lower_wick", "ema_200", "distance_to_ema200", "slope_ema50",
    "atr_14", "rolling_std_100", "volatility_ratio", "adx_14",
    "macd", "macd_signal"
]

# Hyperparamètres PPO
PPO_PARAMS = {
    "gamma": 0.99,
    "learning_rate": 3e-4,
    "batch_size": 64,
    "n_steps": 2048,
    "n_epochs": 10,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": SEED,
}

TOTAL_TIMESTEPS = 100_000
TRANSACTION_COST = 0.0002  # 2 pips de spread


# =============================================================================
# Environnement Gym Custom : TradingEnv
# =============================================================================
class TradingEnv(gym.Env):
    """
    Environnement de trading pour Reinforcement Learning.

    - Observation : 20 features normalisées + position courante + PnL non réalisé = 22 dims
    - Action : Discrete(3) → HOLD=0, BUY=1, SELL=2
    - Reward : PnL du step - coût de transaction - pénalité drawdown
    - Épisode : parcours séquentiel complet du dataset
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, features: np.ndarray, closes: np.ndarray, transaction_cost: float = TRANSACTION_COST):
        """
        Args:
            features: Array (n_steps, 20) des features normalisées
            closes: Array (n_steps,) des prix de clôture
            transaction_cost: Coût de transaction par changement de position
        """
        super().__init__()

        self.features = features
        self.closes = closes
        self.transaction_cost = transaction_cost
        self.n_steps = len(features)

        # Espaces
        # 20 features + position (-1, 0, 1) + unrealized_pnl
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(22,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)  # HOLD=0, BUY=1, SELL=2

        # État interne
        self._reset_state()

    def _reset_state(self):
        """Réinitialise l'état interne de l'environnement."""
        self.current_step = 0
        self.position = 0          # 0=flat, 1=long, -1=short
        self.entry_price = 0.0     # Prix d'entrée de la position courante
        self.cumulative_pnl = 0.0  # PnL cumulé total
        self.peak_pnl = 0.0        # PnL maximal atteint (pour drawdown)
        self.max_drawdown = 0.0    # Drawdown maximal
        self.num_trades = 0        # Nombre de trades
        self.unrealized_pnl = 0.0  # PnL non réalisé

    def _get_observation(self) -> np.ndarray:
        """Construit le vecteur d'observation à partir du step courant."""
        obs = np.zeros(22, dtype=np.float32)
        obs[:20] = self.features[self.current_step]
        obs[20] = float(self.position)
        obs[21] = float(self.unrealized_pnl)
        return obs

    def _map_action_to_position(self, action: int) -> int:
        """Convertit l'action en position cible."""
        # HOLD=0 → garder la position actuelle
        # BUY=1 → position long (+1)
        # SELL=2 → position short (-1)
        if action == 0:
            return self.position  # HOLD
        elif action == 1:
            return 1   # BUY → long
        elif action == 2:
            return -1  # SELL → short
        return self.position

    def reset(self, seed=None, options=None):
        """Réinitialise l'environnement au début du dataset."""
        super().reset(seed=seed)
        self._reset_state()
        obs = self._get_observation()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        """
        Exécute une action et avance d'un step.

        Returns:
            observation, reward, terminated, truncated, info
        """
        assert self.action_space.contains(action), f"Action invalide : {action}"

        # Prix courant et prix suivant
        current_close = self.closes[self.current_step]

        # Vérifier si c'est le dernier step
        if self.current_step >= self.n_steps - 2:
            # Fermer toute position ouverte
            if self.position != 0:
                realized = (current_close - self.entry_price) * self.position
                self.cumulative_pnl += realized - self.transaction_cost
                self.num_trades += 1
                self.position = 0

            obs = self._get_observation()
            info = self._get_info()
            return obs, 0.0, True, False, info

        next_close = self.closes[self.current_step + 1]

        # Déterminer la position cible
        target_position = self._map_action_to_position(action)

        # Calculer le reward
        reward = 0.0
        transaction_penalty = 0.0

        # Si changement de position
        if target_position != self.position:
            # Fermer la position existante si nécessaire
            if self.position != 0:
                realized = (current_close - self.entry_price) * self.position
                self.cumulative_pnl += realized
                transaction_penalty += self.transaction_cost
                self.num_trades += 1

            # Ouvrir la nouvelle position si nécessaire
            if target_position != 0:
                self.entry_price = current_close
                transaction_penalty += self.transaction_cost
                self.num_trades += 1
            else:
                self.entry_price = 0.0

            self.position = target_position

        # PnL du step (variation de prix * position)
        step_pnl = (next_close - current_close) * self.position

        # Mettre à jour le PnL non réalisé
        if self.position != 0:
            self.unrealized_pnl = (next_close - self.entry_price) * self.position
        else:
            self.unrealized_pnl = 0.0

        # Reward = PnL du step - coûts de transaction - pénalité drawdown
        reward = step_pnl - transaction_penalty

        # Mettre à jour le PnL cumulé (avec le step_pnl non réalisé pour le tracking)
        total_equity = self.cumulative_pnl + self.unrealized_pnl

        # Mise à jour du drawdown
        if total_equity > self.peak_pnl:
            self.peak_pnl = total_equity
        current_drawdown = self.peak_pnl - total_equity
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown

        # Pénalité de drawdown (progressive au-delà de 5%)
        drawdown_penalty = max(0.0, current_drawdown - 0.05) * 0.1
        reward -= drawdown_penalty

        # Avancer au step suivant
        self.current_step += 1

        # Vérifier la fin de l'épisode
        terminated = self.current_step >= self.n_steps - 1

        # Si terminé, fermer toute position
        if terminated and self.position != 0:
            close_price = self.closes[self.current_step]
            realized = (close_price - self.entry_price) * self.position
            self.cumulative_pnl += realized - self.transaction_cost
            self.num_trades += 1
            self.position = 0
            self.unrealized_pnl = 0.0

        obs = self._get_observation()
        info = self._get_info()

        return obs, float(reward), terminated, False, info

    def _get_info(self) -> dict:
        """Retourne les informations supplémentaires."""
        return {
            "cumulative_pnl": self.cumulative_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "max_drawdown": self.max_drawdown,
            "num_trades": self.num_trades,
            "current_position": self.position,
            "current_step": self.current_step,
        }


# =============================================================================
# Callback pour le logging
# =============================================================================
class TradingCallback(BaseCallback):
    """Callback pour afficher les métriques d'entraînement."""

    def __init__(self, eval_freq: int = 10_000, verbose: int = 1):
        super().__init__(verbose)
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0 and self.verbose > 0:
            infos = self.locals.get("infos", [])
            if infos:
                last_info = infos[-1]
                pnl = last_info.get("cumulative_pnl", 0.0)
                dd = last_info.get("max_drawdown", 0.0)
                trades = last_info.get("num_trades", 0)
                print(
                    f"  [Step {self.n_calls:>7d}] "
                    f"PnL cumulé: {pnl:+.6f} | "
                    f"Drawdown max: {dd:.6f} | "
                    f"Trades: {trades}"
                )
        return True


# =============================================================================
# Fonctions utilitaires
# =============================================================================
def load_and_prepare_data(parquet_path: str, scaler: StandardScaler = None, fit: bool = False):
    """
    Charge un fichier parquet et prépare les features.

    Args:
        parquet_path: Chemin vers le fichier parquet
        scaler: StandardScaler à utiliser (ou None pour en créer un nouveau)
        fit: Si True, fit le scaler sur ces données

    Returns:
        features: Array (n, 20) des features normalisées
        closes: Array (n,) des prix de clôture
        scaler: Le scaler utilisé
        df: Le DataFrame original
    """
    print(f"Chargement de {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    print(f"  → {len(df)} lignes chargées")

    # Vérification des colonnes
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    # Extraire les features et les prix de clôture
    features_raw = df[FEATURE_COLUMNS].values.astype(np.float64)
    closes = df["close_15m"].values.astype(np.float64)

    # Remplacer les NaN et Inf
    features_raw = np.nan_to_num(features_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # Normalisation
    if scaler is None:
        scaler = StandardScaler()

    if fit:
        features_normalized = scaler.fit_transform(features_raw).astype(np.float32)
        print("  → Scaler ajusté sur ces données (fit)")
    else:
        features_normalized = scaler.transform(features_raw).astype(np.float32)
        print("  → Scaler appliqué (transform uniquement)")

    # Remplacer les NaN post-normalisation
    features_normalized = np.nan_to_num(features_normalized, nan=0.0, posinf=0.0, neginf=0.0)

    return features_normalized, closes, scaler, df


def evaluate_model(model, env: TradingEnv) -> dict:
    """
    Évalue le modèle sur un environnement complet.

    Returns:
        Dictionnaire de métriques
    """
    obs, info = env.reset()
    done = False

    step_pnls = []
    actions_taken = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        prev_pnl = env.cumulative_pnl + env.unrealized_pnl

        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated

        current_pnl = env.cumulative_pnl + env.unrealized_pnl
        step_pnls.append(current_pnl - prev_pnl)
        actions_taken.append(int(action))

    # Calculer les métriques
    step_pnls = np.array(step_pnls)
    cumulative_pnl = info["cumulative_pnl"]
    max_drawdown = info["max_drawdown"]
    num_trades = info["num_trades"]

    # Ratio de Sharpe annualisé (M15 → 252 jours * 24h * 4 bougies/heure)
    periods_per_year = 252 * 24 * 4
    if len(step_pnls) > 0 and np.std(step_pnls) > 0:
        sharpe = (np.mean(step_pnls) / np.std(step_pnls)) * np.sqrt(periods_per_year)
    else:
        sharpe = 0.0

    # Distribution des actions
    actions_array = np.array(actions_taken)
    n_hold = np.sum(actions_array == 0)
    n_buy = np.sum(actions_array == 1)
    n_sell = np.sum(actions_array == 2)

    metrics = {
        "cumulative_pnl": float(cumulative_pnl),
        "max_drawdown": float(max_drawdown),
        "sharpe_ratio": float(sharpe),
        "num_trades": int(num_trades),
        "total_steps": len(step_pnls),
        "action_distribution": {
            "hold": int(n_hold),
            "buy": int(n_buy),
            "sell": int(n_sell),
        },
    }

    return metrics


# =============================================================================
# Script principal
# =============================================================================
def main():
    print("=" * 70)
    print("T08 — Entraînement RL (PPO) pour Trading GBP/USD M15")
    print("=" * 70)

    # Fixer le seed global
    np.random.seed(SEED)

    # -------------------------------------------------------------------
    # 1. Charger les données
    # -------------------------------------------------------------------
    print("\n[1/5] Chargement des données...")

    train_features, train_closes, scaler, train_df = load_and_prepare_data(
        str(FEATURES_DIR / "features_2022.parquet"), fit=True
    )

    val_features, val_closes, _, val_df = load_and_prepare_data(
        str(FEATURES_DIR / "features_2023.parquet"), scaler=scaler, fit=False
    )

    print(f"\n  Train : {len(train_features)} steps")
    print(f"  Validation : {len(val_features)} steps")

    # -------------------------------------------------------------------
    # 2. Créer les environnements
    # -------------------------------------------------------------------
    print("\n[2/5] Création des environnements...")

    train_env = TradingEnv(
        features=train_features,
        closes=train_closes,
        transaction_cost=TRANSACTION_COST,
    )

    val_env = TradingEnv(
        features=val_features,
        closes=val_closes,
        transaction_cost=TRANSACTION_COST,
    )

    print("  → Environnements train et validation créés")
    print(f"  → Observation space : {train_env.observation_space.shape}")
    print(f"  → Action space : {train_env.action_space}")

    # -------------------------------------------------------------------
    # 3. Entraîner le modèle PPO
    # -------------------------------------------------------------------
    print("\n[3/5] Entraînement PPO...")
    print(f"  Hyperparamètres : {json.dumps(PPO_PARAMS, indent=2)}")
    print(f"  Total timesteps : {TOTAL_TIMESTEPS:,}")

    model = PPO(
        "MlpPolicy",
        train_env,
        gamma=PPO_PARAMS["gamma"],
        learning_rate=PPO_PARAMS["learning_rate"],
        batch_size=PPO_PARAMS["batch_size"],
        n_steps=PPO_PARAMS["n_steps"],
        n_epochs=PPO_PARAMS["n_epochs"],
        clip_range=PPO_PARAMS["clip_range"],
        ent_coef=PPO_PARAMS["ent_coef"],
        vf_coef=PPO_PARAMS["vf_coef"],
        max_grad_norm=PPO_PARAMS["max_grad_norm"],
        seed=PPO_PARAMS["seed"],
        verbose=1,
        policy_kwargs={"net_arch": [64, 64]},
    )

    callback = TradingCallback(eval_freq=10_000)

    print("\n  Début de l'entraînement...\n")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    print("\n  Entrainement termine !")

    # -------------------------------------------------------------------
    # 4. Évaluation sur train et validation
    # -------------------------------------------------------------------
    print("\n[4/5] Évaluation du modèle...")

    # Évaluation sur train (2022)
    train_env_eval = TradingEnv(
        features=train_features,
        closes=train_closes,
        transaction_cost=TRANSACTION_COST,
    )
    train_metrics = evaluate_model(model, train_env_eval)

    print("\n  === Métriques TRAIN (2022) ===")
    print(f"  PnL cumulé     : {train_metrics['cumulative_pnl']:+.6f}")
    print(f"  Drawdown max   : {train_metrics['max_drawdown']:.6f}")
    print(f"  Sharpe ratio   : {train_metrics['sharpe_ratio']:.4f}")
    print(f"  Nombre trades  : {train_metrics['num_trades']}")
    print(f"  Actions : HOLD={train_metrics['action_distribution']['hold']}, "
          f"BUY={train_metrics['action_distribution']['buy']}, "
          f"SELL={train_metrics['action_distribution']['sell']}")

    # Évaluation sur validation (2023)
    val_metrics = evaluate_model(model, val_env)

    print("\n  === Métriques VALIDATION (2023) ===")
    print(f"  PnL cumulé     : {val_metrics['cumulative_pnl']:+.6f}")
    print(f"  Drawdown max   : {val_metrics['max_drawdown']:.6f}")
    print(f"  Sharpe ratio   : {val_metrics['sharpe_ratio']:.4f}")
    print(f"  Nombre trades  : {val_metrics['num_trades']}")
    print(f"  Actions : HOLD={val_metrics['action_distribution']['hold']}, "
          f"BUY={val_metrics['action_distribution']['buy']}, "
          f"SELL={val_metrics['action_distribution']['sell']}")

    # -------------------------------------------------------------------
    # 5. Sauvegarde du modèle et des métadonnées
    # -------------------------------------------------------------------
    print("\n[5/5] Sauvegarde du modèle...")

    # Créer le dossier de sortie
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Sauvegarder le modèle PPO
    model_path = MODELS_DIR / "best_rl_model"
    model.save(str(model_path))
    print(f"  → Modèle sauvegardé : {model_path}.zip")

    # Sauvegarder le scaler
    import joblib
    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, str(scaler_path))
    print(f"  → Scaler sauvegardé : {scaler_path}")

    # Sauvegarder les métadonnées (model_info.json)
    model_info = {
        "version": "v2",
        "type": "rl",
        "algorithm": "PPO",
        "framework": "stable-baselines3",
        "policy": "MlpPolicy",
        "net_arch": [64, 64],
        "hyperparameters": {
            "gamma": PPO_PARAMS["gamma"],
            "learning_rate": PPO_PARAMS["learning_rate"],
            "batch_size": PPO_PARAMS["batch_size"],
            "n_steps": PPO_PARAMS["n_steps"],
            "n_epochs": PPO_PARAMS["n_epochs"],
            "clip_range": PPO_PARAMS["clip_range"],
            "ent_coef": PPO_PARAMS["ent_coef"],
            "vf_coef": PPO_PARAMS["vf_coef"],
            "max_grad_norm": PPO_PARAMS["max_grad_norm"],
            "total_timesteps": TOTAL_TIMESTEPS,
            "seed": SEED,
            "transaction_cost": TRANSACTION_COST,
        },
        "features": FEATURE_COLUMNS,
        "observation_space": "Box(22,) = 20 features + position + unrealized_pnl",
        "action_space": "Discrete(3) = HOLD(0), BUY(1), SELL(2)",
        "training_data": "features_2022.parquet (24617 lignes)",
        "validation_data": "features_2023.parquet (21434 lignes)",
        "metrics_train": train_metrics,
        "metrics_validation": val_metrics,
        "created_at": datetime.now().isoformat(),
        "model_path": "models/v2/best_rl_model.zip",
        "scaler_path": "models/v2/scaler.joblib",
    }

    info_path = MODELS_DIR / "model_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    print(f"  → Métadonnées sauvegardées : {info_path}")

    print("\n" + "=" * 70)
    print("T08 — Entraînement RL terminé avec succès !")
    print("=" * 70)

    return model, val_metrics


if __name__ == "__main__":
    main()
