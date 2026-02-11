"""
T08 -- Reinforcement Learning : Entrainement PPO ameliore pour Trading GBP/USD M15
====================================================================================
Ameliorations :
  - 56 features V3 au lieu de 20 -> state de dimension 58 (56 + position + unrealized_pnl)
  - Reward normalise en pips (signal plus fort)
  - 500K timesteps (au lieu de 100K)
  - Penalite de drawdown progressive plus douce
  - Evaluation sur 2023 ET 2024
"""

import os
import sys
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
import joblib

# =============================================================================
# Configuration
# =============================================================================
SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FEATURES_DIR = PROJECT_ROOT / "features"
MODELS_DIR = PROJECT_ROOT / "models" / "v2"

# 56 features V3
FEATURE_COLUMNS = [
    # Court terme (11)
    "return_1", "return_4", "ema_20", "ema_50", "ema_diff",
    "rsi_14", "rolling_std_20", "range_15m", "body",
    "upper_wick", "lower_wick",
    # Contexte & regime (9)
    "ema_200", "distance_to_ema200", "slope_ema50",
    "atr_14", "rolling_std_100", "volatility_ratio",
    "adx_14", "macd", "macd_signal",
    # Temporel (9)
    "hour", "day_of_week", "is_london", "is_ny", "is_overlap",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
    # Price Action (10)
    "return_8", "return_16", "return_96",
    "momentum_4", "momentum_8", "body_ratio",
    "consecutive_candles", "high_20_ratio", "low_20_ratio",
    "high_96_ratio",
    # Regime (12)
    "bb_width", "bb_position", "stoch_k", "stoch_d",
    "macd_hist", "macd_hist_diff",
    "above_ema20", "above_ema50", "above_ema200", "trend_alignment",
    "atr_normalized", "volatility_change",
    # Micro-structure (5)
    "volume_ratio", "vwap_proxy",
    "rsi_7", "rsi_21", "rsi_divergence",
]

N_FEATURES = len(FEATURE_COLUMNS)  # 56
OBS_DIM = N_FEATURES + 2  # 58

PPO_PARAMS = {
    "gamma": 0.995,
    "learning_rate": 1e-4,
    "batch_size": 128,
    "n_steps": 4096,
    "n_epochs": 10,
    "clip_range": 0.2,
    "ent_coef": 0.005,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "seed": SEED,
}

TOTAL_TIMESTEPS = 500_000
TRANSACTION_COST_PIPS = 2
PIP_VALUE = 0.0001


# =============================================================================
# Environnement : reward en pips
# =============================================================================
class TradingEnv(gym.Env):
    """
    Environnement de trading ameliore.
    - Observation : 56 features + position + unrealized_pnl = 58 dims
    - Action : Discrete(3) -> HOLD=0, BUY=1, SELL=2
    - Reward : PnL en PIPS (normalise) - cout transaction - penalite drawdown
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, features, closes, transaction_cost=0.0002):
        super().__init__()
        self.features = features
        self.closes = closes
        self.n_steps = len(features)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)
        self._reset_state()

    def _reset_state(self):
        self.current_step = 0
        self.position = 0
        self.entry_price = 0.0
        self.cumulative_pnl = 0.0
        self.peak_pnl = 0.0
        self.max_drawdown = 0.0
        self.num_trades = 0
        self.unrealized_pnl = 0.0

    def _get_obs(self):
        obs = np.zeros(OBS_DIM, dtype=np.float32)
        obs[:N_FEATURES] = self.features[self.current_step]
        obs[N_FEATURES] = float(self.position)
        obs[N_FEATURES + 1] = float(self.unrealized_pnl) / 100.0
        return obs

    def _get_info(self):
        return {
            "cumulative_pnl": self.cumulative_pnl,
            "unrealized_pnl": self.unrealized_pnl,
            "max_drawdown": self.max_drawdown,
            "num_trades": self.num_trades,
            "current_position": self.position,
            "current_step": self.current_step,
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_state()
        return self._get_obs(), self._get_info()

    def step(self, action):
        current_close = self.closes[self.current_step]

        if self.current_step >= self.n_steps - 2:
            if self.position != 0:
                realized = (current_close - self.entry_price) * self.position / PIP_VALUE
                self.cumulative_pnl += realized - TRANSACTION_COST_PIPS
                self.num_trades += 1
                self.position = 0
            return self._get_obs(), 0.0, True, False, self._get_info()

        next_close = self.closes[self.current_step + 1]

        if action == 0:
            target_pos = self.position
        elif action == 1:
            target_pos = 1
        else:
            target_pos = -1

        reward_pips = 0.0
        tx_cost = 0.0

        if target_pos != self.position:
            if self.position != 0:
                realized = (current_close - self.entry_price) * self.position / PIP_VALUE
                self.cumulative_pnl += realized
                tx_cost += TRANSACTION_COST_PIPS
                self.num_trades += 1
            if target_pos != 0:
                self.entry_price = current_close
                tx_cost += TRANSACTION_COST_PIPS
                self.num_trades += 1
            else:
                self.entry_price = 0.0
            self.position = target_pos

        step_pnl = (next_close - current_close) * self.position / PIP_VALUE

        if self.position != 0:
            self.unrealized_pnl = (next_close - self.entry_price) * self.position / PIP_VALUE
        else:
            self.unrealized_pnl = 0.0

        reward_pips = step_pnl - tx_cost

        total_equity = self.cumulative_pnl + self.unrealized_pnl
        if total_equity > self.peak_pnl:
            self.peak_pnl = total_equity
        current_dd = self.peak_pnl - total_equity
        if current_dd > self.max_drawdown:
            self.max_drawdown = current_dd

        dd_penalty = max(0.0, current_dd - 50) * 0.01
        reward_pips -= dd_penalty

        reward = reward_pips / 10.0

        self.current_step += 1
        terminated = self.current_step >= self.n_steps - 1

        if terminated and self.position != 0:
            close_price = self.closes[self.current_step]
            realized = (close_price - self.entry_price) * self.position / PIP_VALUE
            self.cumulative_pnl += realized - TRANSACTION_COST_PIPS
            self.num_trades += 1
            self.position = 0
            self.unrealized_pnl = 0.0

        return self._get_obs(), float(reward), terminated, False, self._get_info()


# =============================================================================
# Callback
# =============================================================================
class TradingCallback(BaseCallback):
    def __init__(self, eval_freq=25_000, verbose=1):
        super().__init__(verbose)
        self.eval_freq = eval_freq

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0 and self.verbose > 0:
            infos = self.locals.get("infos", [])
            if infos:
                info = infos[-1]
                print(
                    f"  [Step {self.n_calls:>7,}] "
                    f"PnL: {info.get('cumulative_pnl', 0):+.1f} pips | "
                    f"DD: {info.get('max_drawdown', 0):.1f} pips | "
                    f"Trades: {info.get('num_trades', 0)}"
                )
        return True


# =============================================================================
# Utilitaires
# =============================================================================
def load_and_prepare_data(path, scaler=None, fit=False):
    print(f"  Chargement : {path}")
    df = pd.read_parquet(path)
    print(f"    -> {len(df)} lignes")

    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes : {missing}")

    features = df[FEATURE_COLUMNS].values.astype(np.float64)
    closes = df["close_15m"].values.astype(np.float64)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    if scaler is None:
        scaler = StandardScaler()
    if fit:
        features_norm = scaler.fit_transform(features).astype(np.float32)
    else:
        features_norm = scaler.transform(features).astype(np.float32)
    features_norm = np.nan_to_num(features_norm, nan=0.0, posinf=0.0, neginf=0.0)

    return features_norm, closes, scaler, df


def evaluate_model(model, env):
    obs, info = env.reset()
    done = False
    step_pnls = []
    actions = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        prev = env.cumulative_pnl + env.unrealized_pnl
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        cur = env.cumulative_pnl + env.unrealized_pnl
        step_pnls.append(cur - prev)
        actions.append(int(action))

    pnls = np.array(step_pnls)
    actions = np.array(actions)

    if len(pnls) > 1 and np.std(pnls) > 0:
        sharpe = (np.mean(pnls) / np.std(pnls)) * np.sqrt(252 * 96)
    else:
        sharpe = 0.0

    gains = pnls[pnls > 0]
    losses = pnls[pnls < 0]
    pf = float(gains.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 0.0
    active = pnls[pnls != 0]
    wr = float((active > 0).sum() / len(active) * 100) if len(active) > 0 else 0.0

    return {
        "cumulative_pnl_pips": round(info["cumulative_pnl"], 1),
        "max_drawdown_pips": round(info["max_drawdown"], 1),
        "sharpe_ratio": round(sharpe, 4),
        "profit_factor": round(pf, 4),
        "num_trades": info["num_trades"],
        "win_rate": round(wr, 2),
        "n_hold": int((actions == 0).sum()),
        "n_buy": int((actions == 1).sum()),
        "n_sell": int((actions == 2).sum()),
    }


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  T08 -- RL : PPO ameliore avec 56 features")
    print("  500K timesteps | reward en pips | gamma=0.995")
    print("=" * 70)

    np.random.seed(SEED)

    print("\n  [1/5] Chargement des donnees...")
    train_feat, train_close, scaler, _ = load_and_prepare_data(
        str(FEATURES_DIR / "features_2022.parquet"), fit=True
    )
    val_feat, val_close, _, _ = load_and_prepare_data(
        str(FEATURES_DIR / "features_2023.parquet"), scaler=scaler
    )
    test_feat, test_close, _, _ = load_and_prepare_data(
        str(FEATURES_DIR / "features_2024.parquet"), scaler=scaler
    )

    print(f"\n  Train  : {len(train_feat)} steps")
    print(f"  Val    : {len(val_feat)} steps")
    print(f"  Test   : {len(test_feat)} steps")
    print(f"  State  : {OBS_DIM} dims")

    print("\n  [2/5] Creation des environnements...")
    train_env = TradingEnv(train_feat, train_close)
    val_env = TradingEnv(val_feat, val_close)
    test_env = TradingEnv(test_feat, test_close)

    print("\n  [3/5] Entrainement PPO...")
    print(f"  Timesteps : {TOTAL_TIMESTEPS:,}")

    model = PPO(
        "MlpPolicy", train_env,
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
        policy_kwargs={"net_arch": [128, 64]},
    )

    callback = TradingCallback(eval_freq=50_000)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)
    print("\n  Entrainement termine !")

    print("\n  [4/5] Evaluation...")

    train_env_eval = TradingEnv(train_feat, train_close)
    train_m = evaluate_model(model, train_env_eval)
    print(f"\n  === TRAIN (2022) ===")
    print(f"  PnL: {train_m['cumulative_pnl_pips']:+.1f} | Sharpe: {train_m['sharpe_ratio']:.4f} | Trades: {train_m['num_trades']}")

    val_m = evaluate_model(model, val_env)
    print(f"\n  === VALIDATION (2023) ===")
    print(f"  PnL: {val_m['cumulative_pnl_pips']:+.1f} | Sharpe: {val_m['sharpe_ratio']:.4f} | PF: {val_m['profit_factor']:.4f} | Trades: {val_m['num_trades']}")

    test_m = evaluate_model(model, test_env)
    print(f"\n  === TEST (2024) ===")
    print(f"  PnL: {test_m['cumulative_pnl_pips']:+.1f} | Sharpe: {test_m['sharpe_ratio']:.4f} | PF: {test_m['profit_factor']:.4f} | Trades: {test_m['num_trades']}")

    print("\n  [5/5] Sauvegarde...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "best_rl_model"
    model.save(str(model_path))
    print(f"  Modele : {model_path}.zip")

    scaler_path = MODELS_DIR / "scaler.joblib"
    joblib.dump(scaler, str(scaler_path))
    print(f"  Scaler : {scaler_path}")

    info = {
        "version": "v2", "type": "rl", "algorithm": "PPO",
        "policy": "MlpPolicy", "net_arch": [128, 64],
        "n_features": N_FEATURES, "observation_dim": OBS_DIM,
        "hyperparameters": PPO_PARAMS,
        "total_timesteps": TOTAL_TIMESTEPS,
        "transaction_cost_pips": TRANSACTION_COST_PIPS,
        "metrics_train": train_m,
        "metrics_validation": val_m,
        "metrics_test": test_m,
        "created_at": datetime.now().isoformat(),
    }
    with open(MODELS_DIR / "model_info.json", "w") as f:
        json.dump(info, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print(f"  T08 TERMINE")
    print(f"  Test 2024 : PnL={test_m['cumulative_pnl_pips']:+.1f} pips | Sharpe={test_m['sharpe_ratio']:.4f}")
    print(f"{'='*70}")

    return model, test_m


if __name__ == "__main__":
    main()
