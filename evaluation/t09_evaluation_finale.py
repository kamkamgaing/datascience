"""
T09 -- Evaluation Finale : Comparaison de toutes les strategies sur 2024
=========================================================================
Comparaison obligatoire (cf. cahier des charges) :
  - Random
  - Regles (EMA Cross 20/50)
  - ML (meilleur modele)
  - RL (PPO)

Metriques :
  - Profit cumule (pips)
  - Maximum drawdown (pips)
  - Sharpe ratio simplifie
  - Profit factor

Sorties :
  - evaluation/final_comparison.csv
  - evaluation/final_equity_curves.png
  - evaluation/final_comparison_table.txt
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

# Ajouter le projet root au path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# =============================================================================
# Configuration
# =============================================================================
FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

SPREAD_PIPS = 2
PIP_VALUE = 0.0001
SPREAD_COST = SPREAD_PIPS * PIP_VALUE
ANNUALIZATION = np.sqrt(252 * 24 * 4)  # M15 annualization


# =============================================================================
# Utilitaires communs
# =============================================================================
def compute_metrics(pnl_per_bar, positions, label=""):
    """Calcule les 4 metriques obligatoires + extras."""
    cumulative = np.cumsum(pnl_per_bar)
    total_profit = cumulative[-1] if len(cumulative) > 0 else 0.0

    # Max Drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdown = running_max - cumulative
    max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0.0

    # Nombre de trades
    pos_changes = np.diff(positions)
    n_trades = int(np.count_nonzero(pos_changes))

    # Sharpe ratio
    if len(pnl_per_bar) > 1 and np.std(pnl_per_bar) > 0:
        sharpe = float(np.mean(pnl_per_bar) / np.std(pnl_per_bar) * ANNUALIZATION)
    else:
        sharpe = 0.0

    # Profit factor
    gains = pnl_per_bar[pnl_per_bar > 0]
    losses = pnl_per_bar[pnl_per_bar < 0]
    if len(losses) > 0 and abs(losses.sum()) > 0:
        pf = float(gains.sum() / abs(losses.sum()))
    else:
        pf = float("inf") if len(gains) > 0 else 0.0

    # Win rate
    active = pnl_per_bar[pnl_per_bar != 0]
    win_rate = float((active > 0).sum() / len(active) * 100) if len(active) > 0 else 0.0

    return {
        "strategy": label,
        "profit_cumule_pips": round(total_profit, 2),
        "max_drawdown_pips": round(max_dd, 2),
        "sharpe_ratio": round(sharpe, 4),
        "profit_factor": round(pf, 4),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 2),
        "cumulative_curve": cumulative,
    }


def backtest_signals(close, positions):
    """Backtest a partir de signaux de position (-1, 0, +1)."""
    n = len(close)
    pnl_per_bar = np.zeros(n - 1)

    for i in range(1, n):
        price_change = (close[i] - close[i - 1]) / PIP_VALUE
        pnl = positions[i - 1] * price_change
        if positions[i] != positions[i - 1]:
            pnl -= SPREAD_PIPS
        pnl_per_bar[i - 1] = pnl

    return pnl_per_bar, positions


# =============================================================================
# Strategie 1 : ALEATOIRE
# =============================================================================
def strategy_random(close, seed=42):
    """BUY/SELL/HOLD aleatoire avec proba 1/3."""
    rng = np.random.RandomState(seed)
    n = len(close)
    positions = rng.choice([-1, 0, 1], size=n)
    pnl, pos = backtest_signals(close, positions)
    return compute_metrics(pnl, pos, "Random")


# =============================================================================
# Strategie 2 : REGLES FIXES (EMA Cross 20/50)
# =============================================================================
def strategy_ema_cross(close):
    """EMA20 > EMA50 -> BUY, EMA20 < EMA50 -> SELL."""
    close_s = pd.Series(close)
    ema20 = close_s.ewm(span=20, adjust=False).mean().values
    ema50 = close_s.ewm(span=50, adjust=False).mean().values

    n = len(close)
    positions = np.zeros(n, dtype=int)
    for i in range(1, n):
        if ema20[i] > ema50[i]:
            positions[i] = 1
        elif ema20[i] < ema50[i]:
            positions[i] = -1
        else:
            positions[i] = positions[i - 1]

    pnl, pos = backtest_signals(close, positions)
    return compute_metrics(pnl, pos, "Regles (EMA Cross)")


# =============================================================================
# Strategie 3 : MACHINE LEARNING
# =============================================================================
def strategy_ml(close, timestamps):
    """Backtest du meilleur modele ML avec seuil HOLD optimise."""
    model_path = os.path.join(MODELS_DIR, "v1", "best_ml_model.pkl")
    model_obj = joblib.load(model_path)

    model = model_obj["model"]
    feature_cols = model_obj["feature_cols"]
    needs_scaling = model_obj["needs_scaling"]
    scaler = model_obj["scaler"]
    threshold = model_obj.get("hold_threshold", 0.55)

    # Charger les features 2024
    df_test = pd.read_parquet(os.path.join(FEATURES_DIR, "features_2024.parquet"))
    X_test = df_test[feature_cols].values

    if needs_scaling and scaler is not None:
        X_test = scaler.transform(X_test)

    y_proba = model.predict_proba(X_test)[:, 1]

    # Backtest avec HOLD
    n = len(close) - 1
    pnl_pips = np.zeros(n)
    positions = np.zeros(n + 1, dtype=int)

    for i in range(n):
        price_move = close[i + 1] - close[i]
        if y_proba[i] >= threshold:
            profit = (price_move - SPREAD_COST) / PIP_VALUE
            pnl_pips[i] = profit
            positions[i] = 1
        elif y_proba[i] <= (1 - threshold):
            profit = (-price_move - SPREAD_COST) / PIP_VALUE
            pnl_pips[i] = profit
            positions[i] = -1
        # else HOLD

    return compute_metrics(pnl_pips, positions, "ML (LogReg + HOLD)")


# =============================================================================
# Strategie 4 : REINFORCEMENT LEARNING (PPO)
# =============================================================================
def strategy_rl(close):
    """Evaluation du modele RL PPO sur 2024."""
    from stable_baselines3 import PPO
    from sklearn.preprocessing import StandardScaler as SS
    import gymnasium as gym
    from gymnasium import spaces

    # Importer TradingEnv depuis t08
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "training"))
    from t08_rl_training import TradingEnv, FEATURE_COLUMNS, load_and_prepare_data

    # Charger le scaler (ajuste sur 2022)
    scaler_path = os.path.join(MODELS_DIR, "v2", "scaler.joblib")
    scaler = joblib.load(scaler_path)

    # Charger et preparer les donnees 2024
    features_2024, closes_2024, _, df_2024 = load_and_prepare_data(
        os.path.join(FEATURES_DIR, "features_2024.parquet"),
        scaler=scaler,
        fit=False,
    )

    # Charger le modele PPO
    model_path = os.path.join(MODELS_DIR, "v2", "best_rl_model.zip")
    model = PPO.load(model_path)

    # Creer l'environnement de test
    env = TradingEnv(
        features=features_2024,
        closes=closes_2024,
        transaction_cost=SPREAD_COST,
    )

    # Evaluer
    obs, info = env.reset()
    done = False
    step_pnls = []
    positions_list = [0]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        prev_pnl = env.cumulative_pnl + env.unrealized_pnl
        obs, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        current_pnl = env.cumulative_pnl + env.unrealized_pnl
        step_pnls.append(current_pnl - prev_pnl)
        positions_list.append(env.position)

    # Convertir le PnL en pips
    step_pnls_pips = np.array(step_pnls) / PIP_VALUE
    positions = np.array(positions_list)

    return compute_metrics(step_pnls_pips, positions, "RL (PPO)")


# =============================================================================
# Graphiques
# =============================================================================
def plot_comparison(results, timestamps, output_path):
    """Equity curves comparatives de toutes les strategies."""
    fig, ax = plt.subplots(figsize=(18, 8))

    colors = {
        "Random": "#e74c3c",
        "Regles (EMA Cross)": "#f39c12",
        "ML (LogReg + HOLD)": "#3498db",
        "RL (PPO)": "#2ecc71",
    }

    for r in results:
        name = r["strategy"]
        curve = r["cumulative_curve"]
        n = min(len(curve), len(timestamps) - 1)
        ts = timestamps[:n]
        c = curve[:n]
        ax.plot(ts, c, label=f"{name} ({r['profit_cumule_pips']:+.0f} pips)",
                color=colors.get(name, "#95a5a6"), linewidth=1.5, alpha=0.9)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(
        "Evaluation Finale -- Comparaison des 4 strategies sur 2024\n"
        "GBP/USD M15 | Spread = 2 pips",
        fontsize=14, fontweight="bold", pad=15
    )
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Profit cumule (pips)", fontsize=12)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {output_path}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 70)
    print("  T09 -- EVALUATION FINALE : COMPARAISON DES 4 STRATEGIES")
    print("  GBP/USD M15 -- Test 2024")
    print("=" * 70)

    os.makedirs(EVAL_DIR, exist_ok=True)

    # Charger les donnees 2024
    print("\n  --- Chargement des donnees 2024 ---")
    df_clean = pd.read_parquet(os.path.join(DATA_DIR, "m15_clean.parquet"))
    df_2024 = df_clean[df_clean["year"] == 2024].reset_index(drop=True)
    close_2024 = df_2024["close_15m"].values
    timestamps = pd.to_datetime(df_2024["timestamp"].values)
    print(f"  Bougies M15 2024 : {len(df_2024):,}")

    # --- Strategie 1 : Random ---
    print("\n  [1/4] Strategie Random...")
    r_random = strategy_random(close_2024)
    print(f"    Profit: {r_random['profit_cumule_pips']:+.2f} pips | "
          f"Sharpe: {r_random['sharpe_ratio']:.4f}")

    # --- Strategie 2 : Regles (EMA Cross) ---
    print("\n  [2/4] Strategie Regles (EMA Cross 20/50)...")
    r_rules = strategy_ema_cross(close_2024)
    print(f"    Profit: {r_rules['profit_cumule_pips']:+.2f} pips | "
          f"Sharpe: {r_rules['sharpe_ratio']:.4f}")

    # --- Strategie 3 : ML ---
    print("\n  [3/4] Strategie ML...")
    r_ml = strategy_ml(close_2024, timestamps)
    print(f"    Profit: {r_ml['profit_cumule_pips']:+.2f} pips | "
          f"Sharpe: {r_ml['sharpe_ratio']:.4f}")

    # --- Strategie 4 : RL ---
    print("\n  [4/4] Strategie RL (PPO)...")
    r_rl = strategy_rl(close_2024)
    print(f"    Profit: {r_rl['profit_cumule_pips']:+.2f} pips | "
          f"Sharpe: {r_rl['sharpe_ratio']:.4f}")

    # --- Tableau comparatif ---
    results = [r_random, r_rules, r_ml, r_rl]

    print(f"\n{'=' * 90}")
    print("  TABLEAU COMPARATIF FINAL -- TEST 2024")
    print(f"{'=' * 90}")
    header = (f"  {'Strategie':<22} {'Profit(pips)':>14} {'MaxDD(pips)':>12} "
              f"{'Sharpe':>10} {'PF':>10} {'Trades':>8} {'WinRate':>8}")
    print(header)
    print("  " + "-" * 86)

    for r in results:
        print(f"  {r['strategy']:<22} "
              f"{r['profit_cumule_pips']:>+14.2f} "
              f"{r['max_drawdown_pips']:>12.2f} "
              f"{r['sharpe_ratio']:>10.4f} "
              f"{r['profit_factor']:>10.4f} "
              f"{r['n_trades']:>8} "
              f"{r['win_rate']:>7.1f}%")

    # Identifier le meilleur modele
    best = max(results, key=lambda x: x["sharpe_ratio"])
    print(f"\n  >> Meilleur modele (Sharpe) : {best['strategy']} "
          f"(Sharpe = {best['sharpe_ratio']:.4f})")

    # Robustesse : le modele est valide s'il bat le random
    for r in results[2:]:  # ML et RL
        if r["sharpe_ratio"] > r_random["sharpe_ratio"]:
            print(f"  >> {r['strategy']} : ROBUSTE sur 2024 (bat le random)")
        else:
            print(f"  >> {r['strategy']} : NON ROBUSTE sur 2024")

    # --- Sauvegarde CSV ---
    rows = []
    for r in results:
        rows.append({
            "strategy": r["strategy"],
            "profit_cumule_pips": r["profit_cumule_pips"],
            "max_drawdown_pips": r["max_drawdown_pips"],
            "sharpe_ratio": r["sharpe_ratio"],
            "profit_factor": r["profit_factor"],
            "n_trades": r["n_trades"],
            "win_rate": r["win_rate"],
        })
    df_results = pd.DataFrame(rows)
    csv_path = os.path.join(EVAL_DIR, "final_comparison.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"\n  -> Resultats CSV : {csv_path}")

    # --- Sauvegarde JSON ---
    json_path = os.path.join(EVAL_DIR, "final_comparison.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  -> Resultats JSON : {json_path}")

    # --- Graphique comparatif ---
    print("\n  --- Graphiques ---")
    plot_comparison(results, timestamps, os.path.join(EVAL_DIR, "final_equity_curves.png"))

    print(f"\n{'=' * 70}")
    print(f"  T09 TERMINE -- Evaluation finale complete")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
