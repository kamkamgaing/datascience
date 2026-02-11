"""
T09 -- Evaluation Finale : Comparaison de toutes les strategies + Stress Tests
================================================================================
Comparaison obligatoire :
  - Random
  - Regles (EMA Cross 20/50)
  - ML (meilleur modele)
  - RL (PPO)

Metriques : Profit cumule, Max Drawdown, Sharpe, Profit Factor

Stress Tests : impact du spread (1-5 pips)

Sorties :
  - evaluation/final_comparison.csv
  - evaluation/final_equity_curves.png
  - evaluation/final_stress_test.csv
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

FEATURES_DIR = os.path.join(PROJECT_ROOT, "features")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

PIP_VALUE = 0.0001
ANNUALIZATION = np.sqrt(252 * 96)


def compute_metrics(pnl_pips, label, spread=2):
    cumulative = np.cumsum(pnl_pips)
    total = float(cumulative[-1]) if len(cumulative) > 0 else 0.0
    running_max = np.maximum.accumulate(cumulative)
    dd = running_max - cumulative
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0

    active = pnl_pips[pnl_pips != 0]
    n_trades = len(active)

    if len(pnl_pips) > 1 and np.std(pnl_pips) > 0:
        sharpe = float(np.mean(pnl_pips) / np.std(pnl_pips) * ANNUALIZATION)
    else:
        sharpe = 0.0

    gains = active[active > 0] if len(active) > 0 else np.array([])
    losses = active[active < 0] if len(active) > 0 else np.array([])
    pf = float(gains.sum() / abs(losses.sum())) if len(losses) > 0 and abs(losses.sum()) > 0 else 0.0
    wr = float(len(gains) / len(active) * 100) if len(active) > 0 else 0.0

    return {
        "strategy": label,
        "profit_pips": round(total, 1),
        "max_dd_pips": round(max_dd, 1),
        "sharpe": round(sharpe, 4),
        "profit_factor": round(pf, 4),
        "n_trades": n_trades,
        "win_rate": round(wr, 1),
        "curve": cumulative,
    }


def backtest_signals(close, positions, spread=2):
    n = len(close)
    pnl = np.zeros(n - 1)
    for i in range(n - 1):
        p = (close[i + 1] - close[i]) / PIP_VALUE
        pnl[i] = positions[i] * p
        if i > 0 and positions[i] != positions[i - 1]:
            pnl[i] -= spread
    return pnl


def strat_random(close, spread=2, seed=42):
    rng = np.random.RandomState(seed)
    pos = rng.choice([-1, 0, 1], size=len(close))
    pnl = backtest_signals(close, pos, spread)
    return compute_metrics(pnl, "Random", spread)


def strat_ema_cross(close, spread=2):
    s = pd.Series(close)
    e20 = s.ewm(span=20, adjust=False).mean().values
    e50 = s.ewm(span=50, adjust=False).mean().values
    n = len(close)
    pos = np.zeros(n, dtype=int)
    for i in range(1, n):
        if e20[i] > e50[i]:
            pos[i] = 1
        elif e20[i] < e50[i]:
            pos[i] = -1
        else:
            pos[i] = pos[i - 1]
    pnl = backtest_signals(close, pos, spread)
    return compute_metrics(pnl, "Regles (EMA)", spread)


def strat_ml(close, spread=2):
    path = os.path.join(MODELS_DIR, "v1", "best_ml_model.pkl")
    obj = joblib.load(path)
    model = obj["model"]
    fc = obj["feature_cols"]
    sc = obj["scaler"]
    thr = obj.get("hold_threshold", 0.55)

    df = pd.read_parquet(os.path.join(FEATURES_DIR, "features_2024.parquet"))
    X = df[fc].values
    if sc is not None:
        X = sc.transform(X)
    X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
    proba = model.predict_proba(X)[:, 1]

    n = min(len(proba), len(close) - 1)
    pnl = np.zeros(n)
    for i in range(n):
        mv = close[i + 1] - close[i]
        if proba[i] >= thr:
            pnl[i] = (mv - spread * PIP_VALUE) / PIP_VALUE
        elif proba[i] <= (1 - thr):
            pnl[i] = (-mv - spread * PIP_VALUE) / PIP_VALUE
    return compute_metrics(pnl, "ML", spread)


def strat_rl(close, spread=2):
    from stable_baselines3 import PPO
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "training"))
    from t08_rl_training import TradingEnv, FEATURE_COLUMNS, load_and_prepare_data

    sc = joblib.load(os.path.join(MODELS_DIR, "v2", "scaler.joblib"))
    feat, closes, _, _ = load_and_prepare_data(
        os.path.join(FEATURES_DIR, "features_2024.parquet"), scaler=sc
    )
    m = PPO.load(os.path.join(MODELS_DIR, "v2", "best_rl_model.zip"))
    env = TradingEnv(feat, closes)

    obs, _ = env.reset()
    done = False
    pnls = []
    while not done:
        a, _ = m.predict(obs, deterministic=True)
        prev = env.cumulative_pnl + env.unrealized_pnl
        obs, _, term, trunc, info = env.step(int(a))
        done = term or trunc
        cur = env.cumulative_pnl + env.unrealized_pnl
        pnls.append(cur - prev)

    return compute_metrics(np.array(pnls), "RL (PPO)", spread)


def main():
    print("=" * 70)
    print("  T09 -- EVALUATION FINALE")
    print("  Comparaison des 4 strategies + Stress Tests | 2024")
    print("=" * 70)

    os.makedirs(EVAL_DIR, exist_ok=True)

    df_clean = pd.read_parquet(os.path.join(DATA_DIR, "m15_clean.parquet"))
    df_2024 = df_clean[df_clean["year"] == 2024].reset_index(drop=True)
    close = df_2024["close_15m"].values
    timestamps = pd.to_datetime(df_2024["timestamp"].values)
    print(f"\n  Bougies M15 2024 : {len(df_2024):,}")

    # --- Strategies ---
    print("\n  [1/3] Evaluation (spread=2 pips)...")
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "training"))

    results = []
    for name, func in [
        ("Random", lambda: strat_random(close)),
        ("EMA Cross", lambda: strat_ema_cross(close)),
        ("ML", lambda: strat_ml(close)),
        ("RL (PPO)", lambda: strat_rl(close)),
    ]:
        print(f"\n    {name}...")
        try:
            r = func()
            results.append(r)
            print(f"      Profit={r['profit_pips']:+.1f} | Sharpe={r['sharpe']:.4f} | Trades={r['n_trades']}")
        except Exception as e:
            print(f"      [ERREUR: {e}]")

    # --- Tableau ---
    print(f"\n{'='*90}")
    print("  TABLEAU COMPARATIF -- TEST 2024 (spread=2 pips)")
    print(f"{'='*90}")
    print(f"  {'Strategie':<20} {'Profit(pips)':>12} {'MaxDD(pips)':>12} {'Sharpe':>10} {'PF':>8} {'Trades':>8} {'WR':>7}")
    print(f"  {'-'*80}")
    for r in results:
        print(f"  {r['strategy']:<20} {r['profit_pips']:>+12.1f} {r['max_dd_pips']:>12.1f} {r['sharpe']:>10.4f} {r['profit_factor']:>8.4f} {r['n_trades']:>8} {r['win_rate']:>6.1f}%")

    best = max(results, key=lambda x: x["sharpe"])
    print(f"\n  >> MEILLEUR : {best['strategy']} (Sharpe = {best['sharpe']:.4f})")

    # --- Stress Test ---
    print(f"\n{'='*70}")
    print("  [2/3] STRESS TEST : Impact du spread")
    print(f"{'='*70}")

    spreads = [1, 2, 3, 4, 5]
    stress = []
    for sp in spreads:
        r_rand = strat_random(close, spread=sp)
        r_ema = strat_ema_cross(close, spread=sp)
        r_rl = strat_rl(close, spread=sp)
        stress.append({
            "spread": sp,
            "random_sharpe": r_rand["sharpe"],
            "ema_sharpe": r_ema["sharpe"],
            "rl_sharpe": r_rl["sharpe"],
            "rl_profit": r_rl["profit_pips"],
        })

    print(f"\n  {'Spread':>7} {'Random':>10} {'EMA':>10} {'RL':>10} {'RL PnL':>12}")
    print(f"  {'-'*52}")
    for s in stress:
        print(f"  {s['spread']:>5} pip {s['random_sharpe']:>10.4f} {s['ema_sharpe']:>10.4f} {s['rl_sharpe']:>10.4f} {s['rl_profit']:>+12.1f}")

    # --- Graphiques ---
    print(f"\n{'='*70}")
    print("  [3/3] GRAPHIQUES")
    print(f"{'='*70}")

    fig, axes = plt.subplots(2, 1, figsize=(18, 12), gridspec_kw={"height_ratios": [3, 1]})
    colors = {"Random": "#e74c3c", "Regles (EMA)": "#f39c12", "ML": "#3498db", "RL (PPO)": "#2ecc71"}

    ax = axes[0]
    for r in results:
        c = r["curve"]
        n = min(len(c), len(timestamps) - 1)
        ax.plot(timestamps[:n], c[:n],
                label=f"{r['strategy']} ({r['profit_pips']:+.0f} pips)",
                color=colors.get(r["strategy"], "#95a5a6"), linewidth=1.5, alpha=0.85)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_title("Evaluation Finale -- Equity Curves | GBP/USD M15 | 2024\nSpread = 2 pips",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Profit cumule (pips)", fontsize=12)
    ax.legend(fontsize=11, loc="best")
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    ax2 = axes[1]
    for key, label, color in [("rl_sharpe", "RL (PPO)", "#2ecc71"), ("ema_sharpe", "EMA Cross", "#f39c12")]:
        vals = [s[key] for s in stress]
        ax2.plot(spreads, vals, marker="o", label=label, color=color, linewidth=2)
    ax2.axhline(y=0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_xlabel("Spread (pips)", fontsize=12)
    ax2.set_ylabel("Sharpe Ratio", fontsize=12)
    ax2.set_title("Stress Test : Impact du spread", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(EVAL_DIR, "final_equity_curves.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> evaluation/final_equity_curves.png")

    # --- Save ---
    rows = [{k: v for k, v in r.items() if k != "curve"} for r in results]
    pd.DataFrame(rows).to_csv(os.path.join(EVAL_DIR, "final_comparison.csv"), index=False)
    pd.DataFrame(stress).to_csv(os.path.join(EVAL_DIR, "final_stress_test.csv"), index=False)
    with open(os.path.join(EVAL_DIR, "final_comparison.json"), "w") as f:
        json.dump(rows, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  T09 TERMINE -- Evaluation finale complete")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
