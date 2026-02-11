"""
T09 — Évaluation Finale ML + Backtest (CORRIGÉ)
==================================================
Corrections :
  - Ajout du mécanisme HOLD (seuil de confiance)
  - Optimisation du seuil sur validation avant test
  - Métriques financières réalistes
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings("ignore")

FEATURES_DIR = "features"
MODEL_PATH = "models/v1/best_ml_model.pkl"
EVAL_DIR = "evaluation"

SPREAD_PIPS = 2
PIP_VALUE = 0.0001
ANNUALIZATION = np.sqrt(252 * 96)  # 96 bougies M15 par jour


def load_model_and_data():
    print("\n  --- Chargement du modèle ---")
    model_obj = joblib.load(MODEL_PATH)
    model = model_obj["model"]
    model_name = model_obj["model_name"]
    feature_cols = model_obj["feature_cols"]
    needs_scaling = model_obj["needs_scaling"]
    scaler = model_obj["scaler"]
    threshold = model_obj.get("hold_threshold", 0.55)

    print(f"  Modèle : {model_name}")
    print(f"  Features : {len(feature_cols)}")
    print(f"  Seuil HOLD : {threshold}")

    df_val = pd.read_parquet(os.path.join(FEATURES_DIR, "features_2023.parquet"))
    df_test = pd.read_parquet(os.path.join(FEATURES_DIR, "features_2024.parquet"))
    print(f"  Validation (2023) : {len(df_val):,} lignes")
    print(f"  Test (2024) : {len(df_test):,} lignes")

    return model, model_name, feature_cols, needs_scaling, scaler, threshold, df_val, df_test


def backtest(y_proba, close_prices, threshold, timestamps):
    """Backtest avec BUY/SELL/HOLD."""
    n = len(close_prices) - 1
    pnl_pips = np.zeros(n)
    actions = np.zeros(n, dtype=int)  # 0=HOLD, 1=BUY, 2=SELL
    n_trades = 0

    for i in range(n):
        price_move = close_prices[i + 1] - close_prices[i]

        if y_proba[i] >= threshold:
            # BUY
            profit = (price_move - SPREAD_PIPS * PIP_VALUE) / PIP_VALUE
            pnl_pips[i] = profit
            actions[i] = 1
            n_trades += 1
        elif y_proba[i] <= (1 - threshold):
            # SELL
            profit = (-price_move - SPREAD_PIPS * PIP_VALUE) / PIP_VALUE
            pnl_pips[i] = profit
            actions[i] = 2
            n_trades += 1
        # else: HOLD → pnl = 0

    cumulative = np.cumsum(pnl_pips)
    total_profit = float(cumulative[-1]) if len(cumulative) > 0 else 0.0

    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = float(drawdown.min()) if len(drawdown) > 0 else 0.0

    trading_pnls = pnl_pips[pnl_pips != 0]
    if len(trading_pnls) > 1 and np.std(trading_pnls) > 0:
        sharpe = float(np.mean(trading_pnls) / np.std(trading_pnls) * ANNUALIZATION)
    else:
        sharpe = 0.0

    gains = trading_pnls[trading_pnls > 0]
    losses = trading_pnls[trading_pnls < 0]
    pf = float(gains.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 0.0
    win_rate = float(len(gains) / n_trades * 100) if n_trades > 0 else 0.0

    return {
        "total_profit_pips": round(total_profit, 1),
        "max_drawdown_pips": round(max_dd, 1),
        "sharpe_ratio": round(sharpe, 4),
        "profit_factor": round(pf, 4),
        "n_trades": n_trades,
        "n_buy": int((actions == 1).sum()),
        "n_sell": int((actions == 2).sum()),
        "n_hold": int((actions == 0).sum()),
        "win_rate": round(win_rate, 2),
        "hold_pct": round((1 - n_trades / n) * 100, 1) if n > 0 else 0,
        "avg_win": round(float(gains.mean()), 2) if len(gains) > 0 else 0,
        "avg_loss": round(float(losses.mean()), 2) if len(losses) > 0 else 0,
        "cumulative_profit": cumulative,
        "drawdown": drawdown,
        "timestamps": timestamps[:n],
    }


def optimize_threshold(model, X_val, close_val, timestamps_val, needs_scaling, scaler):
    """Trouve le meilleur seuil sur validation."""
    print("\n  --- Optimisation du seuil HOLD sur validation ---")

    if needs_scaling and scaler is not None:
        X = scaler.transform(X_val)
    else:
        X = X_val

    y_proba = model.predict_proba(X)[:, 1]

    best_sharpe = -999
    best_threshold = 0.55
    results_by_threshold = []

    for thr in np.arange(0.52, 0.65, 0.01):
        bt = backtest(y_proba, close_val, thr, timestamps_val)
        results_by_threshold.append({
            "threshold": round(thr, 2),
            "profit": bt["total_profit_pips"],
            "sharpe": bt["sharpe_ratio"],
            "n_trades": bt["n_trades"],
            "win_rate": bt["win_rate"],
        })
        if bt["sharpe_ratio"] > best_sharpe and bt["n_trades"] >= 50:
            best_sharpe = bt["sharpe_ratio"]
            best_threshold = round(thr, 2)

    print(f"  {'Seuil':>7} {'Profit':>10} {'Sharpe':>8} {'Trades':>8} {'WR':>6}")
    print(f"  {'─'*42}")
    for r in results_by_threshold:
        marker = " <-- BEST" if r["threshold"] == best_threshold else ""
        print(f"  {r['threshold']:>7.2f} {r['profit']:>+10.1f} {r['sharpe']:>8.4f} {r['n_trades']:>8} {r['win_rate']:>5.1f}%{marker}")

    print(f"\n  Seuil optimal : {best_threshold} (Sharpe = {best_sharpe:.4f})")
    return best_threshold


def plot_equity_curve(bt_result, model_name, output_path):
    fig, ax = plt.subplots(figsize=(16, 7))
    ts = bt_result["timestamps"]
    cum = bt_result["cumulative_profit"]

    ax.plot(ts, cum, color="#3498db", linewidth=1.0, alpha=0.9)
    ax.fill_between(ts, cum, 0, where=(cum >= 0), color="#2ecc71", alpha=0.15, label="Profit")
    ax.fill_between(ts, cum, 0, where=(cum < 0), color="#e74c3c", alpha=0.15, label="Perte")
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    ax.set_title(
        f"Equity Curve — {model_name} | GBP/USD M15 | 2024\n"
        f"Profit: {bt_result['total_profit_pips']:+.1f} pips | "
        f"Sharpe: {bt_result['sharpe_ratio']:.4f} | "
        f"Trades: {bt_result['n_trades']:,} | "
        f"WR: {bt_result['win_rate']:.1f}% | "
        f"HOLD: {bt_result['hold_pct']:.1f}%",
        fontsize=12, fontweight="bold", pad=15
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Profit cumulé (pips)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


def plot_drawdown(bt_result, model_name, output_path):
    fig, ax = plt.subplots(figsize=(16, 5))
    ts = bt_result["timestamps"]
    dd = bt_result["drawdown"]

    ax.fill_between(ts, dd, 0, color="#e74c3c", alpha=0.4)
    ax.plot(ts, dd, color="#c0392b", linewidth=0.8, alpha=0.8)
    ax.axhline(y=bt_result["max_drawdown_pips"], color="#8e44ad", linestyle="--",
               linewidth=1.0, alpha=0.7, label=f"Max DD: {bt_result['max_drawdown_pips']:.1f} pips")

    ax.set_title(f"Drawdown — {model_name} | 2024 | Max DD: {bt_result['max_drawdown_pips']:.1f} pips",
                 fontsize=12, fontweight="bold", pad=15)
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Drawdown (pips)", fontsize=11)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    plt.xticks(rotation=45)
    ax.legend(loc="lower left", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {output_path}")


def main():
    print("=" * 70)
    print("  T09 — ÉVALUATION FINALE ML + BACKTEST (CORRIGÉ)")
    print("=" * 70)

    os.makedirs(EVAL_DIR, exist_ok=True)

    model, model_name, feature_cols, needs_scaling, scaler, default_threshold, df_val, df_test = load_model_and_data()

    X_val = df_val[feature_cols].values
    X_test = df_test[feature_cols].values
    close_val = df_val["close_15m"].values
    close_test = df_test["close_15m"].values
    ts_val = pd.to_datetime(df_val["timestamp"].values)
    ts_test = pd.to_datetime(df_test["timestamp"].values)

    # Optimiser le seuil sur validation
    best_threshold = optimize_threshold(model, X_val, close_val, ts_val, needs_scaling, scaler)

    # Backtest sur TEST 2024 avec seuil optimisé
    print("\n" + "=" * 70)
    print(f"  BACKTEST FINAL SUR TEST (2024) — seuil = {best_threshold}")
    print("=" * 70)

    if needs_scaling and scaler is not None:
        X_test_scaled = scaler.transform(X_test)
    else:
        X_test_scaled = X_test

    y_proba_test = model.predict_proba(X_test_scaled)[:, 1]
    bt = backtest(y_proba_test, close_test, best_threshold, ts_test)

    print(f"\n  {'═'*55}")
    print(f"  RÉSULTATS — {model_name} sur 2024")
    print(f"  {'═'*55}")
    print(f"  Seuil HOLD     : {best_threshold}")
    print(f"  Trades BUY     : {bt['n_buy']:,}")
    print(f"  Trades SELL    : {bt['n_sell']:,}")
    print(f"  HOLD           : {bt['n_hold']:,} ({bt['hold_pct']:.1f}%)")
    print(f"  ─────────────────────────────────────")
    print(f"  Profit cumulé  : {bt['total_profit_pips']:+.1f} pips")
    print(f"  Max drawdown   : {bt['max_drawdown_pips']:.1f} pips")
    print(f"  Sharpe ratio   : {bt['sharpe_ratio']:.4f}")
    print(f"  Profit factor  : {bt['profit_factor']:.4f}")
    print(f"  Win rate       : {bt['win_rate']:.1f}%")
    print(f"  Gain moyen     : {bt['avg_win']:+.2f} pips")
    print(f"  Perte moyenne  : {bt['avg_loss']:.2f} pips")
    print(f"  {'═'*55}")

    # Sauvegarde
    results_save = {k: v for k, v in bt.items() if k not in ("cumulative_profit", "drawdown", "timestamps")}
    results_save["model_name"] = model_name
    results_save["period"] = "2024"
    results_save["threshold"] = best_threshold
    results_save["spread_pips"] = SPREAD_PIPS
    results_save["date_evaluated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(os.path.join(EVAL_DIR, "ml_backtest_results.json"), "w") as f:
        json.dump(results_save, f, indent=2)

    print("\n  --- Graphiques ---")
    plot_equity_curve(bt, model_name, os.path.join(EVAL_DIR, "ml_equity_curve.png"))
    plot_drawdown(bt, model_name, os.path.join(EVAL_DIR, "ml_drawdown.png"))

    print(f"\n{'='*70}")
    print(f"  T09 TERMINE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
