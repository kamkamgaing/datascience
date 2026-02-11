"""
T06 — Stratégies Baseline
============================
3 stratégies de référence pour évaluer les futurs modèles ML.

Stratégies :
  a) Aléatoire (BUY/SELL/HOLD, prob 1/3 chacune, seed=42)
  b) Buy & Hold
  c) Croisement EMA 20/50

Métriques par stratégie et par année :
  - Profit cumulé (en pips)
  - Maximum drawdown
  - Sharpe ratio simplifié = mean(returns) / std(returns) * sqrt(252*24*4)
  - Profit factor = sum(gains) / abs(sum(pertes))
  - Nombre de trades
  - Win rate

Sorties :
  - evaluation/baseline_results.csv
  - evaluation/baseline_equity_curves.png
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
INPUT_PATH = "data/m15_clean.parquet"
OUTPUT_DIR = "evaluation"
SPREAD_PIPS = 2        # coût de transaction en pips
PIP_VALUE = 0.0001     # 1 pip = 0.0001 pour GBP/USD
SPREAD_COST = SPREAD_PIPS * PIP_VALUE  # 0.0002

# Sharpe : sqrt(nombre de périodes M15 par an)
# 252 jours de trading × 24h × 4 bougies/h = 24 192 bougies M15/an
ANNUALIZATION_FACTOR = np.sqrt(252 * 24 * 4)


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITAIRES
# ═══════════════════════════════════════════════════════════════════════════════

def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=span, adjust=False).mean()


def compute_metrics(equity_curve: np.ndarray, positions: np.ndarray,
                    pnl_per_bar: np.ndarray) -> dict:
    """
    Calcule les métriques de performance à partir de la courbe d'equity.

    Parameters
    ----------
    equity_curve : array, profit cumulé en pips à chaque barre
    positions : array, position à chaque barre (-1, 0, +1)
    pnl_per_bar : array, P&L par barre en pips (après spread)
    """
    # Profit cumulé
    total_profit = equity_curve[-1] if len(equity_curve) > 0 else 0.0

    # Maximum Drawdown (en pips)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = running_max - equity_curve
    max_drawdown = drawdown.max() if len(drawdown) > 0 else 0.0

    # Nombre de trades (changements de position)
    pos_changes = np.diff(positions)
    n_trades = int(np.count_nonzero(pos_changes))

    # Séparer gains et pertes (par barre)
    gains = pnl_per_bar[pnl_per_bar > 0]
    losses = pnl_per_bar[pnl_per_bar < 0]

    # Win rate (sur les barres avec position non nulle)
    active_bars = pnl_per_bar[positions[1:] != 0] if len(positions) > 1 else pnl_per_bar
    n_active = len(active_bars)
    n_wins = int((active_bars > 0).sum()) if n_active > 0 else 0
    win_rate = n_wins / n_active if n_active > 0 else 0.0

    # Sharpe ratio simplifié
    if len(pnl_per_bar) > 1 and pnl_per_bar.std() > 0:
        sharpe = (pnl_per_bar.mean() / pnl_per_bar.std()) * ANNUALIZATION_FACTOR
    else:
        sharpe = 0.0

    # Profit factor
    sum_gains = gains.sum() if len(gains) > 0 else 0.0
    sum_losses = abs(losses.sum()) if len(losses) > 0 else 0.0
    profit_factor = sum_gains / sum_losses if sum_losses > 0 else np.inf

    return {
        "profit_cumule_pips": round(total_profit, 2),
        "max_drawdown_pips": round(max_drawdown, 2),
        "sharpe_ratio": round(sharpe, 4),
        "profit_factor": round(profit_factor, 4),
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STRATÉGIE A : ALÉATOIRE
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_random(close: np.ndarray, seed: int = 42) -> tuple:
    """
    À chaque barre : BUY (+1), SELL (-1), ou HOLD (0) avec proba 1/3.
    """
    rng = np.random.RandomState(seed)
    n = len(close)
    signals = rng.choice([-1, 0, 1], size=n)

    return backtest(close, signals)


# ═══════════════════════════════════════════════════════════════════════════════
# STRATÉGIE B : BUY & HOLD
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_buy_hold(close: np.ndarray) -> tuple:
    """
    Acheter au début (position +1 constante), vendre à la fin.
    Un seul trade → un seul spread payé.
    """
    n = len(close)
    # Position constante +1 (long)
    signals = np.ones(n, dtype=int)

    return backtest(close, signals)


# ═══════════════════════════════════════════════════════════════════════════════
# STRATÉGIE C : CROISEMENT EMA 20/50
# ═══════════════════════════════════════════════════════════════════════════════

def strategy_ema_cross(close: np.ndarray) -> tuple:
    """
    EMA20 > EMA50 → BUY (+1)
    EMA20 < EMA50 → SELL (-1)
    EMA20 == EMA50 → HOLD (0) (quasi impossible en pratique)
    """
    close_series = pd.Series(close)
    ema_20 = ema(close_series, 20).values
    ema_50 = ema(close_series, 50).values

    n = len(close)
    signals = np.zeros(n, dtype=int)

    for i in range(1, n):
        if ema_20[i] > ema_50[i]:
            signals[i] = 1   # BUY
        elif ema_20[i] < ema_50[i]:
            signals[i] = -1  # SELL
        else:
            signals[i] = signals[i - 1]  # HOLD (garder la position)

    return backtest(close, signals)


# ═══════════════════════════════════════════════════════════════════════════════
# MOTEUR DE BACKTEST
# ═══════════════════════════════════════════════════════════════════════════════

def backtest(close: np.ndarray, positions: np.ndarray) -> tuple:
    """
    Simule un backtest simple.

    Parameters
    ----------
    close : array de prix de clôture
    positions : array de signaux (-1, 0, +1) à chaque barre

    Returns
    -------
    equity_curve : array, profit cumulé en pips
    positions : array, positions à chaque barre
    pnl_per_bar : array, P&L par barre en pips
    """
    n = len(close)
    pnl_per_bar = np.zeros(n - 1)

    for i in range(1, n):
        # P&L de la position en cours (en pips)
        price_change = (close[i] - close[i - 1]) / PIP_VALUE
        pnl = positions[i - 1] * price_change

        # Coût de transaction si changement de position
        if positions[i] != positions[i - 1]:
            pnl -= SPREAD_PIPS  # spread en pips

        pnl_per_bar[i - 1] = pnl

    equity_curve = np.cumsum(pnl_per_bar)
    # Préfixer avec 0 pour que la courbe commence à 0
    equity_curve = np.concatenate([[0.0], equity_curve])

    return equity_curve, positions, pnl_per_bar


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  T06 — STRATÉGIES BASELINE")
    print("  GBP/USD M15 — 2022-2024")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Charger les données
    print(f"\n  Chargement : {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Bougies M15 : {len(df):,}")

    years = sorted(df["year"].unique())
    strategies = {
        "Aléatoire": strategy_random,
        "Buy & Hold": strategy_buy_hold,
        "EMA Cross 20/50": strategy_ema_cross,
    }

    all_results = []
    equity_data = {}  # {(strategy, year): equity_curve}

    for year in years:
        df_year = df[df["year"] == year].reset_index(drop=True)
        close = df_year["close_15m"].values

        print(f"\n  {'─'*60}")
        print(f"  Année {year} — {len(df_year):,} bougies")
        print(f"  {'─'*60}")

        for strat_name, strat_func in strategies.items():
            # Exécuter la stratégie
            equity_curve, positions, pnl_per_bar = strat_func(close)

            # Calculer les métriques
            metrics = compute_metrics(equity_curve, positions, pnl_per_bar)
            metrics["strategy"] = strat_name
            metrics["year"] = year

            all_results.append(metrics)
            equity_data[(strat_name, year)] = equity_curve

            print(f"\n    {strat_name}:")
            print(f"      Profit cumulé  : {metrics['profit_cumule_pips']:>10.2f} pips")
            print(f"      Max Drawdown   : {metrics['max_drawdown_pips']:>10.2f} pips")
            print(f"      Sharpe Ratio   : {metrics['sharpe_ratio']:>10.4f}")
            print(f"      Profit Factor  : {metrics['profit_factor']:>10.4f}")
            print(f"      Trades         : {metrics['n_trades']:>10d}")
            print(f"      Win Rate       : {metrics['win_rate']:>10.2%}")

    # ─── Sauvegarder les résultats en CSV ─────────────────────────────────────
    results_df = pd.DataFrame(all_results)
    # Réordonner les colonnes
    cols = ["strategy", "year", "profit_cumule_pips", "max_drawdown_pips",
            "sharpe_ratio", "profit_factor", "n_trades", "win_rate"]
    results_df = results_df[cols]

    csv_path = os.path.join(OUTPUT_DIR, "baseline_results.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"\n  → Résultats sauvegardés : {csv_path}")

    # ─── Equity Curves ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, len(years), figsize=(18, 5), sharey=False)
    if len(years) == 1:
        axes = [axes]

    colors = {
        "Aléatoire": "#e74c3c",
        "Buy & Hold": "#2ecc71",
        "EMA Cross 20/50": "#3498db",
    }

    for idx, year in enumerate(years):
        ax = axes[idx]
        df_year = df[df["year"] == year].reset_index(drop=True)

        for strat_name in strategies.keys():
            ec = equity_data[(strat_name, year)]
            ax.plot(ec, label=strat_name, color=colors[strat_name],
                    linewidth=1.2, alpha=0.9)

        ax.set_title(f"Equity Curves — {year}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Bougies M15")
        ax.set_ylabel("Profit cumulé (pips)")
        ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--")
        ax.legend(fontsize=8, loc="best")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    eq_path = os.path.join(OUTPUT_DIR, "baseline_equity_curves.png")
    plt.savefig(eq_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"  → Equity curves sauvegardées : {eq_path}")

    # ─── Tableau comparatif final ────────────────────────────────────────────
    print(f"\n{'='*90}")
    print("  TABLEAU COMPARATIF — BASELINES")
    print(f"{'='*90}")

    header = (f"  {'Stratégie':<20} {'Année':>6} {'Profit(pips)':>14} "
              f"{'MaxDD(pips)':>12} {'Sharpe':>10} {'PF':>10} "
              f"{'Trades':>8} {'WinRate':>10}")
    print(header)
    print("  " + "─" * 86)

    for _, row in results_df.iterrows():
        line = (f"  {row['strategy']:<20} {row['year']:>6} "
                f"{row['profit_cumule_pips']:>14.2f} "
                f"{row['max_drawdown_pips']:>12.2f} "
                f"{row['sharpe_ratio']:>10.4f} "
                f"{row['profit_factor']:>10.4f} "
                f"{row['n_trades']:>8} "
                f"{row['win_rate']:>10.2%}")
        print(line)

    # Résumé global (moyenne sur les 3 années)
    print(f"\n  {'─'*86}")
    print(f"  {'MOYENNE 3 ANS':}")
    print(f"  {'─'*86}")

    for strat_name in strategies.keys():
        sub = results_df[results_df["strategy"] == strat_name]
        print(f"  {strat_name:<20} "
              f"{'MOY':>6} "
              f"{sub['profit_cumule_pips'].mean():>14.2f} "
              f"{sub['max_drawdown_pips'].mean():>12.2f} "
              f"{sub['sharpe_ratio'].mean():>10.4f} "
              f"{sub['profit_factor'].mean():>10.4f} "
              f"{sub['n_trades'].mean():>8.0f} "
              f"{sub['win_rate'].mean():>10.2%}")

    print(f"\n{'='*90}")
    print("  T06 TERMINE")
    print(f"  Fichiers générés dans {OUTPUT_DIR}/ :")
    print("    - baseline_results.csv")
    print("    - baseline_equity_curves.png")
    print(f"{'='*90}")


if __name__ == "__main__":
    main()
