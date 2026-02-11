"""
T07 -- Machine Learning V2 (AMELIORE)
=======================================
Ameliorations par rapport a V1 :
  - 56 features V3 (price action, regime, micro-structure)
  - Hyperparametres optimises pour chaque modele
  - Selection du seuil HOLD sur validation (Sharpe)
  - Walk-forward validation (4 fenetres glissantes)
  - Entrainement final sur 2022+2023 pour le meilleur modele
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================
FEATURES_DIR = "features"
MODELS_DIR = "models/v1"
EVAL_DIR = "evaluation"

SPREAD_PIPS = 2
PIP_VALUE = 0.0001
ANNUALIZATION = np.sqrt(252 * 96)

# 56 features V3
FEATURE_COLS = [
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


# =============================================================================
# Backtest avec seuil de confiance
# =============================================================================
def backtest_proba(y_proba, close_prices, threshold=0.55):
    n = min(len(y_proba), len(close_prices) - 1)
    pnl_pips = np.zeros(n)
    n_trades = 0

    for i in range(n):
        price_move = close_prices[i + 1] - close_prices[i]
        if y_proba[i] >= threshold:
            pnl_pips[i] = (price_move - SPREAD_PIPS * PIP_VALUE) / PIP_VALUE
            n_trades += 1
        elif y_proba[i] <= (1 - threshold):
            pnl_pips[i] = (-price_move - SPREAD_PIPS * PIP_VALUE) / PIP_VALUE
            n_trades += 1

    cumulative = np.cumsum(pnl_pips)
    total = float(cumulative[-1]) if len(cumulative) > 0 else 0.0
    running_max = np.maximum.accumulate(cumulative)
    dd = cumulative - running_max
    max_dd = float(dd.min()) if len(dd) > 0 else 0.0
    trading_pnls = pnl_pips[pnl_pips != 0]
    if len(trading_pnls) > 1 and np.std(trading_pnls) > 0:
        sharpe = float(np.mean(trading_pnls) / np.std(trading_pnls) * ANNUALIZATION)
    else:
        sharpe = 0.0
    gains = trading_pnls[trading_pnls > 0]
    losses = trading_pnls[trading_pnls < 0]
    pf = float(gains.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else 0.0
    wr = float(len(gains) / n_trades * 100) if n_trades > 0 else 0.0

    return {
        "total_profit_pips": round(total, 1),
        "max_drawdown_pips": round(max_dd, 1),
        "sharpe_ratio": round(sharpe, 4),
        "profit_factor": round(pf, 4),
        "n_trades": n_trades,
        "win_rate": round(wr, 2),
        "hold_pct": round((1 - n_trades / max(n, 1)) * 100, 1),
    }


def optimize_threshold(y_proba, close_prices, min_trades=30):
    best_sharpe = -999
    best_thr = 0.55
    for thr in np.arange(0.51, 0.70, 0.01):
        bt = backtest_proba(y_proba, close_prices, thr)
        if bt["sharpe_ratio"] > best_sharpe and bt["n_trades"] >= min_trades:
            best_sharpe = bt["sharpe_ratio"]
            best_thr = round(thr, 2)
    return best_thr, best_sharpe


def get_models():
    return {
        "LogReg": LogisticRegression(
            C=0.05, max_iter=2000, class_weight="balanced", random_state=42
        ),
        "RF": RandomForestClassifier(
            n_estimators=500, max_depth=6, min_samples_leaf=100,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "XGB": XGBClassifier(
            n_estimators=500, max_depth=4, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.6,
            reg_alpha=0.5, reg_lambda=2.0, min_child_weight=50,
            random_state=42, verbosity=0, use_label_encoder=False,
            eval_metric="logloss"
        ),
        "LGBM": LGBMClassifier(
            n_estimators=500, max_depth=5, learning_rate=0.03,
            subsample=0.7, colsample_bytree=0.6,
            reg_alpha=0.5, reg_lambda=2.0, min_child_weight=50,
            class_weight="balanced", random_state=42, verbose=-1
        ),
        "GB": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.03,
            subsample=0.7, min_samples_leaf=100,
            random_state=42
        ),
    }


def walk_forward_validation(df_all, feature_cols):
    df_all = df_all.copy()
    df_all["timestamp"] = pd.to_datetime(df_all["timestamp"])

    splits = [
        ("WF1: 2022H1->2022H2", "2022-01-01", "2022-07-01", "2022-07-01", "2023-01-01"),
        ("WF2: 2022->2023H1",   "2022-01-01", "2023-01-01", "2023-01-01", "2023-07-01"),
        ("WF3: 2022-23H1->23H2","2022-01-01", "2023-07-01", "2023-07-01", "2024-01-01"),
        ("WF4: 2022-23->2024H1","2022-01-01", "2024-01-01", "2024-01-01", "2024-07-01"),
    ]

    results = []
    for name, ts, te, vs, ve in splits:
        df_t = df_all[(df_all["timestamp"] >= ts) & (df_all["timestamp"] < te)]
        df_v = df_all[(df_all["timestamp"] >= vs) & (df_all["timestamp"] < ve)]
        if len(df_t) < 100 or len(df_v) < 100:
            continue

        X_t = df_t[feature_cols].values
        y_t = df_t["target"].values
        X_v = df_v[feature_cols].values
        close_v = df_v["close_15m"].values

        sc = StandardScaler()
        X_t_s = np.nan_to_num(sc.fit_transform(X_t), nan=0, posinf=0, neginf=0)
        X_v_s = np.nan_to_num(sc.transform(X_v), nan=0, posinf=0, neginf=0)

        for mname, model in get_models().items():
            model.fit(X_t_s, y_t)
            proba = model.predict_proba(X_v_s)[:, 1]
            thr, _ = optimize_threshold(proba, close_v, min_trades=20)
            bt = backtest_proba(proba, close_v, thr)
            results.append({
                "split": name, "model": mname, "threshold": thr,
                "profit": bt["total_profit_pips"], "sharpe": bt["sharpe_ratio"],
                "dd": bt["max_drawdown_pips"], "pf": bt["profit_factor"],
                "trades": bt["n_trades"], "wr": bt["win_rate"],
            })

    return pd.DataFrame(results)


def main():
    print("=" * 70)
    print("  T07 -- MACHINE LEARNING (AMELIORE)")
    print("  56 features + seuil HOLD optimise + walk-forward")
    print("=" * 70)

    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    # --- Charger ---
    print("\n  [1/5] Chargement des donnees...")
    df_train = pd.read_parquet(os.path.join(FEATURES_DIR, "features_2022.parquet"))
    df_val = pd.read_parquet(os.path.join(FEATURES_DIR, "features_2023.parquet"))
    df_test = pd.read_parquet(os.path.join(FEATURES_DIR, "features_2024.parquet"))
    df_all = pd.concat([df_train, df_val, df_test], ignore_index=True)

    print(f"  Train (2022) : {len(df_train):,}")
    print(f"  Val   (2023) : {len(df_val):,}")
    print(f"  Test  (2024) : {len(df_test):,}")
    print(f"  Features     : {len(FEATURE_COLS)}")

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train["target"].values
    X_val = df_val[FEATURE_COLS].values
    X_test = df_test[FEATURE_COLS].values
    close_val = df_val["close_15m"].values
    close_test = df_test["close_15m"].values

    scaler = StandardScaler()
    X_train_s = np.nan_to_num(scaler.fit_transform(X_train), nan=0, posinf=0, neginf=0)
    X_val_s = np.nan_to_num(scaler.transform(X_val), nan=0, posinf=0, neginf=0)
    X_test_s = np.nan_to_num(scaler.transform(X_test), nan=0, posinf=0, neginf=0)

    # --- Entrainer ---
    print(f"\n{'='*70}")
    print("  [2/5] ENTRAINEMENT + EVALUATION")
    print(f"{'='*70}")

    models = get_models()
    results = []

    for name, model in models.items():
        print(f"\n  --- {name} ---")
        model.fit(X_train_s, y_train)
        proba_val = model.predict_proba(X_val_s)[:, 1]
        proba_test = model.predict_proba(X_test_s)[:, 1]
        best_thr, best_sharpe_val = optimize_threshold(proba_val, close_val)
        bt_val = backtest_proba(proba_val, close_val, best_thr)
        bt_test = backtest_proba(proba_test, close_test, best_thr)

        results.append({
            "model": name, "threshold": best_thr,
            "val_profit": bt_val["total_profit_pips"],
            "val_sharpe": bt_val["sharpe_ratio"],
            "val_trades": bt_val["n_trades"],
            "val_wr": bt_val["win_rate"],
            "test_profit": bt_test["total_profit_pips"],
            "test_sharpe": bt_test["sharpe_ratio"],
            "test_dd": bt_test["max_drawdown_pips"],
            "test_pf": bt_test["profit_factor"],
            "test_trades": bt_test["n_trades"],
            "test_wr": bt_test["win_rate"],
            "test_hold": bt_test["hold_pct"],
        })

        print(f"    Seuil : {best_thr}")
        print(f"    VAL  : Profit={bt_val['total_profit_pips']:+.1f} | Sharpe={bt_val['sharpe_ratio']:.4f} | Trades={bt_val['n_trades']} | WR={bt_val['win_rate']:.1f}%")
        print(f"    TEST : Profit={bt_test['total_profit_pips']:+.1f} | Sharpe={bt_test['sharpe_ratio']:.4f} | Trades={bt_test['n_trades']} | WR={bt_test['win_rate']:.1f}%")

    df_results = pd.DataFrame(results)

    print(f"\n{'='*90}")
    print("  TABLEAU COMPARATIF")
    print(f"{'='*90}")
    print(f"  {'Modele':<10} {'Seuil':>6} {'Val Sharpe':>12} {'Test Sharpe':>12} {'Test Profit':>12} {'Trades':>8} {'WR':>6}")
    print(f"  {'-'*68}")
    for _, r in df_results.iterrows():
        print(f"  {r['model']:<10} {r['threshold']:>6.2f} {r['val_sharpe']:>12.4f} {r['test_sharpe']:>12.4f} {r['test_profit']:>+12.1f} {r['test_trades']:>8} {r['test_wr']:>5.1f}%")

    # --- Walk-Forward ---
    print(f"\n{'='*70}")
    print("  [3/5] WALK-FORWARD VALIDATION")
    print(f"{'='*70}")

    wf_results = walk_forward_validation(df_all, FEATURE_COLS)

    print(f"\n  {'Modele':<10} {'Split':<25} {'Sharpe':>8} {'Profit':>10} {'Trades':>8}")
    print(f"  {'-'*65}")
    for _, row in wf_results.iterrows():
        print(f"  {row['model']:<10} {row['split']:<25} {row['sharpe']:>8.4f} {row['profit']:>+10.1f} {row['trades']:>8}")

    wf_avg = wf_results.groupby("model").agg(
        sharpe_mean=("sharpe", "mean"),
        sharpe_std=("sharpe", "std"),
        profit_mean=("profit", "mean"),
        n_positive=("sharpe", lambda x: (x > 0).sum()),
    ).reset_index().sort_values("sharpe_mean", ascending=False)

    print(f"\n  Moyenne Walk-Forward :")
    for _, row in wf_avg.iterrows():
        print(f"    {row['model']:<10} Sharpe={row['sharpe_mean']:>8.4f} | Profit={row['profit_mean']:>+8.1f} | Positif={int(row['n_positive'])}/4")

    # --- Selection ---
    print(f"\n{'='*70}")
    print("  [4/5] SELECTION DU MEILLEUR MODELE")
    print(f"{'='*70}")

    best_wf_model = wf_avg.iloc[0]["model"]
    best_val_row = df_results[df_results["model"] == best_wf_model].iloc[0]

    print(f"\n  >> Meilleur (Walk-Forward) : {best_wf_model}")
    print(f"     Sharpe WF moyen : {wf_avg.iloc[0]['sharpe_mean']:.4f}")
    print(f"     Sharpe test     : {best_val_row['test_sharpe']:.4f}")

    # --- Sauvegarder ---
    print(f"\n{'='*70}")
    print(f"  [5/5] SAUVEGARDE : {best_wf_model}")
    print(f"{'='*70}")

    X_full = np.vstack([X_train_s, X_val_s])
    y_full = np.concatenate([y_train, df_val["target"].values])
    final_model = get_models()[best_wf_model]
    final_model.fit(X_full, y_full)

    best_thr = best_val_row["threshold"]
    save_obj = {
        "model": final_model,
        "model_name": best_wf_model,
        "model_type": "binary",
        "feature_cols": FEATURE_COLS,
        "needs_scaling": True,
        "scaler": scaler,
        "hold_threshold": best_thr,
        "trained_on": "2022+2023",
        "n_features": len(FEATURE_COLS),
        "metrics_test": {
            "sharpe": float(best_val_row["test_sharpe"]),
            "profit_pips": float(best_val_row["test_profit"]),
            "max_dd": float(best_val_row["test_dd"]),
            "profit_factor": float(best_val_row["test_pf"]),
            "n_trades": int(best_val_row["test_trades"]),
            "win_rate": float(best_val_row["test_wr"]),
        },
        "walk_forward_sharpe_mean": float(wf_avg.iloc[0]["sharpe_mean"]),
        "created_at": datetime.now().isoformat(),
    }

    model_path = os.path.join(MODELS_DIR, "best_ml_model.pkl")
    joblib.dump(save_obj, model_path)
    print(f"  Modele : {model_path}")

    scaler_path = os.path.join(MODELS_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler : {scaler_path}")

    df_results.to_csv(os.path.join(EVAL_DIR, "ml_comparison.csv"), index=False)
    wf_results.to_csv(os.path.join(EVAL_DIR, "ml_walkforward.csv"), index=False)

    info = {
        "version": "v1",
        "type": "ml",
        "model_name": best_wf_model,
        "n_features": len(FEATURE_COLS),
        "hold_threshold": best_thr,
        "trained_on": "2022+2023",
        "metrics_test_2024": save_obj["metrics_test"],
        "walk_forward_sharpe_mean": float(wf_avg.iloc[0]["sharpe_mean"]),
        "created_at": datetime.now().isoformat(),
    }
    with open(os.path.join(MODELS_DIR, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  T07 TERMINE -- Meilleur : {best_wf_model}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
