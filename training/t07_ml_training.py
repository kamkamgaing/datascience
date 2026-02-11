"""
T07 — Machine Learning Training (AMÉLIORÉ)
=============================================
Améliorations vs version précédente :
  - 29 features (ajout features temporelles)
  - Hyperparamètres optimisés
  - Seuil de confiance pour HOLD (réduit les trades inutiles)
  - Métriques financières dès la validation
  - Sauvegarde du scaler avec le modèle

Split temporel strict :
  - 2022 = TRAIN
  - 2023 = VALIDATION
  - 2024 = TEST FINAL
"""

import pandas as pd
import numpy as np
import json
import os
import warnings
from datetime import datetime

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────

FEATURES_DIR = "features"
MODEL_DIR = "models/v1"
EVAL_DIR = "evaluation"

FEATURE_COLS = [
    # Court terme (11)
    "return_1", "return_4", "ema_20", "ema_50", "ema_diff",
    "rsi_14", "rolling_std_20", "range_15m", "body",
    "upper_wick", "lower_wick",
    # Contexte & régime (9)
    "ema_200", "distance_to_ema200", "slope_ema50",
    "atr_14", "rolling_std_100", "volatility_ratio",
    "adx_14", "macd", "macd_signal",
    # Temporel (9)
    "hour", "day_of_week", "is_london", "is_ny", "is_overlap",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos",
]

TARGET_COL = "target"
HOLD_THRESHOLD = 0.54  # Si proba < seuil, on HOLD au lieu de trader
SPREAD_PIPS = 2
PIP_VALUE = 0.0001


def load_data():
    """Charge les 3 fichiers features (split temporel strict)."""
    print("\n  --- Chargement des données ---")

    df_train = pd.read_parquet(os.path.join(FEATURES_DIR, "features_2022.parquet"))
    df_val = pd.read_parquet(os.path.join(FEATURES_DIR, "features_2023.parquet"))
    df_test = pd.read_parquet(os.path.join(FEATURES_DIR, "features_2024.parquet"))

    print(f"  TRAIN  (2022) : {len(df_train):,} lignes")
    print(f"  VALID  (2023) : {len(df_val):,} lignes")
    print(f"  TEST   (2024) : {len(df_test):,} lignes")

    for name, df in [("train", df_train), ("valid", df_val), ("test", df_test)]:
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Features manquantes dans {name} : {missing}")

    X_train = df_train[FEATURE_COLS].values
    y_train = df_train[TARGET_COL].values.astype(int)
    X_val = df_val[FEATURE_COLS].values
    y_val = df_val[TARGET_COL].values.astype(int)
    X_test = df_test[FEATURE_COLS].values
    y_test = df_test[TARGET_COL].values.astype(int)

    for name, y in [("TRAIN", y_train), ("VALID", y_val), ("TEST", y_test)]:
        pos_rate = y.mean() * 100
        print(f"  {name:6s} → target=1 : {y.sum():,} ({pos_rate:.1f}%)  |  target=0 : {(1 - y).sum():,} ({100 - pos_rate:.1f}%)")

    return X_train, y_train, X_val, y_val, X_test, y_test, df_val, df_test


def get_models(random_state=42):
    """Modèles avec hyperparamètres optimisés."""
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=random_state,
            solver="lbfgs",
            C=0.1,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_depth=8,
            min_samples_split=50,
            min_samples_leaf=20,
            max_features="sqrt",
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "XGBoost": XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.01,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_weight=10,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0,
            scale_pos_weight=1.0,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.01,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_alpha=0.1,
            reg_lambda=1.0,
            min_child_samples=30,
            num_leaves=31,
            random_state=random_state,
            verbose=-1,
            is_unbalance=True,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.02,
            subsample=0.7,
            min_samples_split=50,
            min_samples_leaf=20,
            random_state=random_state,
        ),
    }
    return models


def evaluate_model(model, X, y, dataset_name="validation"):
    """Métriques de classification."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y, y_proba),
    }

    return metrics, y_pred, y_proba


def backtest_with_hold(y_proba, close_prices, threshold=HOLD_THRESHOLD):
    """
    Backtest avec seuil HOLD : ne trade que si la confiance est suffisante.
    proba > threshold → BUY
    proba < (1-threshold) → SELL
    sinon → HOLD
    """
    n = len(close_prices) - 1
    pnl_pips = []
    n_trades = 0

    for i in range(n):
        price_move = close_prices[i + 1] - close_prices[i]

        if y_proba[i] >= threshold:
            # BUY — confiance hausse
            profit = (price_move - SPREAD_PIPS * PIP_VALUE) / PIP_VALUE
            pnl_pips.append(profit)
            n_trades += 1
        elif y_proba[i] <= (1 - threshold):
            # SELL — confiance baisse
            profit = (-price_move - SPREAD_PIPS * PIP_VALUE) / PIP_VALUE
            pnl_pips.append(profit)
            n_trades += 1
        else:
            # HOLD — pas assez confiant
            pnl_pips.append(0.0)

    pnl_pips = np.array(pnl_pips)
    cumulative = np.cumsum(pnl_pips)
    total_profit = cumulative[-1] if len(cumulative) > 0 else 0

    # Max drawdown
    running_max = np.maximum.accumulate(cumulative)
    drawdown = cumulative - running_max
    max_dd = drawdown.min() if len(drawdown) > 0 else 0

    # Sharpe
    trading_pnls = pnl_pips[pnl_pips != 0]
    if len(trading_pnls) > 1 and np.std(trading_pnls) > 0:
        sharpe = np.mean(trading_pnls) / np.std(trading_pnls) * np.sqrt(252 * 24 * 4)
    else:
        sharpe = 0.0

    # Profit factor
    gains = trading_pnls[trading_pnls > 0]
    losses = trading_pnls[trading_pnls < 0]
    pf = gains.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else 0.0

    # Win rate
    win_rate = (len(gains) / n_trades * 100) if n_trades > 0 else 0

    return {
        "total_profit_pips": round(float(total_profit), 1),
        "max_drawdown_pips": round(float(max_dd), 1),
        "sharpe": round(float(sharpe), 4),
        "profit_factor": round(float(pf), 4),
        "n_trades": n_trades,
        "win_rate": round(float(win_rate), 2),
        "hold_pct": round((1 - n_trades / n) * 100, 1),
    }


def main():
    print("=" * 70)
    print("  T07 — MACHINE LEARNING TRAINING (AMÉLIORÉ)")
    print("  29 features | Seuil HOLD | Hyperparamètres optimisés")
    print("=" * 70)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(EVAL_DIR, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test, df_val, df_test = load_data()

    # Normalisation
    print("\n  --- Normalisation (StandardScaler) ---")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("  Scaler fit sur TRAIN uniquement.")

    # Entraînement
    print("\n" + "=" * 70)
    print("  ENTRAÎNEMENT DES MODÈLES")
    print("=" * 70)

    models = get_models()
    results = []
    best_sharpe = -999
    best_model_name = None
    best_model = None
    best_proba_val = None

    close_val = df_val["close_15m"].values
    close_test = df_test["close_15m"].values

    for name, model in models.items():
        print(f"\n  {'─' * 60}")
        print(f"  Modèle : {name}")
        print(f"  {'─' * 60}")

        # Les modèles linéaires utilisent données normalisées
        if name == "LogisticRegression":
            X_tr, X_v = X_train_scaled, X_val_scaled
        else:
            X_tr, X_v = X_train, X_val

        print(f"  Entraînement en cours...")
        model.fit(X_tr, y_train)
        print(f"  Entraînement terminé.")

        # Métriques classification
        metrics_val, y_pred_val, y_proba_val = evaluate_model(model, X_v, y_val)

        print(f"\n  Métriques CLASSIFICATION (validation 2023) :")
        print(f"    Accuracy  : {metrics_val['accuracy']:.4f}")
        print(f"    Precision : {metrics_val['precision']:.4f}")
        print(f"    Recall    : {metrics_val['recall']:.4f}")
        print(f"    F1-score  : {metrics_val['f1_score']:.4f}")
        print(f"    AUC-ROC   : {metrics_val['auc_roc']:.4f}")

        # Backtest avec HOLD (métriques financières)
        bt = backtest_with_hold(y_proba_val, close_val, threshold=HOLD_THRESHOLD)

        print(f"\n  Métriques FINANCIÈRES (validation 2023, seuil={HOLD_THRESHOLD}) :")
        print(f"    Profit     : {bt['total_profit_pips']:+.1f} pips")
        print(f"    Drawdown   : {bt['max_drawdown_pips']:.1f} pips")
        print(f"    Sharpe     : {bt['sharpe']:.4f}")
        print(f"    PF         : {bt['profit_factor']:.4f}")
        print(f"    Trades     : {bt['n_trades']:,} ({bt['hold_pct']:.1f}% HOLD)")
        print(f"    Win rate   : {bt['win_rate']:.1f}%")

        print(f"\n  Classification Report :")
        print(classification_report(y_val, y_pred_val, target_names=["DOWN (0)", "UP (1)"], digits=4))

        results.append({
            "model": name,
            "accuracy": metrics_val["accuracy"],
            "precision": metrics_val["precision"],
            "recall": metrics_val["recall"],
            "f1_score": metrics_val["f1_score"],
            "auc_roc": metrics_val["auc_roc"],
            "profit_pips": bt["total_profit_pips"],
            "max_dd_pips": bt["max_drawdown_pips"],
            "sharpe": bt["sharpe"],
            "profit_factor": bt["profit_factor"],
            "n_trades": bt["n_trades"],
            "win_rate": bt["win_rate"],
            "hold_pct": bt["hold_pct"],
        })

        # Sélection : on priorise le Sharpe financier
        if bt["sharpe"] > best_sharpe:
            best_sharpe = bt["sharpe"]
            best_model_name = name
            best_model = model
            best_proba_val = y_proba_val

    # Comparaison
    print("\n" + "=" * 70)
    print("  COMPARAISON DES MODÈLES (VALIDATION 2023)")
    print("=" * 70)

    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values("sharpe", ascending=False).reset_index(drop=True)

    print(f"\n  {'Modèle':<22} {'AUC':>7} {'F1':>7} {'Sharpe':>8} {'Profit':>10} {'Trades':>8} {'WR':>6}")
    print(f"  {'─' * 70}")
    for _, row in df_results.iterrows():
        marker = " <-- BEST" if row["model"] == best_model_name else ""
        print(f"  {row['model']:<22} {row['auc_roc']:>7.4f} {row['f1_score']:>7.4f} {row['sharpe']:>8.4f} {row['profit_pips']:>+10.1f} {row['n_trades']:>8,} {row['win_rate']:>5.1f}%{marker}")

    print(f"\n  >> Meilleur modele (Sharpe) : {best_model_name} -- Sharpe = {best_sharpe:.4f}")

    # Sauvegarder comparaison
    comparison_path = os.path.join(EVAL_DIR, "ml_comparison.csv")
    df_results.to_csv(comparison_path, index=False)
    print(f"\n  Comparaison sauvegardée → {comparison_path}")

    # Sauvegarder le meilleur modèle + scaler
    model_path = os.path.join(MODEL_DIR, "best_ml_model.pkl")
    save_obj = {
        "model": best_model,
        "model_name": best_model_name,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "needs_scaling": best_model_name == "LogisticRegression",
        "hold_threshold": HOLD_THRESHOLD,
    }
    joblib.dump(save_obj, model_path)
    print(f"  Modèle sauvegardé → {model_path}")

    # Sauvegarder le scaler séparément pour l'API/RL
    scaler_path = os.path.join(MODEL_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # model_info.json
    best_row = df_results[df_results["model"] == best_model_name].iloc[0].to_dict()
    model_info = {
        "model_name": best_model_name,
        "task": "binary_classification",
        "target": "direction prochaine bougie M15 GBP/USD",
        "split": {
            "train": f"2022 ({len(X_train)} lignes)",
            "validation": f"2023 ({len(X_val)} lignes)",
            "test": f"2024 ({len(X_test)} lignes)",
        },
        "features": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "hold_threshold": HOLD_THRESHOLD,
        "metrics_validation": {
            "accuracy": round(best_row["accuracy"], 4),
            "precision": round(best_row["precision"], 4),
            "recall": round(best_row["recall"], 4),
            "f1_score": round(best_row["f1_score"], 4),
            "auc_roc": round(best_row["auc_roc"], 4),
        },
        "financial_metrics_validation": {
            "profit_pips": best_row["profit_pips"],
            "max_drawdown_pips": best_row["max_dd_pips"],
            "sharpe": best_row["sharpe"],
            "profit_factor": best_row["profit_factor"],
            "n_trades": int(best_row["n_trades"]),
            "win_rate": best_row["win_rate"],
        },
        "selection_criterion": "Sharpe ratio sur validation 2023",
        "date_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": model_path,
        "needs_scaling": best_model_name == "LogisticRegression",
    }

    info_path = os.path.join(MODEL_DIR, "model_info.json")
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(model_info, f, indent=2, ensure_ascii=False)
    print(f"  Model info → {info_path}")

    # Aperçu sur TEST 2024
    print("\n" + "=" * 70)
    print("  APERÇU BACKTEST SUR TEST (2024)")
    print("=" * 70)

    if best_model_name == "LogisticRegression":
        X_test_eval = X_test_scaled
    else:
        X_test_eval = X_test

    y_proba_test = best_model.predict_proba(X_test_eval)[:, 1]
    bt_test = backtest_with_hold(y_proba_test, close_test, threshold=HOLD_THRESHOLD)

    print(f"  Profit     : {bt_test['total_profit_pips']:+.1f} pips")
    print(f"  Drawdown   : {bt_test['max_drawdown_pips']:.1f} pips")
    print(f"  Sharpe     : {bt_test['sharpe']:.4f}")
    print(f"  PF         : {bt_test['profit_factor']:.4f}")
    print(f"  Trades     : {bt_test['n_trades']:,} ({bt_test['hold_pct']:.1f}% HOLD)")
    print(f"  Win rate   : {bt_test['win_rate']:.1f}%")

    print(f"\n{'='*70}")
    print(f"  T07 TERMINE -- Meilleur modele : {best_model_name}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
