"""
T05 — Feature Engineering V2 (AMÉLIORÉ)
=========================================
Toutes les features sont calculées uniquement à partir du passé (pas de data leakage).

Bloc court terme :
  - return_1, return_4
  - ema_20, ema_50, ema_diff
  - rsi_14
  - rolling_std_20
  - range_15m, body, upper_wick, lower_wick

Bloc Contexte & Régime :
  - ema_200, distance_to_ema200, slope_ema50
  - atr_14, rolling_std_100, volatility_ratio
  - adx_14, macd, macd_signal

Bloc Temporel (NOUVEAU — recommandé par l'EDA) :
  - hour, day_of_week, is_london, is_ny, is_overlap

Sauvegarde par année : features/features_YYYY.parquet
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

INPUT_PATH = "data/m15_clean.parquet"
OUTPUT_DIR = "features"


# ─── Indicateurs techniques ────────────────────────────────────────────────

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    plus_dm = high - prev_high
    minus_dm = prev_low - low
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)
    atr_val = atr(high, low, close, period)
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr_val)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr_val)
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di))
    dx = dx.replace([np.inf, -np.inf], np.nan)
    return dx.ewm(span=period, adjust=False).mean()


def macd_indicator(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


# ─── Feature Engineering ───────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    o = df["open_15m"]
    h = df["high_15m"]
    l = df["low_15m"]
    c = df["close_15m"]

    # ═══════════════════════════════════════════════════════════════
    # BLOC COURT TERME
    # ═══════════════════════════════════════════════════════════════
    df["return_1"] = c.pct_change(1)
    df["return_4"] = c.pct_change(4)
    df["ema_20"] = ema(c, 20)
    df["ema_50"] = ema(c, 50)
    df["ema_diff"] = df["ema_20"] - df["ema_50"]
    df["rsi_14"] = rsi(c, 14)
    df["rolling_std_20"] = c.pct_change().rolling(20).std()
    df["range_15m"] = h - l
    df["body"] = c - o
    df["upper_wick"] = h - pd.concat([o, c], axis=1).max(axis=1)
    df["lower_wick"] = pd.concat([o, c], axis=1).min(axis=1) - l

    # ═══════════════════════════════════════════════════════════════
    # BLOC CONTEXTE & RÉGIME
    # ═══════════════════════════════════════════════════════════════
    df["ema_200"] = ema(c, 200)
    df["distance_to_ema200"] = (c - df["ema_200"]) / df["ema_200"]
    df["slope_ema50"] = df["ema_50"].diff(5) / 5
    df["atr_14"] = atr(h, l, c, 14)
    df["rolling_std_100"] = c.pct_change().rolling(100).std()
    df["volatility_ratio"] = df["rolling_std_20"] / df["rolling_std_100"]
    df["adx_14"] = adx(h, l, c, 14)
    df["macd"], df["macd_signal"] = macd_indicator(c)

    # ═══════════════════════════════════════════════════════════════
    # BLOC TEMPOREL (NOUVEAU)
    # L'EDA a montré des patterns horaires significatifs
    # ═══════════════════════════════════════════════════════════════
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek  # 0=lundi, 4=vendredi
    # Sessions de trading (UTC)
    df["is_london"] = ((df["hour"] >= 7) & (df["hour"] < 16)).astype(int)
    df["is_ny"] = ((df["hour"] >= 13) & (df["hour"] < 22)).astype(int)
    df["is_overlap"] = ((df["hour"] >= 13) & (df["hour"] < 16)).astype(int)

    # Encodage cyclique de l'heure (évite la discontinuité 23→0)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 5)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 5)

    # ═══════════════════════════════════════════════════════════════
    # TARGET
    # ═══════════════════════════════════════════════════════════════
    df["target"] = (c.shift(-1) > c).astype(float)

    return df


def main():
    print("=" * 60)
    print("  T05 — FEATURE ENGINEERING V2 (AMÉLIORÉ)")
    print("=" * 60)

    print(f"\n  Chargement : {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Bougies M15 : {len(df):,}")

    print("\n  Calcul des features...")
    df_feat = compute_features(df)

    feature_cols = [
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

    save_cols = ["timestamp", "year", "open_15m", "high_15m", "low_15m", "close_15m",
                 "volume_15m"] + feature_cols + ["target"]

    print(f"\n  --- NaN par feature (warm-up) ---")
    for col in feature_cols:
        n_nan = df_feat[col].isna().sum()
        if n_nan > 0:
            print(f"    {col:25s} : {n_nan:,} NaN")

    n_before = len(df_feat)
    df_feat = df_feat.dropna(subset=feature_cols).reset_index(drop=True)
    print(f"\n  Lignes supprimées (warm-up) : {n_before - len(df_feat):,}")
    print(f"  Lignes restantes : {len(df_feat):,}")

    print(f"\n  --- Sauvegarde par année ---")
    for year in sorted(df_feat["year"].unique()):
        df_year = df_feat[df_feat["year"] == year][save_cols].reset_index(drop=True)
        df_year = df_year.dropna(subset=["target"]).reset_index(drop=True)
        output_path = os.path.join(OUTPUT_DIR, f"features_{year}.parquet")
        df_year.to_parquet(output_path, index=False)
        size_mb = os.path.getsize(output_path) / 1e6
        print(f"  {year} : {len(df_year):,} lignes → {output_path} ({size_mb:.1f} MB)")
        print(f"         target=1 : {df_year['target'].sum():.0f} ({df_year['target'].mean()*100:.1f}%)")
        print(f"         target=0 : {(1-df_year['target']).sum():.0f} ({(1-df_year['target']).mean()*100:.1f}%)")

    full_path = os.path.join(OUTPUT_DIR, "features_all.parquet")
    df_feat_clean = df_feat[save_cols].dropna(subset=["target"]).reset_index(drop=True)
    df_feat_clean.to_parquet(full_path, index=False)
    print(f"\n  Dataset complet : {len(df_feat_clean):,} lignes → {full_path}")

    print(f"\n  --- Résumé : {len(feature_cols)} features ---")
    print(f"  Court terme  (11) : return_1..lower_wick")
    print(f"  Contexte      (9) : ema_200..macd_signal")
    print(f"  Temporel      (9) : hour, day_of_week, sessions, encodage cyclique")

    print(f"\n{'='*60}")
    print("  T05 TERMINE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
