"""
T05 -- Feature Engineering V3 (AMELIORE)
==========================================
Ajouts par rapport a V2 :
  - Bloc Price Action : patterns de bougies, momentum multi-horizon
  - Bloc Regime : detection de regime via volatilite, trend strength
  - Bloc Micro-structure : volume relatif, price acceleration
  - Target ameliore : target multi-horizon + target filtre (mouvement > seuil)
  - Total : ~56 features

Toutes les features sont calculees uniquement a partir du passe (pas de data leakage).
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

INPUT_PATH = "data/m15_clean.parquet"
OUTPUT_DIR = "features"


# --- Indicateurs techniques de base ---

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


# --- Stochastic Oscillator ---
def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3):
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = k.replace([np.inf, -np.inf], np.nan)
    d = k.rolling(d_period).mean()
    return k, d


# --- Bollinger Bands ---
def bollinger_bands(close: pd.Series, period: int = 20, n_std: float = 2.0):
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    bb_width = (upper - lower) / sma
    bb_position = (close - lower) / (upper - lower)
    bb_position = bb_position.replace([np.inf, -np.inf], np.nan)
    return bb_width, bb_position


# =============================================================================
# Feature Engineering V3
# =============================================================================
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    o = df["open_15m"]
    h = df["high_15m"]
    l = df["low_15m"]
    c = df["close_15m"]
    v = df["volume_15m"]

    # =================================================================
    # BLOC 1 : COURT TERME
    # =================================================================
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

    # =================================================================
    # BLOC 2 : CONTEXTE & REGIME
    # =================================================================
    df["ema_200"] = ema(c, 200)
    df["distance_to_ema200"] = (c - df["ema_200"]) / df["ema_200"]
    df["slope_ema50"] = df["ema_50"].diff(5) / 5
    df["atr_14"] = atr(h, l, c, 14)
    df["rolling_std_100"] = c.pct_change().rolling(100).std()
    df["volatility_ratio"] = df["rolling_std_20"] / df["rolling_std_100"]
    df["adx_14"] = adx(h, l, c, 14)
    df["macd"], df["macd_signal"] = macd_indicator(c)

    # =================================================================
    # BLOC 3 : TEMPOREL
    # =================================================================
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_london"] = ((df["hour"] >= 7) & (df["hour"] < 16)).astype(int)
    df["is_ny"] = ((df["hour"] >= 13) & (df["hour"] < 22)).astype(int)
    df["is_overlap"] = ((df["hour"] >= 13) & (df["hour"] < 16)).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 5)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 5)

    # =================================================================
    # BLOC 4 : PRICE ACTION (NOUVEAU)
    # =================================================================
    df["return_8"] = c.pct_change(8)
    df["return_16"] = c.pct_change(16)
    df["return_96"] = c.pct_change(96)
    df["momentum_4"] = df["return_1"] - df["return_1"].shift(4)
    df["momentum_8"] = df["return_1"] - df["return_1"].shift(8)
    df["body_ratio"] = df["body"].abs() / df["range_15m"].replace(0, np.nan)
    df["body_ratio"] = df["body_ratio"].clip(-1, 1).fillna(0)

    direction = (c > o).astype(int) * 2 - 1
    consec = direction.copy()
    for i in range(1, len(consec)):
        if direction.iloc[i] == direction.iloc[i - 1]:
            consec.iloc[i] = consec.iloc[i - 1] + direction.iloc[i]
        else:
            consec.iloc[i] = direction.iloc[i]
    df["consecutive_candles"] = consec

    df["high_20_ratio"] = c / h.rolling(20).max()
    df["low_20_ratio"] = c / l.rolling(20).min()
    df["high_96_ratio"] = c / h.rolling(96).max()

    # =================================================================
    # BLOC 5 : REGIME DE MARCHE (NOUVEAU)
    # =================================================================
    df["bb_width"], df["bb_position"] = bollinger_bands(c, 20, 2.0)
    df["stoch_k"], df["stoch_d"] = stochastic(h, l, c, 14, 3)
    df["macd_hist"] = df["macd"] - df["macd_signal"]
    df["macd_hist_diff"] = df["macd_hist"].diff()
    df["above_ema20"] = (c > df["ema_20"]).astype(int)
    df["above_ema50"] = (c > df["ema_50"]).astype(int)
    df["above_ema200"] = (c > df["ema_200"]).astype(int)
    df["trend_alignment"] = df["above_ema20"] + df["above_ema50"] + df["above_ema200"]
    df["atr_normalized"] = df["atr_14"] / c
    df["volatility_change"] = df["atr_14"].pct_change(4)

    # =================================================================
    # BLOC 6 : MICRO-STRUCTURE (NOUVEAU)
    # =================================================================
    df["volume_sma_20"] = v.rolling(20).mean()
    df["volume_ratio"] = v / df["volume_sma_20"].replace(0, np.nan)
    df["volume_ratio"] = df["volume_ratio"].fillna(1.0)
    df["vwap_proxy"] = (df["return_1"] * df["volume_ratio"]).rolling(4).mean()
    df["rsi_7"] = rsi(c, 7)
    df["rsi_21"] = rsi(c, 21)
    df["rsi_divergence"] = df["rsi_7"] - df["rsi_21"]

    # =================================================================
    # TARGETS
    # =================================================================
    df["target"] = (c.shift(-1) > c).astype(float)

    future_move = (c.shift(-1) - c) / 0.0001
    df["target_3pips"] = 0.0
    df.loc[future_move > 3, "target_3pips"] = 1.0
    df.loc[future_move < -3, "target_3pips"] = -1.0

    future_move_4 = (c.shift(-4) - c) / 0.0001
    df["target_4bars"] = 0.0
    df.loc[future_move_4 > 5, "target_4bars"] = 1.0
    df.loc[future_move_4 < -5, "target_4bars"] = -1.0

    return df


def main():
    print("=" * 60)
    print("  T05 -- FEATURE ENGINEERING V3 (AMELIORE)")
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

    target_cols = ["target", "target_3pips", "target_4bars"]

    save_cols = ["timestamp", "year", "open_15m", "high_15m", "low_15m", "close_15m",
                 "volume_15m"] + feature_cols + target_cols

    print(f"\n  --- NaN par feature (warm-up) ---")
    nan_counts = {}
    for col in feature_cols:
        n_nan = df_feat[col].isna().sum()
        if n_nan > 0:
            nan_counts[col] = n_nan
    for col, n in sorted(nan_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"    {col:25s} : {n:,} NaN")
    if len(nan_counts) > 10:
        print(f"    ... et {len(nan_counts) - 10} autres features avec NaN")

    n_before = len(df_feat)
    df_feat = df_feat.dropna(subset=feature_cols).reset_index(drop=True)
    print(f"\n  Lignes supprimees (warm-up) : {n_before - len(df_feat):,}")
    print(f"  Lignes restantes : {len(df_feat):,}")

    for col in feature_cols:
        df_feat[col] = df_feat[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    print(f"\n  --- Sauvegarde par annee ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for year in sorted(df_feat["year"].unique()):
        df_year = df_feat[df_feat["year"] == year][save_cols].reset_index(drop=True)
        df_year = df_year.dropna(subset=["target"]).reset_index(drop=True)
        output_path = os.path.join(OUTPUT_DIR, f"features_{year}.parquet")
        df_year.to_parquet(output_path, index=False)
        size_mb = os.path.getsize(output_path) / 1e6
        n_buy = (df_year["target_3pips"] == 1).sum()
        n_sell = (df_year["target_3pips"] == -1).sum()
        n_hold = (df_year["target_3pips"] == 0).sum()
        print(f"  {year} : {len(df_year):,} lignes -> {output_path} ({size_mb:.1f} MB)")
        print(f"         target binaire  : {df_year['target'].mean()*100:.1f}% UP")
        print(f"         target 3pips    : BUY={n_buy} HOLD={n_hold} SELL={n_sell}")

    full_path = os.path.join(OUTPUT_DIR, "features_all.parquet")
    df_feat_clean = df_feat[save_cols].dropna(subset=["target"]).reset_index(drop=True)
    df_feat_clean.to_parquet(full_path, index=False)
    print(f"\n  Dataset complet : {len(df_feat_clean):,} lignes -> {full_path}")

    print(f"\n  --- Resume : {len(feature_cols)} features ---")
    print(f"  Court terme   (11) : return_1..lower_wick")
    print(f"  Contexte       (9) : ema_200..macd_signal")
    print(f"  Temporel       (9) : hour..dow_cos")
    print(f"  Price Action  (10) : return_8..high_96_ratio")
    print(f"  Regime        (12) : bb_width..volatility_change")
    print(f"  Micro-struct   (5) : volume_ratio..rsi_divergence")
    print(f"  TOTAL         ({len(feature_cols)}) features")

    print(f"\n{'='*60}")
    print("  T05 TERMINE")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
