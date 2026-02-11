"""
T03 — Nettoyage M15
=====================
- Suppression bougies incomplètes (m1_count < 15)
- Contrôle prix négatifs
- Détection et rapport des gaps anormaux
- Sauvegarde M15 propre
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

INPUT_PATH = "data/m15_raw.parquet"
OUTPUT_PATH = "data/m15_clean.parquet"
REPORT_PATH = "data/t03_quality_report.csv"
GAPS_PATH = "data/t03_gaps_report.csv"

# Seuil : on considère une bougie complète si elle a au moins 13 M1 sur 15
# (tolérance pour les petits gaps de marché)
MIN_M1_COUNT = 13


def remove_incomplete_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les bougies M15 avec trop peu de bougies M1."""
    n_before = len(df)
    incomplete = df[df["m1_count"] < MIN_M1_COUNT]

    print(f"\n  --- Bougies incomplètes (m1_count < {MIN_M1_COUNT}) ---")
    print(f"  Nombre : {len(incomplete)}")

    if len(incomplete) > 0:
        # Distribution par année
        for year in sorted(incomplete["year"].unique()):
            n = len(incomplete[incomplete["year"] == year])
            print(f"    {year} : {n} bougies supprimées")

    df_clean = df[df["m1_count"] >= MIN_M1_COUNT].copy().reset_index(drop=True)
    print(f"  Avant : {n_before:,} → Après : {len(df_clean):,}")

    return df_clean


def check_negative_prices(df: pd.DataFrame) -> pd.DataFrame:
    """Contrôle et supprime les bougies avec prix négatifs ou nuls."""
    price_cols = ["open_15m", "high_15m", "low_15m", "close_15m"]
    mask_neg = (df[price_cols] <= 0).any(axis=1)
    n_neg = mask_neg.sum()

    print(f"\n  --- Prix négatifs ou nuls ---")
    print(f"  Nombre : {n_neg}")

    if n_neg > 0:
        df = df[~mask_neg].reset_index(drop=True)
        print(f"  → Lignes supprimées.")

    return df


def check_ohlc_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """Vérifie la cohérence OHLC (high >= low, etc.)."""
    # High >= Low
    bad_hl = (df["high_15m"] < df["low_15m"]).sum()

    # High >= Open et High >= Close
    bad_high = (
        (df["high_15m"] < df["open_15m"]) | (df["high_15m"] < df["close_15m"])
    ).sum()

    # Low <= Open et Low <= Close
    bad_low = (
        (df["low_15m"] > df["open_15m"]) | (df["low_15m"] > df["close_15m"])
    ).sum()

    print(f"\n  --- Cohérence OHLC ---")
    print(f"  High < Low         : {bad_hl}")
    print(f"  High incohérent    : {bad_high}")
    print(f"  Low incohérent     : {bad_low}")

    # Supprimer les incohérences
    mask_bad = (
        (df["high_15m"] < df["low_15m"])
        | (df["high_15m"] < df["open_15m"])
        | (df["high_15m"] < df["close_15m"])
        | (df["low_15m"] > df["open_15m"])
        | (df["low_15m"] > df["close_15m"])
    )

    if mask_bad.sum() > 0:
        df = df[~mask_bad].reset_index(drop=True)
        print(f"  → {mask_bad.sum()} lignes supprimées.")

    return df


def detect_gaps(df: pd.DataFrame) -> pd.DataFrame:
    """Détecte les gaps anormaux entre bougies M15 consécutives."""
    df_sorted = df.sort_values("timestamp").reset_index(drop=True)
    diffs = df_sorted["timestamp"].diff().dropna()

    fifteen_min = pd.Timedelta(minutes=15)

    # Gaps > 15 min (hors weekends = gaps > 2 jours)
    gaps = diffs[diffs > fifteen_min]
    gaps_no_weekend = gaps[gaps < pd.Timedelta(days=2)]

    print(f"\n  --- Gaps entre bougies M15 ---")
    print(f"  Intervalle attendu  : 15 min")
    print(f"  Intervalle médian   : {diffs.median()}")
    print(f"  Gaps > 15 min (total)  : {len(gaps)}")
    print(f"  Gaps > 15 min (hors WE): {len(gaps_no_weekend)}")
    print(f"  Gaps > 1 heure         : {len(diffs[diffs > pd.Timedelta(hours=1)])}")
    print(f"  Gaps > 1 jour (WE)     : {len(diffs[diffs > pd.Timedelta(days=1)])}")

    # Créer rapport de gaps significatifs (> 30 min, hors weekend)
    gap_mask = (diffs > pd.Timedelta(minutes=30)) & (diffs < pd.Timedelta(days=2))
    gap_indices = gap_mask[gap_mask].index

    gap_records = []
    for idx in gap_indices:
        gap_records.append(
            {
                "gap_start": df_sorted.loc[idx - 1, "timestamp"],
                "gap_end": df_sorted.loc[idx, "timestamp"],
                "gap_duration": diffs.loc[idx],
                "year": df_sorted.loc[idx, "year"],
            }
        )

    gaps_df = pd.DataFrame(gap_records)
    if len(gaps_df) > 0:
        print(f"\n  Top 10 plus grands gaps (hors WE) :")
        top_gaps = gaps_df.nlargest(10, "gap_duration")
        for _, row in top_gaps.iterrows():
            print(f"    {row['gap_start']} → {row['gap_end']} ({row['gap_duration']})")

    return gaps_df


def main():
    print("=" * 60)
    print("  T03 — NETTOYAGE M15")
    print("=" * 60)

    # Charger M15 brut
    print(f"\n  Chargement : {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Bougies M15 brutes : {len(df):,}")

    # 1. Suppression bougies incomplètes
    df = remove_incomplete_candles(df)

    # 2. Prix négatifs
    df = check_negative_prices(df)

    # 3. Cohérence OHLC
    df = check_ohlc_consistency(df)

    # 4. Détection gaps
    gaps_df = detect_gaps(df)

    # Résumé par année
    print(f"\n  --- Résumé final ---")
    summary = {}
    for year in sorted(df["year"].unique()):
        sub = df[df["year"] == year]
        summary[year] = {
            "n_candles": len(sub),
            "start": str(sub["timestamp"].min()),
            "end": str(sub["timestamp"].max()),
            "mean_open": sub["open_15m"].mean(),
            "mean_close": sub["close_15m"].mean(),
        }
        print(f"  {year} : {len(sub):,} bougies | {sub['timestamp'].min()} → {sub['timestamp'].max()}")

    print(f"\n  Total M15 propres : {len(df):,}")

    # Supprimer la colonne m1_count (plus besoin)
    df.drop(columns=["m1_count"], inplace=True)

    # Sauvegarde
    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Sauvegardé → {OUTPUT_PATH}")
    print(f"  Taille : {os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB")

    # Rapport qualité
    report = pd.DataFrame(summary).T
    report.index.name = "year"
    report.to_csv(REPORT_PATH)
    print(f"  Rapport qualité → {REPORT_PATH}")

    # Rapport gaps
    if len(gaps_df) > 0:
        gaps_df.to_csv(GAPS_PATH, index=False)
        print(f"  Rapport gaps → {GAPS_PATH}")

    print(f"\n{'='*60}")
    print("  T03 TERMINÉ ✓")
    print(f"{'='*60}")

    return df


if __name__ == "__main__":
    main()
