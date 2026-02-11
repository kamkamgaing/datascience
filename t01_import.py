"""
T01 — Importation M1
=====================
- Fusion date + time → timestamp
- Vérification régularité 1 minute
- Tri chronologique
- Détection incohérences (doublons, prix négatifs, OHLC invalides)
- Sauvegarde en parquet
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────────────────
DATA_DIR = "."
OUTPUT_DIR = "data"
FILES = {
    2022: "DAT_MT_GBPUSD_M1_2022.csv",
    2023: "DAT_MT_GBPUSD_M1_2023.csv",
    2024: "DAT_MT_GBPUSD_M1_2024.csv",
}
COLUMNS = ["date", "time", "open", "high", "low", "close", "volume"]


def load_single_year(filepath: str, year: int) -> pd.DataFrame:
    """Charge un fichier CSV M1 et crée le timestamp."""
    print(f"\n{'='*60}")
    print(f"  Chargement {year} : {filepath}")
    print(f"{'='*60}")

    df = pd.read_csv(filepath, header=None, names=COLUMNS)
    print(f"  Lignes brutes : {len(df):,}")

    # Fusion date + time → timestamp
    df["timestamp"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        format="%Y.%m.%d %H:%M",
    )
    df.drop(columns=["date", "time"], inplace=True)

    # Réordonner les colonnes
    df = df[["timestamp", "open", "high", "low", "close", "volume"]]

    # Ajouter colonne année
    df["year"] = year

    return df


def check_chronological_order(df: pd.DataFrame) -> pd.DataFrame:
    """Vérifie et impose le tri chronologique."""
    is_sorted = df["timestamp"].is_monotonic_increasing
    print(f"  Tri chronologique OK : {is_sorted}")

    if not is_sorted:
        df = df.sort_values("timestamp").reset_index(drop=True)
        print("  → Données re-triées chronologiquement.")

    return df


def detect_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Détecte et supprime les doublons de timestamp."""
    n_dupes = df["timestamp"].duplicated().sum()
    print(f"  Doublons de timestamp : {n_dupes}")

    if n_dupes > 0:
        df = df.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)
        print(f"  → Doublons supprimés. Lignes restantes : {len(df):,}")

    return df


def check_regularity(df: pd.DataFrame) -> dict:
    """Vérifie la régularité 1 minute et détecte les gaps."""
    diffs = df["timestamp"].diff().dropna()
    one_min = pd.Timedelta(minutes=1)

    # Statistiques des intervalles
    stats = {
        "interval_min": str(diffs.min()),
        "interval_max": str(diffs.max()),
        "interval_median": str(diffs.median()),
        "total_rows": len(df),
    }

    # Gaps > 1 minute (hors weekends)
    gaps = diffs[diffs > one_min]
    # On considère un gap "anormal" si > 5 minutes (hors pause marché)
    significant_gaps = diffs[diffs > pd.Timedelta(minutes=5)]

    stats["n_gaps_gt_1min"] = len(gaps)
    stats["n_gaps_gt_5min"] = len(significant_gaps)
    stats["n_gaps_gt_1h"] = len(diffs[diffs > pd.Timedelta(hours=1)])

    print(f"  Intervalle min  : {stats['interval_min']}")
    print(f"  Intervalle max  : {stats['interval_max']}")
    print(f"  Intervalle médian : {stats['interval_median']}")
    print(f"  Gaps > 1 min    : {stats['n_gaps_gt_1min']}")
    print(f"  Gaps > 5 min    : {stats['n_gaps_gt_5min']}")
    print(f"  Gaps > 1 heure  : {stats['n_gaps_gt_1h']}")

    return stats


def check_ohlc_integrity(df: pd.DataFrame) -> dict:
    """Vérifie la cohérence OHLC."""
    issues = {}

    # Prix négatifs ou nuls
    neg_prices = (df[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
    issues["prix_negatifs_ou_nuls"] = int(neg_prices)
    print(f"  Prix négatifs/nuls : {neg_prices}")

    # High doit être >= Open, Close, Low
    bad_high = (
        (df["high"] < df["open"])
        | (df["high"] < df["close"])
        | (df["high"] < df["low"])
    ).sum()
    issues["high_invalide"] = int(bad_high)
    print(f"  High < (Open|Close|Low) : {bad_high}")

    # Low doit être <= Open, Close, High
    bad_low = (
        (df["low"] > df["open"])
        | (df["low"] > df["close"])
        | (df["low"] > df["high"])
    ).sum()
    issues["low_invalide"] = int(bad_low)
    print(f"  Low > (Open|Close|High) : {bad_low}")

    # Volume négatif
    neg_vol = (df["volume"] < 0).sum()
    issues["volume_negatif"] = int(neg_vol)
    print(f"  Volume négatif : {neg_vol}")

    return issues


def main():
    print("=" * 60)
    print("  T01 — IMPORTATION M1 GBP/USD")
    print("=" * 60)

    all_dfs = []
    all_stats = {}

    for year, filename in FILES.items():
        filepath = os.path.join(DATA_DIR, filename)

        if not os.path.exists(filepath):
            print(f"  ERREUR : Fichier introuvable → {filepath}")
            continue

        # Charger
        df = load_single_year(filepath, year)

        # Tri chronologique
        df = check_chronological_order(df)

        # Doublons
        df = detect_duplicates(df)

        # Régularité
        print(f"\n  --- Régularité {year} ---")
        reg_stats = check_regularity(df)

        # Intégrité OHLC
        print(f"\n  --- Intégrité OHLC {year} ---")
        ohlc_stats = check_ohlc_integrity(df)

        all_stats[year] = {**reg_stats, **ohlc_stats}
        all_dfs.append(df)

    # Fusion des 3 années
    print(f"\n{'='*60}")
    print("  FUSION DES 3 ANNÉES")
    print(f"{'='*60}")

    df_combined = pd.concat(all_dfs, ignore_index=True)
    df_combined = df_combined.sort_values("timestamp").reset_index(drop=True)

    # Vérification finale des doublons inter-années
    n_dupes_final = df_combined["timestamp"].duplicated().sum()
    print(f"  Doublons inter-années : {n_dupes_final}")
    if n_dupes_final > 0:
        df_combined = df_combined.drop_duplicates(subset="timestamp", keep="first").reset_index(drop=True)

    print(f"  Total lignes combinées : {len(df_combined):,}")
    print(f"  Période : {df_combined['timestamp'].min()} → {df_combined['timestamp'].max()}")

    # Sauvegarde
    output_path = os.path.join(OUTPUT_DIR, "m1_combined.parquet")
    df_combined.to_parquet(output_path, index=False)
    print(f"\n  Sauvegardé → {output_path}")
    print(f"  Taille : {os.path.getsize(output_path) / 1e6:.1f} MB")

    # Rapport de qualité
    report_path = os.path.join(OUTPUT_DIR, "t01_quality_report.csv")
    report_df = pd.DataFrame(all_stats).T
    report_df.index.name = "year"
    report_df.to_csv(report_path)
    print(f"  Rapport qualité → {report_path}")

    print(f"\n{'='*60}")
    print("  T01 TERMINÉ ✓")
    print(f"{'='*60}")

    return df_combined


if __name__ == "__main__":
    main()
