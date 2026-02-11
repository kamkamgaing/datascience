"""
T02 — Agrégation M1 → M15
===========================
- open_15m  = open de la 1ère minute
- high_15m  = max(high) sur 15 minutes
- low_15m   = min(low) sur 15 minutes
- close_15m = close de la dernière minute
- volume_15m = sum(volume)
- count     = nombre de bougies M1 dans le groupe (pour détecter les bougies incomplètes)

Aucune modélisation autorisée en M1.
"""

import pandas as pd
import numpy as np
import os
import warnings

warnings.filterwarnings("ignore")

INPUT_PATH = "data/m1_combined.parquet"
OUTPUT_PATH = "data/m15_raw.parquet"


def aggregate_m1_to_m15(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les bougies M1 en bougies M15."""

    # Créer le floor à 15 minutes pour le groupement
    # Ex: 17:01 → 17:00, 17:14 → 17:00, 17:15 → 17:15
    df["timestamp_15m"] = df["timestamp"].dt.floor("15min")

    # Agrégation selon les règles imposées
    agg_dict = {
        "open": "first",       # open de la 1ère minute
        "high": "max",         # max(high) sur 15 min
        "low": "min",          # min(low) sur 15 min
        "close": "last",       # close de la dernière minute
        "volume": "sum",       # somme des volumes
        "timestamp": "count",  # nb de bougies M1 (pour contrôle)
    }

    df_15m = (
        df.groupby(["timestamp_15m", "year"])
        .agg(agg_dict)
        .rename(
            columns={
                "open": "open_15m",
                "high": "high_15m",
                "low": "low_15m",
                "close": "close_15m",
                "volume": "volume_15m",
                "timestamp": "m1_count",
            }
        )
        .reset_index()
    )

    # Renommer timestamp
    df_15m.rename(columns={"timestamp_15m": "timestamp"}, inplace=True)

    # Tri chronologique
    df_15m = df_15m.sort_values("timestamp").reset_index(drop=True)

    return df_15m


def main():
    print("=" * 60)
    print("  T02 — AGRÉGATION M1 → M15")
    print("=" * 60)

    # Charger M1
    print(f"\n  Chargement : {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Lignes M1 : {len(df):,}")

    # Agrégation
    print("\n  Agrégation en cours...")
    df_15m = aggregate_m1_to_m15(df)

    print(f"  Bougies M15 générées : {len(df_15m):,}")
    print(f"  Période : {df_15m['timestamp'].min()} → {df_15m['timestamp'].max()}")

    # Statistiques sur le nombre de M1 par bougie M15
    print(f"\n  --- Bougies M1 par groupe M15 ---")
    print(f"  Min    : {df_15m['m1_count'].min()}")
    print(f"  Max    : {df_15m['m1_count'].max()}")
    print(f"  Médiane: {df_15m['m1_count'].median()}")
    print(f"  Moyenne: {df_15m['m1_count'].mean():.1f}")

    # Distribution du m1_count
    count_dist = df_15m["m1_count"].value_counts().sort_index()
    print(f"\n  Distribution m1_count :")
    for count, freq in count_dist.items():
        pct = freq / len(df_15m) * 100
        bar = "#" * int(pct / 2)
        print(f"    {count:2d} bougies M1 : {freq:6,} ({pct:5.1f}%) {bar}")

    # Par année
    for year in sorted(df_15m["year"].unique()):
        sub = df_15m[df_15m["year"] == year]
        print(f"\n  --- {year} ---")
        print(f"  Bougies M15 : {len(sub):,}")
        print(f"  m1_count moyen : {sub['m1_count'].mean():.1f}")
        print(f"  Période : {sub['timestamp'].min()} → {sub['timestamp'].max()}")

    # Sauvegarde
    df_15m.to_parquet(OUTPUT_PATH, index=False)
    print(f"\n  Sauvegardé → {OUTPUT_PATH}")
    print(f"  Taille : {os.path.getsize(OUTPUT_PATH) / 1e6:.1f} MB")

    print(f"\n{'='*60}")
    print("  T02 TERMINE")
    print(f"{'='*60}")

    return df_15m


if __name__ == "__main__":
    main()
