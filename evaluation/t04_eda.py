"""
T04 — Analyse Exploratoire (EDA)
==================================
Script d'analyse exploratoire des données M15 GBP/USD (2022-2024).

Produit :
  - evaluation/eda_returns_distribution.png
  - evaluation/eda_volatility.png
  - evaluation/eda_hourly_patterns.png
  - evaluation/eda_autocorrelation.png
  - evaluation/eda_adf_test.txt
  - evaluation/t04_eda_summary.txt
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend non-interactif
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats as sp_stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import os
import warnings

warnings.filterwarnings("ignore")

# ─── Configuration ────────────────────────────────────────────────────────────
INPUT_PATH = "data/m15_clean.parquet"
OUTPUT_DIR = "evaluation"

# Style global des graphiques
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
})
sns.set_style("whitegrid")


# ═══════════════════════════════════════════════════════════════════════════════
# A) Distribution des rendements
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_returns_distribution(df: pd.DataFrame, returns: pd.Series) -> dict:
    """
    Histogramme des rendements, QQ-plot, et statistiques descriptives.
    Sauvegarde → evaluation/eda_returns_distribution.png
    """
    print("\n  [A] Distribution des rendements...")

    # Statistiques
    mean_ret = returns.mean()
    std_ret = returns.std()
    skew_ret = returns.skew()
    kurt_ret = returns.kurtosis()  # excess kurtosis (Fisher)
    median_ret = returns.median()
    min_ret = returns.min()
    max_ret = returns.max()
    q01 = returns.quantile(0.01)
    q99 = returns.quantile(0.99)

    stats_dict = {
        "mean": mean_ret,
        "std": std_ret,
        "median": median_ret,
        "skewness": skew_ret,
        "kurtosis_excess": kurt_ret,
        "min": min_ret,
        "max": max_ret,
        "quantile_1%": q01,
        "quantile_99%": q99,
        "n_observations": len(returns),
    }

    print(f"      Mean     : {mean_ret:.8f}")
    print(f"      Std      : {std_ret:.6f}")
    print(f"      Skewness : {skew_ret:.4f}")
    print(f"      Kurtosis : {kurt_ret:.4f}")

    # --- Graphiques ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogramme avec courbe normale superposée
    ax = axes[0]
    n_bins = 200
    # Filtrer les rendements extrêmes pour la visualisation
    ret_clipped = returns.clip(q01, q99)
    ax.hist(ret_clipped, bins=n_bins, density=True, alpha=0.7,
            color="steelblue", edgecolor="none", label="Rendements M15")

    # Courbe normale
    x_range = np.linspace(ret_clipped.min(), ret_clipped.max(), 300)
    normal_pdf = sp_stats.norm.pdf(x_range, mean_ret, std_ret)
    ax.plot(x_range, normal_pdf, "r-", lw=2, label="Distribution normale")

    ax.set_title("Distribution des rendements M15 GBP/USD")
    ax.set_xlabel("Rendement")
    ax.set_ylabel("Densité")
    ax.legend(fontsize=9)

    # Ajouter les stats en textbox
    textstr = (f"Mean: {mean_ret:.6f}\n"
               f"Std: {std_ret:.6f}\n"
               f"Skew: {skew_ret:.4f}\n"
               f"Kurt: {kurt_ret:.2f}")
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment="top", bbox=props)

    # QQ-plot
    ax = axes[1]
    sp_stats.probplot(returns.dropna().values, dist="norm", plot=ax)
    ax.set_title("QQ-Plot vs Distribution Normale")
    ax.get_lines()[0].set(markerfacecolor="steelblue", markeredgecolor="steelblue",
                          markersize=1, alpha=0.4)
    ax.get_lines()[1].set(color="red", linewidth=2)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_returns_distribution.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"      → Sauvegardé : {path}")

    return stats_dict


# ═══════════════════════════════════════════════════════════════════════════════
# B) Volatilité dans le temps
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_volatility(df: pd.DataFrame, returns: pd.Series) -> dict:
    """
    Rolling std 20 et 100 périodes, comparaison par année.
    Sauvegarde → evaluation/eda_volatility.png
    """
    print("\n  [B] Volatilité dans le temps...")

    # Rolling std
    roll_20 = returns.rolling(20).std()
    roll_100 = returns.rolling(100).std()

    # Stats par année
    vol_stats = {}
    for year in sorted(df["year"].unique()):
        mask = df["year"] == year
        vol_stats[year] = {
            "vol_mean_20": roll_20[mask].mean(),
            "vol_std_20": roll_20[mask].std(),
            "vol_mean_100": roll_100[mask].mean(),
            "vol_max_20": roll_20[mask].max(),
        }
        print(f"      {year} — Vol moy (20p): {vol_stats[year]['vol_mean_20']:.6f}, "
              f"Vol moy (100p): {vol_stats[year]['vol_mean_100']:.6f}")

    # --- Graphiques ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)

    # Panel 1 : Rolling std au fil du temps
    ax = axes[0]
    timestamps = df["timestamp"]
    ax.plot(timestamps, roll_20, color="steelblue", alpha=0.7, lw=0.5,
            label="Rolling Std 20p")
    ax.plot(timestamps, roll_100, color="red", alpha=0.9, lw=1.2,
            label="Rolling Std 100p")
    ax.set_title("Volatilité des rendements M15 dans le temps")
    ax.set_ylabel("Écart-type glissant")
    ax.legend(fontsize=9)

    # Ajouter des bandes par année
    years = sorted(df["year"].unique())
    colors_year = ["#d4e6f1", "#d5f5e3", "#fdebd0"]
    for i, year in enumerate(years):
        mask = df["year"] == year
        ts_year = timestamps[mask]
        if len(ts_year) > 0:
            ax.axvspan(ts_year.iloc[0], ts_year.iloc[-1],
                       alpha=0.1, color=colors_year[i % len(colors_year)])
            ax.text(ts_year.iloc[len(ts_year)//2], ax.get_ylim()[1] * 0.95,
                    str(year), ha="center", fontsize=9, fontweight="bold", alpha=0.6)

    # Panel 2 : Boxplot par année
    ax = axes[1]
    vol_data = pd.DataFrame({
        "year": df["year"],
        "vol_20": roll_20,
    }).dropna()
    sns.boxplot(data=vol_data, x="year", y="vol_20", ax=ax,
                palette="Set2", width=0.5)
    ax.set_title("Distribution de la volatilité (Std 20p) par année")
    ax.set_xlabel("Année")
    ax.set_ylabel("Écart-type glissant (20p)")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_volatility.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"      → Sauvegardé : {path}")

    return vol_stats


# ═══════════════════════════════════════════════════════════════════════════════
# C) Analyse horaire
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_hourly_patterns(df: pd.DataFrame, returns: pd.Series) -> dict:
    """
    Rendement, volatilité et volume moyens par heure de la journée.
    Sauvegarde → evaluation/eda_hourly_patterns.png
    """
    print("\n  [C] Analyse horaire...")

    # Extraire l'heure
    hours = df["timestamp"].dt.hour

    # Préparer les données horaires
    hourly_df = pd.DataFrame({
        "hour": hours,
        "return": returns,
        "abs_return": returns.abs(),
        "volume": df["volume_15m"],
    })

    # Agrégations
    hourly_agg = hourly_df.groupby("hour").agg(
        mean_return=("return", "mean"),
        std_return=("return", "std"),
        mean_abs_return=("abs_return", "mean"),
        mean_volume=("volume", "mean"),
        count=("return", "count"),
    )

    hourly_stats = {
        "best_hour_return": int(hourly_agg["mean_return"].idxmax()),
        "worst_hour_return": int(hourly_agg["mean_return"].idxmin()),
        "most_volatile_hour": int(hourly_agg["std_return"].idxmax()),
        "least_volatile_hour": int(hourly_agg["std_return"].idxmin()),
        "highest_volume_hour": int(hourly_agg["mean_volume"].idxmax()),
    }

    print(f"      Heure la plus rentable  : {hourly_stats['best_hour_return']}h")
    print(f"      Heure la plus volatile  : {hourly_stats['most_volatile_hour']}h")
    print(f"      Heure plus fort volume  : {hourly_stats['highest_volume_hour']}h")

    # --- Graphiques ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # Rendement moyen par heure
    ax = axes[0]
    colors = ["green" if v >= 0 else "red" for v in hourly_agg["mean_return"]]
    ax.bar(hourly_agg.index, hourly_agg["mean_return"] * 10000, color=colors, alpha=0.8)
    ax.set_title("Rendement moyen par heure (en pips × 0.1)")
    ax.set_xlabel("Heure (UTC)")
    ax.set_ylabel("Rendement moyen (×10⁴)")
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_xticks(range(24))

    # Volatilité par heure
    ax = axes[1]
    ax.bar(hourly_agg.index, hourly_agg["std_return"] * 10000,
           color="steelblue", alpha=0.8)
    ax.set_title("Volatilité (écart-type) par heure")
    ax.set_xlabel("Heure (UTC)")
    ax.set_ylabel("Std des rendements (×10⁴)")
    ax.set_xticks(range(24))

    # Volume moyen par heure
    ax = axes[2]
    ax.bar(hourly_agg.index, hourly_agg["mean_volume"],
           color="orange", alpha=0.8)
    ax.set_title("Volume moyen par heure")
    ax.set_xlabel("Heure (UTC)")
    ax.set_ylabel("Volume moyen")
    ax.set_xticks(range(24))

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_hourly_patterns.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"      → Sauvegardé : {path}")

    return hourly_stats


# ═══════════════════════════════════════════════════════════════════════════════
# D) Autocorrélation
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_autocorrelation(returns: pd.Series) -> dict:
    """
    ACF/PACF des rendements et ACF des rendements au carré.
    Sauvegarde → evaluation/eda_autocorrelation.png
    """
    print("\n  [D] Autocorrélation...")

    ret_clean = returns.dropna()
    ret_squared = ret_clean ** 2

    # --- Graphiques ---
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    # ACF des rendements
    plot_acf(ret_clean, lags=40, ax=axes[0], alpha=0.05,
             title="ACF des rendements M15 (40 lags)")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrélation")

    # PACF des rendements
    plot_pacf(ret_clean, lags=40, ax=axes[1], alpha=0.05,
              method="ywm",
              title="PACF des rendements M15 (40 lags)")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Autocorrélation partielle")

    # ACF des rendements au carré (proxy volatilité)
    plot_acf(ret_squared, lags=40, ax=axes[2], alpha=0.05,
             title="ACF des rendements² (proxy volatilité, 40 lags)")
    axes[2].set_xlabel("Lag")
    axes[2].set_ylabel("Autocorrélation")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "eda_autocorrelation.png")
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"      → Sauvegardé : {path}")

    # Statistiques : premiers lags significatifs
    from statsmodels.tsa.stattools import acf
    acf_vals = acf(ret_clean, nlags=40, alpha=0.05)
    acf_sq_vals = acf(ret_squared, nlags=40, alpha=0.05)

    auto_stats = {
        "acf_lag1_returns": float(acf_vals[0][1]),
        "acf_lag1_returns_sq": float(acf_sq_vals[0][1]),
        "acf_lag5_returns": float(acf_vals[0][5]),
        "acf_lag5_returns_sq": float(acf_sq_vals[0][5]),
    }

    print(f"      ACF lag1 (rendements)  : {auto_stats['acf_lag1_returns']:.4f}")
    print(f"      ACF lag1 (rendements²) : {auto_stats['acf_lag1_returns_sq']:.4f}")
    print(f"      ACF lag5 (rendements)  : {auto_stats['acf_lag5_returns']:.4f}")
    print(f"      ACF lag5 (rendements²) : {auto_stats['acf_lag5_returns_sq']:.4f}")

    return auto_stats


# ═══════════════════════════════════════════════════════════════════════════════
# E) Test ADF de stationnarité
# ═══════════════════════════════════════════════════════════════════════════════

def analyse_adf_test(df: pd.DataFrame, returns: pd.Series) -> dict:
    """
    Test Augmented Dickey-Fuller sur les prix et les rendements.
    Sauvegarde → evaluation/eda_adf_test.txt
    """
    print("\n  [E] Test ADF de stationnarité...")

    close_prices = df["close_15m"].dropna()
    ret_clean = returns.dropna()

    # Test ADF sur les prix close
    adf_price = adfuller(close_prices, maxlag=50, autolag="AIC")
    # Test ADF sur les rendements
    adf_returns = adfuller(ret_clean, maxlag=50, autolag="AIC")

    # Interprétation
    def interpret_adf(result, name):
        stat, pvalue, usedlag, nobs, crit, icbest = result
        stationary = pvalue < 0.05
        return {
            "name": name,
            "adf_statistic": stat,
            "p_value": pvalue,
            "used_lag": usedlag,
            "n_observations": nobs,
            "critical_1%": crit["1%"],
            "critical_5%": crit["5%"],
            "critical_10%": crit["10%"],
            "ic_best": icbest,
            "stationary": stationary,
            "interpretation": "STATIONNAIRE" if stationary else "NON STATIONNAIRE",
        }

    result_price = interpret_adf(adf_price, "Prix close (close_15m)")
    result_returns = interpret_adf(adf_returns, "Rendements (pct_change)")

    # Sauvegarder le rapport texte
    report_lines = [
        "=" * 65,
        "  TEST ADF (Augmented Dickey-Fuller) — Stationnarité",
        "=" * 65,
        "",
        "  H0 : La série possède une racine unitaire (non stationnaire)",
        "  H1 : La série est stationnaire",
        "  Seuil de significativité : 5%",
        "",
        "-" * 65,
        f"  1) {result_price['name']}",
        "-" * 65,
        f"     ADF Statistique : {result_price['adf_statistic']:.6f}",
        f"     P-value         : {result_price['p_value']:.6f}",
        f"     Lag utilisé     : {result_price['used_lag']}",
        f"     Observations    : {result_price['n_observations']}",
        f"     Valeur critique 1%  : {result_price['critical_1%']:.4f}",
        f"     Valeur critique 5%  : {result_price['critical_5%']:.4f}",
        f"     Valeur critique 10% : {result_price['critical_10%']:.4f}",
        f"     → Conclusion    : {result_price['interpretation']}",
        f"       (p-value {'<' if result_price['stationary'] else '>'} 0.05)",
        "",
        "-" * 65,
        f"  2) {result_returns['name']}",
        "-" * 65,
        f"     ADF Statistique : {result_returns['adf_statistic']:.6f}",
        f"     P-value         : {result_returns['p_value']:.6f}",
        f"     Lag utilisé     : {result_returns['used_lag']}",
        f"     Observations    : {result_returns['n_observations']}",
        f"     Valeur critique 1%  : {result_returns['critical_1%']:.4f}",
        f"     Valeur critique 5%  : {result_returns['critical_5%']:.4f}",
        f"     Valeur critique 10% : {result_returns['critical_10%']:.4f}",
        f"     → Conclusion    : {result_returns['interpretation']}",
        f"       (p-value {'<' if result_returns['stationary'] else '>'} 0.05)",
        "",
        "=" * 65,
        "  INTERPRÉTATION GLOBALE",
        "=" * 65,
        "",
    ]

    if not result_price["stationary"] and result_returns["stationary"]:
        report_lines.extend([
            "  Les prix close sont NON STATIONNAIRES (marche aléatoire),",
            "  ce qui est attendu pour une série de prix financiers.",
            "  Les rendements (différence première en %) SONT STATIONNAIRES,",
            "  confirmant que la série des prix est intégrée d'ordre 1 (I(1)).",
            "  → On travaillera avec les RENDEMENTS pour la modélisation.",
        ])
    elif result_price["stationary"] and result_returns["stationary"]:
        report_lines.extend([
            "  Fait inattendu : les prix et les rendements sont tous les deux",
            "  stationnaires. Cela peut arriver sur certaines périodes.",
        ])
    else:
        report_lines.extend([
            f"  Prix : {result_price['interpretation']}",
            f"  Rendements : {result_returns['interpretation']}",
        ])

    report_lines.append("")

    report_text = "\n".join(report_lines)

    path = os.path.join(OUTPUT_DIR, "eda_adf_test.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_text)

    print(f"      Prix close  → {result_price['interpretation']} (p={result_price['p_value']:.4f})")
    print(f"      Rendements  → {result_returns['interpretation']} (p={result_returns['p_value']:.4f})")
    print(f"      → Sauvegardé : {path}")

    return {
        "adf_price_stat": result_price["adf_statistic"],
        "adf_price_pvalue": result_price["p_value"],
        "adf_price_stationary": result_price["stationary"],
        "adf_returns_stat": result_returns["adf_statistic"],
        "adf_returns_pvalue": result_returns["p_value"],
        "adf_returns_stationary": result_returns["stationary"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# RAPPORT RÉSUMÉ
# ═══════════════════════════════════════════════════════════════════════════════

def save_summary(stats_dist, vol_stats, hourly_stats, auto_stats, adf_stats,
                 df, returns):
    """Sauvegarde le rapport résumé complet → evaluation/t04_eda_summary.txt."""

    lines = [
        "=" * 70,
        "  T04 — RAPPORT D'ANALYSE EXPLORATOIRE (EDA)",
        "  GBP/USD M15 — 2022-2024",
        "=" * 70,
        "",
        f"  Période analysée  : {df['timestamp'].min()} → {df['timestamp'].max()}",
        f"  Nombre de bougies : {len(df):,}",
        f"  Nombre de rendements : {len(returns.dropna()):,}",
        "",
        "─" * 70,
        "  A) DISTRIBUTION DES RENDEMENTS",
        "─" * 70,
        f"  Mean (rendement moyen)   : {stats_dist['mean']:.8f}",
        f"  Std (écart-type)         : {stats_dist['std']:.6f}",
        f"  Médiane                  : {stats_dist['median']:.8f}",
        f"  Skewness (asymétrie)     : {stats_dist['skewness']:.4f}",
        f"  Kurtosis (excès)         : {stats_dist['kurtosis_excess']:.4f}",
        f"  Min                      : {stats_dist['min']:.6f}",
        f"  Max                      : {stats_dist['max']:.6f}",
        f"  Quantile 1%              : {stats_dist['quantile_1%']:.6f}",
        f"  Quantile 99%             : {stats_dist['quantile_99%']:.6f}",
        "",
        "  Interprétation :",
    ]

    if abs(stats_dist["skewness"]) < 0.5:
        lines.append("    - Distribution quasi-symétrique (skewness faible)")
    elif stats_dist["skewness"] < 0:
        lines.append("    - Asymétrie négative (queue gauche plus lourde → risque de baisse)")
    else:
        lines.append("    - Asymétrie positive (queue droite plus lourde)")

    if stats_dist["kurtosis_excess"] > 3:
        lines.append(f"    - Kurtosis élevé ({stats_dist['kurtosis_excess']:.1f}) → queues épaisses (fat tails)")
        lines.append("      Les événements extrêmes sont plus fréquents que sous une loi normale.")
    elif stats_dist["kurtosis_excess"] > 0:
        lines.append(f"    - Kurtosis modéré ({stats_dist['kurtosis_excess']:.1f}) → légèrement leptokurtique")

    lines.extend([
        "",
        "─" * 70,
        "  B) VOLATILITÉ DANS LE TEMPS",
        "─" * 70,
    ])

    for year, vs in vol_stats.items():
        lines.append(f"  {year} — Vol moy 20p: {vs['vol_mean_20']:.6f}, "
                      f"Vol moy 100p: {vs['vol_mean_100']:.6f}, "
                      f"Vol max 20p: {vs['vol_max_20']:.6f}")

    lines.extend([
        "",
        "  Interprétation :",
        "    - La volatilité varie significativement dans le temps (clustering).",
        "    - Des périodes de haute et basse volatilité se succèdent.",
        "",
        "─" * 70,
        "  C) PATTERNS HORAIRES",
        "─" * 70,
        f"  Heure la plus rentable         : {hourly_stats['best_hour_return']}h UTC",
        f"  Heure la moins rentable        : {hourly_stats['worst_hour_return']}h UTC",
        f"  Heure la plus volatile          : {hourly_stats['most_volatile_hour']}h UTC",
        f"  Heure la moins volatile         : {hourly_stats['least_volatile_hour']}h UTC",
        f"  Heure avec le plus fort volume  : {hourly_stats['highest_volume_hour']}h UTC",
        "",
        "  Interprétation :",
        "    - La volatilité est typiquement plus forte durant les sessions",
        "      de Londres (7h-16h UTC) et de New York (13h-21h UTC).",
        "    - Le volume suit un pattern similaire avec un pic durant les",
        "      heures de chevauchement Londres/NY.",
        "",
        "─" * 70,
        "  D) AUTOCORRÉLATION",
        "─" * 70,
        f"  ACF lag 1 (rendements)   : {auto_stats['acf_lag1_returns']:.4f}",
        f"  ACF lag 5 (rendements)   : {auto_stats['acf_lag5_returns']:.4f}",
        f"  ACF lag 1 (rendements²)  : {auto_stats['acf_lag1_returns_sq']:.4f}",
        f"  ACF lag 5 (rendements²)  : {auto_stats['acf_lag5_returns_sq']:.4f}",
        "",
        "  Interprétation :",
    ])

    if abs(auto_stats["acf_lag1_returns"]) < 0.05:
        lines.append("    - Les rendements montrent très peu d'autocorrélation → difficilement")
        lines.append("      prévisibles par de simples modèles linéaires AR.")
    else:
        lines.append(f"    - Légère autocorrélation des rendements au lag 1 ({auto_stats['acf_lag1_returns']:.4f}).")

    if auto_stats["acf_lag1_returns_sq"] > 0.05:
        lines.append("    - Les rendements au carré montrent une autocorrélation significative")
        lines.append("      → clustering de volatilité (effet ARCH/GARCH).")
        lines.append("      La volatilité passée aide à prévoir la volatilité future.")

    lines.extend([
        "",
        "─" * 70,
        "  E) STATIONNARITÉ (TEST ADF)",
        "─" * 70,
        f"  Prix close   : ADF stat = {adf_stats['adf_price_stat']:.4f}, "
        f"p-value = {adf_stats['adf_price_pvalue']:.6f} "
        f"→ {'STATIONNAIRE' if adf_stats['adf_price_stationary'] else 'NON STATIONNAIRE'}",
        f"  Rendements   : ADF stat = {adf_stats['adf_returns_stat']:.4f}, "
        f"p-value = {adf_stats['adf_returns_pvalue']:.6f} "
        f"→ {'STATIONNAIRE' if adf_stats['adf_returns_stationary'] else 'NON STATIONNAIRE'}",
        "",
        "  Interprétation :",
        "    - Les prix sont non stationnaires (marche aléatoire, I(1)).",
        "    - Les rendements sont stationnaires → modélisables.",
        "",
        "=" * 70,
        "  CONCLUSIONS POUR LA MODÉLISATION",
        "=" * 70,
        "",
        "  1. Les rendements ont des queues épaisses → utiliser des métriques",
        "     robustes (médiane, MAD) et des modèles gérant les outliers.",
        "  2. Le clustering de volatilité suggère un régime de marché variable",
        "     → un modèle adaptatif ou des features de volatilité sont utiles.",
        "  3. Les patterns horaires montrent une saisonnalité intra-journalière",
        "     → l'heure comme feature peut améliorer les prédictions.",
        "  4. La faible autocorrélation linéaire des rendements suggère que",
        "     des approches non-linéaires (ML) sont nécessaires.",
        "",
        "=" * 70,
    ]
    )

    path = os.path.join(OUTPUT_DIR, "t04_eda_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n  → Rapport résumé sauvegardé : {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("  T04 — ANALYSE EXPLORATOIRE (EDA)")
    print("  GBP/USD M15 — 2022-2024")
    print("=" * 60)

    # Créer le dossier output si nécessaire
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Charger les données
    print(f"\n  Chargement : {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"  Bougies M15 : {len(df):,}")
    print(f"  Période : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Colonnes : {list(df.columns)}")

    # Calculer les rendements
    returns = df["close_15m"].pct_change()
    print(f"  Rendements calculés : {returns.dropna().shape[0]:,} valeurs")

    # A) Distribution des rendements
    stats_dist = analyse_returns_distribution(df, returns)

    # B) Volatilité dans le temps
    vol_stats = analyse_volatility(df, returns)

    # C) Analyse horaire
    hourly_stats = analyse_hourly_patterns(df, returns)

    # D) Autocorrélation
    auto_stats = analyse_autocorrelation(returns)

    # E) Test ADF
    adf_stats = analyse_adf_test(df, returns)

    # Rapport résumé
    save_summary(stats_dist, vol_stats, hourly_stats, auto_stats, adf_stats,
                 df, returns)

    print(f"\n{'='*60}")
    print("  T04 TERMINÉ ✓")
    print(f"  Fichiers générés dans {OUTPUT_DIR}/ :")
    print("    - eda_returns_distribution.png")
    print("    - eda_volatility.png")
    print("    - eda_hourly_patterns.png")
    print("    - eda_autocorrelation.png")
    print("    - eda_adf_test.txt")
    print("    - t04_eda_summary.txt")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
