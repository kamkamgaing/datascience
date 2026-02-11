"""
T11 — Script de mise à jour du registry des modèles
=====================================================
Lit les fichiers model_info.json de v1/ et v2/ et met à jour
le fichier registry.json avec les informations actuelles.

Usage :
    python models/update_registry.py
    python models/update_registry.py --promote v2   # Met v2 en production
"""

import json
import argparse
import sys
from datetime import datetime
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"

# Versions connues
VERSIONS = {
    "v1": {
        "type": "ml",
        "default_algorithm": "unknown",
        "model_file": "best_ml_model.pkl",
        "default_metrics": {"f1": 0.0, "sharpe": 0.0},
    },
    "v2": {
        "type": "rl",
        "default_algorithm": "PPO",
        "model_file": "best_rl_model.zip",
        "default_metrics": {"sharpe": 0.0, "profit_cumule": 0.0},
    },
}


def load_model_info(version: str) -> dict:
    """
    Charge le fichier model_info.json d'une version donnée.

    Args:
        version: "v1" ou "v2"

    Returns:
        Dictionnaire avec les infos du modèle, ou None si le fichier n'existe pas
    """
    info_path = MODELS_DIR / version / "model_info.json"
    if info_path.exists():
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        print(f"  OK {version}/model_info.json charge")
        return info
    else:
        print(f"  MISSING {version}/model_info.json introuvable")
        return None


def check_model_exists(version: str) -> bool:
    """Vérifie si le fichier modèle existe pour une version donnée."""
    config = VERSIONS.get(version, {})
    model_file = config.get("model_file", "")
    model_path = MODELS_DIR / version / model_file
    return model_path.exists()


def build_registry_entry(version: str, info: dict = None, is_production: bool = False) -> dict:
    """
    Construit une entrée de registry pour une version donnée.

    Args:
        version: "v1" ou "v2"
        info: Dictionnaire model_info.json (ou None)
        is_production: Si ce modèle est en production

    Returns:
        Entrée de registry
    """
    config = VERSIONS[version]

    if info is not None:
        # Extraire les métriques selon le type
        if config["type"] == "rl":
            metrics_val = info.get("metrics_validation", {})
            metrics = {
                "sharpe": metrics_val.get("sharpe_ratio", 0.0),
                "profit_cumule": metrics_val.get("cumulative_pnl", 0.0),
                "max_drawdown": metrics_val.get("max_drawdown", 0.0),
                "num_trades": metrics_val.get("num_trades", 0),
            }
        elif config["type"] == "ml":
            # Gère les différents formats de model_info.json
            metrics_raw = info.get("metrics_validation", info.get("metrics", info.get("metrics_test", {})))
            metrics = {
                "f1": metrics_raw.get("f1", metrics_raw.get("f1_score", 0.0)),
                "sharpe": metrics_raw.get("sharpe", metrics_raw.get("sharpe_ratio", 0.0)),
                "accuracy": metrics_raw.get("accuracy", 0.0),
                "auc_roc": metrics_raw.get("auc_roc", 0.0),
            }
        else:
            metrics = config["default_metrics"]

        # Gère "algorithm" ou "model_name" selon le format
        algorithm = info.get("algorithm", info.get("model_name", config["default_algorithm"]))
        created_at = info.get("created_at", datetime.now().isoformat())
        model_path = info.get("model_path", info.get("path", f"models/{version}/{config['model_file']}"))
    else:
        algorithm = config["default_algorithm"]
        created_at = datetime.now().isoformat()
        model_path = f"models/{version}/{config['model_file']}"
        metrics = config["default_metrics"]

    entry = {
        "version": version,
        "type": config["type"],
        "algorithm": algorithm,
        "path": model_path,
        "metrics": metrics,
        "created_at": created_at,
        "is_production": is_production,
        "model_exists": check_model_exists(version),
    }

    return entry


def load_existing_registry() -> dict:
    """Charge le registry existant ou retourne un registry vide."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"models": []}


def get_current_production(registry: dict) -> str:
    """Retourne la version actuellement en production."""
    for model in registry.get("models", []):
        if model.get("is_production", False):
            return model["version"]
    return None


def update_registry(promote_version: str = None):
    """
    Met à jour le registry avec les infos actuelles des modèles.

    Args:
        promote_version: Si spécifié, met cette version en production
    """
    print("=" * 60)
    print("Mise à jour du registry des modèles")
    print("=" * 60)

    # Charger le registry existant pour connaître le modèle en production
    existing_registry = load_existing_registry()
    current_production = get_current_production(existing_registry)
    print(f"\nModèle en production actuel : {current_production or 'aucun'}")

    # Déterminer le nouveau modèle en production
    if promote_version:
        if promote_version not in VERSIONS:
            print(f"\nERREUR : Version '{promote_version}' inconnue. Versions disponibles : {list(VERSIONS.keys())}")
            sys.exit(1)
        if not check_model_exists(promote_version):
            print(f"\nATTENTION : Le fichier modele de {promote_version} n'existe pas encore !")
        production_version = promote_version
        print(f"→ Promotion de {promote_version} en production")
    else:
        production_version = current_production or "v1"

    # Charger les infos de chaque version
    print("\nChargement des informations des modèles :")
    models = []
    for version in VERSIONS:
        info = load_model_info(version)
        is_production = (version == production_version)
        entry = build_registry_entry(version, info, is_production)
        models.append(entry)

    # Construire le registry
    registry = {
        "models": models,
        "last_updated": datetime.now().isoformat(),
        "production_version": production_version,
    }

    # Sauvegarder
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)

    print(f"\nOK Registry sauvegarde : {REGISTRY_PATH}")
    print(f"\nRésumé :")
    print("-" * 40)
    for model in models:
        status = ">> PRODUCTION" if model["is_production"] else "   standby"
        exists = "OK" if model.get("model_exists", False) else "MISSING"
        print(
            f"  [{exists}] {model['version']} ({model['type']}/{model['algorithm']}) "
            f"— {status}"
        )
    print("-" * 40)


# =============================================================================
# Point d'entrée
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Met à jour le registry des modèles de trading"
    )
    parser.add_argument(
        "--promote",
        type=str,
        choices=list(VERSIONS.keys()),
        default=None,
        help="Version à mettre en production (ex: v1, v2)",
    )

    args = parser.parse_args()
    update_registry(promote_version=args.promote)
