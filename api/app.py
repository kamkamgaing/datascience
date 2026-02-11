"""
T10 — API FastAPI pour le système de trading GBP/USD M15
=========================================================
Endpoints :
- GET  /health      → Vérification de santé + version modèle
- GET  /model/info  → Informations du modèle en production
- POST /predict     → Prédiction BUY/SELL/HOLD à partir des 20 features
"""

import json
import os
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# =============================================================================
# Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"

# Logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("trading-api")

# =============================================================================
# Schémas Pydantic
# =============================================================================

class PredictRequest(BaseModel):
    """Schéma de la requête de prédiction avec les 29 features."""
    # Court terme (11)
    return_1: float = Field(..., description="Rendement sur 1 période")
    return_4: float = Field(..., description="Rendement sur 4 périodes")
    ema_20: float = Field(..., description="EMA 20 périodes")
    ema_50: float = Field(..., description="EMA 50 périodes")
    ema_diff: float = Field(..., description="Différence EMA20 - EMA50")
    rsi_14: float = Field(..., description="RSI 14 périodes")
    rolling_std_20: float = Field(..., description="Écart-type roulant 20 périodes")
    range_15m: float = Field(..., description="Range de la bougie M15")
    body: float = Field(..., description="Corps de la bougie")
    upper_wick: float = Field(..., description="Mèche haute")
    lower_wick: float = Field(..., description="Mèche basse")
    # Contexte & régime (9)
    ema_200: float = Field(..., description="EMA 200 périodes")
    distance_to_ema200: float = Field(..., description="Distance au EMA200")
    slope_ema50: float = Field(..., description="Pente de l'EMA50")
    atr_14: float = Field(..., description="ATR 14 périodes")
    rolling_std_100: float = Field(..., description="Écart-type roulant 100 périodes")
    volatility_ratio: float = Field(..., description="Ratio de volatilité")
    adx_14: float = Field(..., description="ADX 14 périodes")
    macd: float = Field(..., description="MACD")
    macd_signal: float = Field(..., description="Signal MACD")
    # Temporel (9)
    hour: float = Field(0, description="Heure UTC (0-23)")
    day_of_week: float = Field(0, description="Jour de la semaine (0=lundi)")
    is_london: float = Field(0, description="Session Londres (0/1)")
    is_ny: float = Field(0, description="Session New York (0/1)")
    is_overlap: float = Field(0, description="Chevauchement Londres/NY (0/1)")
    hour_sin: float = Field(0, description="Heure encodage sin")
    hour_cos: float = Field(1, description="Heure encodage cos")
    dow_sin: float = Field(0, description="Jour encodage sin")
    dow_cos: float = Field(1, description="Jour encodage cos")

    class Config:
        json_schema_extra = {
            "example": {
                "return_1": 0.0001, "return_4": 0.0003,
                "ema_20": 1.2650, "ema_50": 1.2640, "ema_diff": 0.001,
                "rsi_14": 55.0, "rolling_std_20": 0.002,
                "range_15m": 0.0015, "body": 0.0008,
                "upper_wick": 0.0004, "lower_wick": 0.0003,
                "ema_200": 1.2600, "distance_to_ema200": 0.005,
                "slope_ema50": 0.0001, "atr_14": 0.0012,
                "rolling_std_100": 0.003, "volatility_ratio": 0.8,
                "adx_14": 25.0, "macd": 0.0002, "macd_signal": 0.0001,
                "hour": 14, "day_of_week": 2,
                "is_london": 1, "is_ny": 1, "is_overlap": 1,
                "hour_sin": 0.866, "hour_cos": -0.5,
                "dow_sin": 0.951, "dow_cos": -0.309,
            }
        }


class PredictResponse(BaseModel):
    """Schéma de la réponse de prédiction."""
    action: str = Field(..., description="Action recommandée : BUY, SELL ou HOLD")
    confidence: float = Field(..., description="Confiance de la prédiction (0 à 1)")
    model_version: str = Field(..., description="Version du modèle utilisé")
    timestamp: str = Field(..., description="Horodatage de la prédiction")


class HealthResponse(BaseModel):
    """Schéma de la réponse de santé."""
    status: str
    model_version: str
    model_type: str


class ModelInfoResponse(BaseModel):
    """Schéma de la réponse d'info modèle."""
    version: str
    type: str
    algorithm: str
    metrics: dict
    created_at: str
    is_production: bool
    path: str


# =============================================================================
# Gestionnaire de modèles
# =============================================================================

class ModelManager:
    """Gère le chargement et l'utilisation des modèles ML et RL."""

    def __init__(self):
        self.model = None
        self.model_info = None
        self.model_version = None
        self.model_type = None
        self.scaler = None
        self._loaded = False

    def load_production_model(self):
        """Charge le modèle marqué 'production' dans le registry."""
        logger.info("Chargement du modèle de production...")

        # Lire le registry
        if not REGISTRY_PATH.exists():
            logger.warning(f"Registry introuvable : {REGISTRY_PATH}")
            raise FileNotFoundError(
                "Registry des modèles introuvable (models/registry.json). "
                "Exécutez d'abord le script d'update du registry."
            )

        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)

        # Trouver le modèle en production
        production_model = None
        for model_entry in registry.get("models", []):
            if model_entry.get("is_production", False):
                production_model = model_entry
                break

        if production_model is None:
            raise ValueError("Aucun modèle marqué 'is_production: true' dans le registry.")

        self.model_version = production_model["version"]
        self.model_type = production_model["type"]
        model_path = PROJECT_ROOT / production_model["path"]

        logger.info(f"Modèle en production : {self.model_version} ({self.model_type})")
        logger.info(f"Chemin : {model_path}")

        # Charger selon le type
        if self.model_type == "rl":
            self._load_rl_model(production_model)
        elif self.model_type == "ml":
            self._load_ml_model(production_model)
        else:
            raise ValueError(f"Type de modèle inconnu : {self.model_type}")

        # Charger les infos du modèle
        version_dir = MODELS_DIR / self.model_version
        info_path = version_dir / "model_info.json"
        if info_path.exists():
            with open(info_path, "r", encoding="utf-8") as f:
                self.model_info = json.load(f)
        else:
            self.model_info = production_model

        self._loaded = True
        logger.info(f"Modèle {self.model_version} chargé avec succès !")

    def _load_rl_model(self, model_entry: dict):
        """Charge un modèle RL (PPO via stable-baselines3)."""
        from stable_baselines3 import PPO

        model_path = PROJECT_ROOT / model_entry["path"]
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle RL introuvable : {model_path}")

        self.model = PPO.load(str(model_path))

        # Charger le scaler si disponible
        scaler_path = MODELS_DIR / self.model_version / "scaler.joblib"
        if scaler_path.exists():
            self.scaler = joblib.load(str(scaler_path))
            logger.info(f"Scaler chargé : {scaler_path}")

    def _load_ml_model(self, model_entry: dict):
        """Charge un modèle ML classique (pickle/joblib)."""
        model_path = PROJECT_ROOT / model_entry["path"]
        if not model_path.exists():
            raise FileNotFoundError(f"Modèle ML introuvable : {model_path}")

        loaded = joblib.load(str(model_path))

        # Le modèle peut être sauvé comme dict {"model": ..., "scaler": ...}
        if isinstance(loaded, dict):
            self.model = loaded["model"]
            if loaded.get("scaler") is not None:
                self.scaler = loaded["scaler"]
                logger.info("Scaler chargé depuis le dict du modèle")
            self._hold_threshold = loaded.get("hold_threshold", 0.55)
            self._needs_scaling = loaded.get("needs_scaling", False)
            logger.info(f"Modèle ML chargé : {loaded.get('model_name', 'unknown')}")
        else:
            self.model = loaded
            # Charger le scaler si disponible
            scaler_path = MODELS_DIR / self.model_version / "scaler.joblib"
            if scaler_path.exists():
                self.scaler = joblib.load(str(scaler_path))
                logger.info(f"Scaler chargé : {scaler_path}")

    def predict(self, features: np.ndarray) -> tuple:
        """
        Effectue une prédiction.

        Args:
            features: Array (1, 20) des features brutes

        Returns:
            action (str), confidence (float)
        """
        if not self._loaded:
            raise RuntimeError("Aucun modèle chargé.")

        # Normaliser si un scaler est disponible
        if self.scaler is not None:
            features = self.scaler.transform(features)

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        if self.model_type == "rl":
            return self._predict_rl(features)
        elif self.model_type == "ml":
            return self._predict_ml(features)
        else:
            raise ValueError(f"Type de modèle inconnu : {self.model_type}")

    def _predict_rl(self, features: np.ndarray) -> tuple:
        """Prédiction avec modèle RL (PPO)."""
        # Construire l'observation complète (20 features + position=0 + unrealized_pnl=0)
        obs = np.zeros(22, dtype=np.float32)
        obs[:20] = features[0]
        obs[20] = 0.0  # position flat par défaut
        obs[21] = 0.0  # pas de PnL non réalisé

        action, _ = self.model.predict(obs, deterministic=True)
        action = int(action)

        # Obtenir les probabilités d'action pour la confiance
        import torch
        obs_tensor = torch.as_tensor(obs.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            distribution = self.model.policy.get_distribution(obs_tensor)
            probs = distribution.distribution.probs.numpy()[0]

        confidence = float(probs[action])

        action_map = {0: "HOLD", 1: "BUY", 2: "SELL"}
        return action_map.get(action, "HOLD"), confidence

    def _predict_ml(self, features: np.ndarray) -> tuple:
        """Prédiction avec modèle ML classique + seuil HOLD."""
        threshold = getattr(self, "_hold_threshold", 0.55)

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(features)[0]
            proba_up = float(proba[1])  # probabilité de hausse
            confidence = float(np.max(proba))
        else:
            prediction = int(self.model.predict(features)[0])
            proba_up = float(prediction)
            confidence = 1.0

        # Décision avec seuil HOLD
        if proba_up >= threshold:
            action = "BUY"
            confidence = proba_up
        elif proba_up <= (1 - threshold):
            action = "SELL"
            confidence = 1 - proba_up
        else:
            action = "HOLD"
            confidence = 1 - abs(proba_up - 0.5) * 2  # confiance dans le HOLD

        return action, confidence

    @property
    def is_loaded(self) -> bool:
        return self._loaded


# =============================================================================
# Application FastAPI
# =============================================================================

app = FastAPI(
    title="Trading GBP/USD M15 — API",
    description=(
        "API de prédiction pour le système de trading GBP/USD en M15. "
        "Expose les modèles ML et RL pour des décisions BUY/SELL/HOLD."
    ),
    version="1.0.0",
)

# Instance du gestionnaire de modèles
model_manager = ModelManager()


@app.on_event("startup")
async def startup_event():
    """Charge le modèle de production au démarrage."""
    try:
        model_manager.load_production_model()
        logger.info("API démarrée avec succès — modèle chargé.")
    except Exception as e:
        logger.error(f"Impossible de charger le modèle : {e}")
        logger.warning("L'API démarre sans modèle. Les prédictions retourneront une erreur 503.")


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Santé"])
async def health_check():
    """
    Vérification de santé de l'API.
    Retourne le statut et la version du modèle chargé.
    """
    if model_manager.is_loaded:
        return HealthResponse(
            status="healthy",
            model_version=model_manager.model_version,
            model_type=model_manager.model_type,
        )
    else:
        return HealthResponse(
            status="degraded",
            model_version="none",
            model_type="none",
        )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Modèle"])
async def model_info():
    """
    Retourne les informations détaillées du modèle en production.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Aucun modèle chargé. Vérifiez le registry et les fichiers modèle.",
        )

    info = model_manager.model_info

    # Extraire les métriques selon le type
    if model_manager.model_type == "rl":
        metrics = info.get("metrics_validation", info.get("metrics", {}))
    else:
        metrics = info.get("metrics", {})

    return ModelInfoResponse(
        version=info.get("version", model_manager.model_version),
        type=info.get("type", model_manager.model_type),
        algorithm=info.get("algorithm", "unknown"),
        metrics=metrics,
        created_at=info.get("created_at", "unknown"),
        is_production=True,
        path=info.get("model_path", info.get("path", "unknown")),
    )


@app.post("/predict", response_model=PredictResponse, tags=["Prédiction"])
async def predict(request: PredictRequest):
    """
    Effectue une prédiction BUY/SELL/HOLD à partir des 20 features.

    L'API utilise le modèle en production (ML ou RL) pour analyser
    les conditions de marché et retourner une recommandation de trading.
    """
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Aucun modèle chargé. Le service n'est pas prêt.",
        )

    # Construire le vecteur de features
    # 20 features de base (pour RL) + 9 temporelles (pour ML)
    base_features = [
        request.return_1, request.return_4,
        request.ema_20, request.ema_50, request.ema_diff,
        request.rsi_14, request.rolling_std_20,
        request.range_15m, request.body,
        request.upper_wick, request.lower_wick,
        request.ema_200, request.distance_to_ema200,
        request.slope_ema50, request.atr_14,
        request.rolling_std_100, request.volatility_ratio,
        request.adx_14, request.macd, request.macd_signal,
    ]
    temporal_features = [
        request.hour, request.day_of_week,
        request.is_london, request.is_ny, request.is_overlap,
        request.hour_sin, request.hour_cos,
        request.dow_sin, request.dow_cos,
    ]

    if model_manager.model_type == "rl":
        # RL utilise 20 features
        features = np.array([base_features], dtype=np.float64)
    else:
        # ML utilise 29 features
        features = np.array([base_features + temporal_features], dtype=np.float64)

    try:
        action, confidence = model_manager.predict(features)
    except Exception as e:
        logger.error(f"Erreur de prédiction : {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction : {str(e)}",
        )

    return PredictResponse(
        action=action,
        confidence=round(confidence, 4),
        model_version=model_manager.model_version,
        timestamp=datetime.now().isoformat(),
    )


# =============================================================================
# Point d'entrée pour le développement
# =============================================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
