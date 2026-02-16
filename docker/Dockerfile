# =============================================================================
# T12 — Dockerfile pour l'API de trading GBP/USD M15
# =============================================================================
# Image de base : Python 3.11 slim pour un bon compromis taille/compatibilité
# L'API charge le modèle de production (ML ou RL) et expose les endpoints
# sur le port 8000.
# =============================================================================

FROM python:3.11-slim

# Métadonnées
LABEL maintainer="DataScience Team"
LABEL description="API de trading GBP/USD M15 — Modèles ML & RL"
LABEL version="1.0.0"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_DEFAULT_TIMEOUT=300

# Répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Couche 1 : API (leger)
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" pydantic

# Couche 2 : Data processing
RUN pip install --no-cache-dir numpy pandas pyarrow joblib

# Couche 3 : scikit-learn (necessaire pour les modeles ML)
RUN pip install --no-cache-dir scikit-learn

# Couche 4 : xgboost CPU only (sans nvidia-nccl)
RUN pip install --no-cache-dir "xgboost<3" lightgbm

# Couche 5 : PyTorch CPU (version legere pour stable-baselines3)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Couche 6 : Reinforcement Learning
RUN pip install --no-cache-dir stable-baselines3 gymnasium

# Copier le code de l'API
COPY api/ /app/api/

# Copier les modèles et le registry
COPY models/ /app/models/

# Copier les features (nécessaires si l'API doit accéder aux données)
COPY features/ /app/features/

# Exposer le port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Commande de démarrage
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
