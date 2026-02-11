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

# Répertoire de travail
WORKDIR /app

# Installer les dépendances système nécessaires
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier et installer les dépendances Python
COPY docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

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
