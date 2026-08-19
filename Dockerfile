FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# libgomp is needed by scikit-learn's histogram gradient boosting and LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[lightgbm]"

COPY configs ./configs

RUN useradd --create-home appuser
USER appuser

ENV FRAUD_BUNDLE=/app/artifacts \
    FRAUD_RULES=/app/configs/rules.yaml

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "fraud_detection.api.app:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
