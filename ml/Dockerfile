# Dockerfile для ML-сервиса (FastAPI + LightGBM)
FROM python:3.10-slim

# Метаданные
LABEL maintainer="mos_transp team"
LABEL description="ML-сервис прогнозирования пассажиропотока"

# Рабочая директория
WORKDIR /app

# Устанавливаем системные зависимости (нужны для LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Копируем requirements и устанавливаем Python-зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код и артефакты
COPY ml_service.py .
COPY artifacts/ ./artifacts/

# Открываем порт
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Запуск сервиса (без --reload для продакшена)
CMD ["uvicorn", "ml_service:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]