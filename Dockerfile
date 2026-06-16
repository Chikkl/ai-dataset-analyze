# ============================================
# Dockerfile для LLM Analytics Agent
# ============================================
# Базовый образ: официальный Python 3.12 (slim)
FROM python:3.12-slim

# Устанавливаем системные зависимости для matplotlib, pandas, lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем только файлы зависимостей (кэширование слоёв Docker)
COPY pyproject.toml uv.lock README.md ./

# Устанавливаем uv
RUN pip install uv

# Устанавливаем зависимости проекта через uv (без дев-зависимостей)
RUN uv sync --no-dev

# Копируем исходный код приложения
COPY app/ ./app/

# Создаём директории для данных
RUN mkdir -p data/uploads data/output

# Порт, на котором работает приложение
EXPOSE 8000

# Переменные окружения по умолчанию (переопределяются через --env-file)
ENV LLM_API_URL=https://api.openai.com/v1/chat/completions
ENV LLM_MODEL=gpt-4o-mini
ENV LLM_TEMPERATURE=0.1
ENV LLM_MAX_TOKENS=4096
ENV HOST=0.0.0.0
ENV PORT=8000
ENV MAX_UPLOAD_SIZE_MB=50

# Запускаем приложение через uvicorn (без reload в production)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]