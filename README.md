# Lab 3 — LLM Analytics Agent

Веб-приложение + Telegram bot для анализа данных с помощью ИИ-агента на FastAPI.

## Возможности

- 📊 Загрузка датасетов (CSV, Excel, TSV, JSON, Parquet)
- 🤖 LLM-агент самостоятельно проводит анализ данных через выполнение Python-кода (tool use / function calling)
- 📈 Генерация графиков (matplotlib/seaborn) для визуализации результатов
- 📝 Детальный отчёт с ключевыми метриками, инсайтами и рекомендациями
- 🔒 Защита от prompt-injection
- 🧠 Возможность задать инструкции для анализа
- 🐳 Готовый Dockerfile для деплоя
- 📦 Использует **uv** для управления зависимостями (быстрая альтернатива pip)

## Как это работает

1. Пользователь загружает датасет и (опционально) инструкции
2. LLM-агент (GPT-4o-mini или другая модель) получает задачу
3. Агент пишет Python-код для анализа данных и выполняет его через защищённую песочницу
4. Результаты выполнения кода возвращаются агенту
5. Агент интерпретирует результаты и делает следующие шаги
6. Цикл повторяется до получения полного отчёта
7. Пользователь получает отчёт и графики

## Быстрый старт

### Способ 1: через uv (рекомендуется)

```bash
# 1. Установите uv (если ещё не установлен)
# curl -LsSf https://astral.sh/uv/install.sh | sh   # Linux/macOS
# powershell -c "irm https://astral.sh/uv/install.ps1 | iex"   # Windows

# 2. Клонируйте репозиторий
git clone <repo-url>
cd lab_3_analytics

# 3. Создайте .env и укажите API-ключ
cp .env.example .env
# Отредактируйте .env: LLM_API_KEY=sk-...

# 4. Установите зависимости (uv сам создаст виртуальное окружение)
uv sync

# 5. Активируйте окружение
# source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate      # Windows

# 6. Запустите веб-интерфейс
uvicorn app.main:app --reload
# Или Telegram-бота:
python -m app.bot
```

### Способ 2: через pip

```bash
git clone <repo-url>
cd lab_3_analytics

cp .env.example .env
# Отредактируйте .env: LLM_API_KEY и TELEGRAM_BOT_TOKEN

pip install -e .
uvicorn app.main:app --reload  # веб-интерфейс
# python -m app.bot            # Telegram-бот
```

### Способ 3: через Docker

```bash
# Соберите образ
docker build -t llm-analytics-agent .

# Веб-интерфейс:
docker run -p 8000:8000 --env-file .env llm-analytics-agent

# Telegram-бот (переопределяем команду):
docker run --env-file .env llm-analytics-agent python -m app.bot
```

## 🎮 Использование

### Веб-интерфейс
1. Откройте http://localhost:8000
2. Загрузите датасет (CSV/Excel/TSV/JSON/Parquet)
3. Опционально укажите инструкции для анализа
4. Нажмите "Запустить анализ"
5. Дождитесь завершения — агент выполнит анализ и покажет отчёт

### Telegram-бот
1. Найдите бота в Telegram (@YourBotName)
2. Отправьте команду `/start`
3. Отправьте CSV/Excel файл
4. Опционально добавьте подпись с инструкциями
5. Получите отчёт и графики

## Структура проекта

```
lab_3_analytics/
├── app/
│   ├── __init__.py
│   ├── agent.py           # LLM-агент + песочница для кода
│   ├── config.py          # Конфигурация из .env
│   ├── main.py            # FastAPI приложение
│   └── templates/
│       ├── index.html     # Главная страница с загрузкой
│       └── result.html    # Страница с отчётом и графиками
├── data/
│   ├── uploads/           # Загруженные датасеты
│   └── output/            # Результаты и графики
├── .env                   # Файл с настройками (НЕ коммитить!)
├── .env.example           # Пример конфигурации
├── .gitignore
├── Dockerfile             # Контейнеризация
├── pyproject.toml         # Зависимости проекта
├── README.md
└── uv.lock                # Lock-файл для воспроизводимости
```

## Развёртывание

### Локально (для разработки)
```bash
uv sync
uvicorn app.main:app --reload
```

### Docker (для production)
```bash
docker build -t llm-analytics-agent .
docker run -d -p 8000:8000 --env-file .env llm-analytics-agent
```

### Бесплатные хостинги
- **Render.com** — Docker-деплой из Git-репозитория
- **Railway.app** — автодеплой по Dockerfile
- **Hugging Face Spaces** — поддерживает FastAPI
- **fly.io** — бесплатный лимит для тестов

## Требования

- Python >= 3.10
- API ключ OpenAI (или совместимого API: DeepSeek, Qwen, LM Studio)
- **uv** (рекомендуется) или pip
