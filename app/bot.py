"""Telegram-бот для анализа данных с помощью LLM-агента."""

from __future__ import annotations

import base64
import io
import logging
import os
import uuid
from pathlib import Path
from typing import Dict

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from app.agent import LLMAnalyticsAgent
from app.config import config

logger = logging.getLogger(__name__)

# ---------- хранилище сессий ----------

sessions: Dict[str, Dict] = {}
USER_SESSION_MAP: Dict[int, str] = {}


def get_agent(session_id: str) -> LLMAnalyticsAgent:
    """Получить или создать агента для сессии."""
    if session_id not in sessions:
        charts_dir = config.output_dir / session_id / "charts"
        charts_dir.mkdir(parents=True, exist_ok=True)

        agent = LLMAnalyticsAgent(
            api_url=config.llm_api_url,
            api_key=config.llm_api_key,
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            data_dir=config.upload_dir / session_id,
            charts_dir=charts_dir,
        )
        sessions[session_id] = {"agent": agent, "dataset_path": None}
    return sessions[session_id]["agent"]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /start."""
    await update.message.reply_text(
        "🤖 <b>LLM Analytics Agent</b>\n\n"
        "Привет! Я AI-агент для анализа данных.\n\n"
        "📤 <b>Отправь мне CSV или Excel файл</b> — я проведу полный анализ:\n"
        "• Загрузка и очистка данных\n"
        "• Разведочный анализ (EDA)\n"
        "• Статистические метрики\n"
        "• Графики и визуализации\n"
        "• Выводы и инсайты\n\n"
        "Также ты можешь написать <b>инструкции</b> в подписи к файлу — "
        "на что обратить внимание.\n\n"
        "Поддерживаемые форматы: <code>.csv</code>, <code>.xls</code>, <code>.xlsx</code>, "
        "<code>.tsv</code>, <code>.json</code>, <code>.parquet</code>",
        parse_mode="HTML",
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик загруженного файла."""
    user_id = update.effective_user.id
    document = update.message.document
    file_name = document.file_name or "dataset.csv"

    # Проверка расширения
    allowed_extensions = {".csv", ".xls", ".xlsx", ".tsv", ".json", ".parquet"}
    ext = Path(file_name).suffix.lower()
    if ext not in allowed_extensions:
        await update.message.reply_text(
            f"❌ Неподдерживаемый формат: <code>{ext}</code>\n"
            f"Допустимые: {', '.join(allowed_extensions)}",
            parse_mode="HTML",
        )
        return

    # Получаем инструкции из подписи к файлу
    instructions = update.message.caption or ""

    await update.message.reply_text(
        "📥 Файл получен! Запускаю анализ...\n"
        "⏳ Это может занять до 1-2 минут.",
    )

    # Скачиваем файл
    session_id = str(uuid.uuid4())
    USER_SESSION_MAP[user_id] = session_id

    session_dir = config.upload_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    file_path = session_dir / file_name

    try:
        file = await document.get_file()
        await file.download_to_drive(str(file_path))
    except Exception as e:
        logger.exception("Ошибка при скачивании файла")
        await update.message.reply_text(f"❌ Ошибка при скачивании файла: {str(e)}")
        return

    # Запускаем агента
    agent = get_agent(session_id)
    sessions[session_id]["dataset_path"] = file_path

    try:
        result = agent.analyze(
            dataset_path=file_path,
            user_instructions=instructions,
        )
    except Exception as e:
        logger.exception("Ошибка при анализе данных")
        await update.message.reply_text(
            f"❌ Ошибка при анализе данных: {str(e)[:200]}"
        )
        return

    # Форматируем отчёт
    report = result["report"]
    if len(report) > 4000:
        report = report[:4000] + "\n\n... (отчёт обрезан)"

    # Отправляем отчёт
    await update.message.reply_text(
        f"📊 <b>Анализ завершён!</b>\n"
        f"Файл: {file_name}\n"
        f"Итераций агента: {result['iterations']}\n\n"
        f"{report}",
        parse_mode="HTML",
    )

    # Отправляем графики (если есть)
    charts = result["charts"]
    if charts:
        await update.message.reply_text("📈 <b>Сгенерированные графики:</b>", parse_mode="HTML")
        for chart in charts:
            try:
                chart_bytes = base64.b64decode(chart["data"])
                await update.message.reply_photo(
                    photo=io.BytesIO(chart_bytes),
                    filename=chart["filename"],
                )
            except Exception as e:
                logger.warning("Не удалось отправить график %s: %s", chart["filename"], e)
    else:
        await update.message.reply_text("📉 Графики не были сгенерированы.")

    # Очищаем сессию
    agent.cleanup()
    if session_id in sessions:
        del sessions[session_id]
    if user_id in USER_SESSION_MAP:
        del USER_SESSION_MAP[user_id]


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик команды /help."""
    await update.message.reply_text(
        "ℹ️ <b>Как пользоваться ботом:</b>\n\n"
        "1. Отправь CSV или Excel файл\n"
        "2. Опционально добавь подпись с инструкциями\n"
        "3. Дождись анализа (1-2 минуты)\n"
        "4. Получи отчёт и графики\n\n"
        "<b>Пример подписи к файлу:</b>\n"
        "<code>Обрати внимание на сезонность продаж, "
        "построй прогноз на следующий период</code>",
        parse_mode="HTML",
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Обработчик ошибок."""
    logger.error("Ошибка в боте: %s", context.error)


def run() -> None:
    """Запуск Telegram-бота."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        token = input("Введите TELEGRAM_BOT_TOKEN: ").strip()
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN не указан")
        return

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))

    app.add_error_handler(error_handler)

    logger.info("Telegram-бот запущен...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run()