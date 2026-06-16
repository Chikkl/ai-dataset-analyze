"""Telegram-бот для анализа данных."""

from __future__ import annotations

import base64
import io
import logging
import os
import uuid
from pathlib import Path
from typing import Any, Dict

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

from app.agent import LLMAnalyticsAgent
from app.config import config

logger = logging.getLogger(__name__)

sessions: Dict[str, Dict[str, Any]] = {}
USER_SESSION_MAP: Dict[int, str] = {}


def get_agent(session_id: str) -> LLMAnalyticsAgent:
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
    await update.message.reply_text(
        "🤖 <b>LLM Analytics</b>\n\n"
        "Отправь CSV/Excel файл для анализа данных.\n\n"
        "Подпись к файлу = инструкции для агента.\n"
        "Форматы: <code>.csv</code>, <code>.xls</code>, <code>.xlsx</code>, "
        "<code>.tsv</code>, <code>.json</code>, <code>.parquet</code>",
        parse_mode="HTML",
    )


async def handle_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = update.effective_user.id
    document = update.message.document
    file_name = document.file_name or "dataset.csv"

    allowed_extensions = {".csv", ".xls", ".xlsx", ".tsv", ".json", ".parquet"}
    ext = Path(file_name).suffix.lower()
    if ext not in allowed_extensions:
        await update.message.reply_text(
            f"Unsupported format: <code>{ext}</code>",
            parse_mode="HTML",
        )
        return

    instructions = update.message.caption or ""

    await update.message.reply_text("📥 Processing...")

    session_id = str(uuid.uuid4())
    USER_SESSION_MAP[user_id] = session_id

    session_dir = config.upload_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    file_path = session_dir / file_name

    try:
        file = await document.get_file()
        await file.download_to_drive(str(file_path))
    except Exception as e:
        logger.exception("Download error")
        await update.message.reply_text(f"Error: {e}")
        return

    agent = get_agent(session_id)
    sessions[session_id]["dataset_path"] = file_path

    try:
        result = agent.analyze(
            dataset_path=file_path,
            user_instructions=instructions,
        )
    except Exception as e:
        logger.exception("Analysis error")
        await update.message.reply_text(f"Analysis error: {e}")
        return

    report = result["report"]
    if len(report) > 4000:
        report = report[:4000] + "\n..."

    await update.message.reply_text(
        f"📊 Done — {file_name}\nIterations: {result['iterations']}\n\n{report}",
        parse_mode="HTML",
    )

    charts = result["charts"]
    if charts:
        for chart in charts:
            try:
                chart_bytes = base64.b64decode(chart["data"])
                await update.message.reply_photo(
                    photo=io.BytesIO(chart_bytes),
                    filename=chart["filename"],
                )
            except Exception as e:
                logger.warning("Chart error %s: %s", chart["filename"], e)

    agent.cleanup()
    if session_id in sessions:
        del sessions[session_id]
    if user_id in USER_SESSION_MAP:
        del USER_SESSION_MAP[user_id]


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "1. Send CSV/Excel file\n"
        "2. Optional caption = analysis instructions\n"
        "3. Wait 1-2 min → report + charts"
    )


async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    logger.error("Bot error: %s", context.error)


def run() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if not token:
        token = input("TELEGRAM_BOT_TOKEN: ").strip()
    if not token:
        logger.error("No TELEGRAM_BOT_TOKEN")
        return

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_document))
    app.add_error_handler(error_handler)

    logger.info("Bot started")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run()