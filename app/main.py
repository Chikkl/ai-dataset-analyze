"""FastAPI приложение для анализа данных с LLM-агентом."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.agent import LLMAnalyticsAgent
from app.config import config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="LLM Analytics Agent",
    description="Веб-интерфейс для анализа данных с помощью ИИ-агента",
    version="1.0.0",
)

# ---------- templates & static ----------

templates = Jinja2Templates(directory="app/templates")

# ---------- хранилище сессий ----------

sessions: Dict[str, Dict] = {}


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


# ---------- routes ----------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Главная страница."""
    return templates.TemplateResponse(
        "index.html", {"request": request, "session_id": ""}
    )


@app.post("/upload")
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    instructions: str = Form(""),
):
    """Загрузить датасет и запустить анализ."""
    # Проверка размера файла
    contents = await file.read()
    if len(contents) > config.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Файл слишком большой. Максимум {config.max_upload_size_mb} MB.",
        )

    # Проверка расширения
    allowed_extensions = {".csv", ".xls", ".xlsx", ".tsv", ".json", ".parquet"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Неподдерживаемый формат файла: {ext}. "
                   f"Допустимые: {', '.join(allowed_extensions)}",
        )

    # Создаём сессию
    session_id = str(uuid.uuid4())

    # Сохраняем файл
    session_dir = config.upload_dir / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    file_path = session_dir / file.filename
    with open(file_path, "wb") as f:
        f.write(contents)

    # Создаём агента и запускаем анализ
    agent = get_agent(session_id)
    sessions[session_id]["dataset_path"] = file_path

    try:
        result = agent.analyze(
            dataset_path=file_path,
            user_instructions=instructions,
        )
    except Exception as e:
        logger.exception("Ошибка при анализе данных")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при анализе данных: {str(e)}",
        )

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "session_id": session_id,
            "report": result["report"],
            "charts": result["charts"],
            "iterations": result["iterations"],
            "filename": file.filename,
        },
    )


@app.get("/result/{session_id}", response_class=HTMLResponse)
async def get_result(request: Request, session_id: str):
    """Показать результат анализа для сессии."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")

    # Повторно запускаем анализ, если результат не сохранился
    agent = get_agent(session_id)
    dataset_path = sessions[session_id].get("dataset_path")
    if not dataset_path or not dataset_path.exists():
        raise HTTPException(status_code=404, detail="Датасет не найден")

    try:
        result = agent.analyze(dataset_path=dataset_path)
    except Exception as e:
        logger.exception("Ошибка при анализе данных")
        raise HTTPException(
            status_code=500,
            detail=f"Ошибка при анализе данных: {str(e)}",
        )

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "session_id": session_id,
            "report": result["report"],
            "charts": result["charts"],
            "iterations": result["iterations"],
            "filename": dataset_path.name,
        },
    )


@app.post("/cleanup/{session_id}")
async def cleanup_session(session_id: str):
    """Очистить сессию."""
    if session_id in sessions:
        agent = sessions[session_id].get("agent")
        if agent:
            agent.close()
            agent.cleanup()
        # Удаляем файлы
        session_dir = config.upload_dir / session_id
        if session_dir.exists():
            import shutil
            shutil.rmtree(session_dir)
        output_dir = config.output_dir / session_id
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)
        del sessions[session_id]
    return JSONResponse({"status": "ok"})


@app.on_event("shutdown")
async def shutdown():
    """Очистить ресурсы при завершении."""
    for session_id in list(sessions.keys()):
        agent = sessions[session_id].get("agent")
        if agent:
            agent.close()
    sessions.clear()


def run() -> None:
    """Запуск через: python -m app.main или app"""
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=True,
    )


if __name__ == "__main__":
    run()