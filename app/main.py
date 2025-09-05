# app/main.py
from __future__ import annotations
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query

from . import db
from .nlp import analyze_text
from .schemas import AnalyzeIn

# Lifespan handler replaces @app.on_event("startup")
@asynccontextmanager
async def lifespan(app: FastAPI):
    db.init_db()        # startup
    yield               # app runs
    # (no shutdown work for now)

app = FastAPI(
    title="LLM Knowledge Extractor – Jouster",
    version="0.1.0",
    lifespan=lifespan,
)

@app.post("/analyze")
def analyze(payload: AnalyzeIn):
    try:
        items = []
        if payload.text is not None:
            items.append(analyze_text(payload.text, title=payload.title))
        elif payload.texts is not None:
            if not payload.texts:
                raise ValueError("texts list is empty")
            for t in payload.texts:
                items.append(analyze_text(t))
        else:
            raise ValueError("Provide 'text' or 'texts'")

        ids = []
        for item in items:
            new_id = db.insert_analysis(item)
            ids.append(new_id)

        rows = db.search()
        out_items = [{
            "id": r["id"],
            "title": r.get("title"),
            "summary": r["summary"],
            "topics": r["topics"],
            "sentiment": r["sentiment"],
            "keywords": r["keywords"],
            "confidence": r["confidence"],
            "created_at": r["created_at"],
        } for r in rows[:len(ids)]]

        return {"items": out_items}
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as ex:
        raise HTTPException(status_code=500, detail=f"analysis_error: {ex.__class__.__name__}: {ex}")

@app.get("/search")
def search(topic: Optional[str] = None, keyword: Optional[str] = None):
    rows = db.search(topic=topic, keyword=keyword)
    out = [{
        "id": r["id"],
        "title": r.get("title"),
        "summary": r["summary"],
        "topics": r["topics"],
        "sentiment": r["sentiment"],
        "keywords": r["keywords"],
        "confidence": r["confidence"],
        "created_at": r["created_at"],
    } for r in rows]
    return {"items": out}

@app.get("/")
def root():
    return {"ok": True, "message": "See /docs for Swagger"}
