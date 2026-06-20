"""
Llama-Nexus — self-hosted FastAPI backend.

This service does **zero** inference. All model execution happens client-side in
the browser on WebGPU (see app/static/llama_core.js). The backend exists purely to:

  1. Serve the single-page dashboard and the Web Worker / static assets, with the
     cross-origin isolation headers (COOP/COEP) that WebGPU + multi-threaded WASM
     fallbacks require.
  2. Persist conversation logs and reusable prompt templates to a local SQLite
     database so your data never leaves the machine.

Cost to run: $0.00. No API keys, no cloud GPU, no egress.
"""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

# --------------------------------------------------------------------------- #
# Paths & configuration
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Allow override so the Dockerfile can mount a persistent volume.
import os

DB_PATH = Path(os.environ.get("NEXUS_DB_PATH", DATA_DIR / "nexus.db"))

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# --------------------------------------------------------------------------- #
# Database layer (stdlib sqlite3 — no ORM, no extra deps)
# --------------------------------------------------------------------------- #

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    title       TEXT NOT NULL DEFAULT 'Untitled session',
    model_id    TEXT,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role            TEXT NOT NULL CHECK (role IN ('system', 'user', 'assistant')),
    content         TEXT NOT NULL,
    -- Optional client-side telemetry captured per assistant turn.
    tokens_per_sec  REAL,
    has_image       INTEGER NOT NULL DEFAULT 0,
    has_audio       INTEGER NOT NULL DEFAULT 0,
    created_at      REAL NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations (id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages (conversation_id, created_at);

CREATE TABLE IF NOT EXISTS prompt_templates (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    body        TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL
);
"""

# A couple of ready-to-use system prompts seeded on first boot.
SEED_TEMPLATES = [
    (
        "Concise Researcher",
        "You are Llama-Nexus, a precise, citation-minded research assistant. "
        "Answer directly, show your reasoning briefly, and never invent facts.",
    ),
    (
        "Vision Analyst",
        "You are a meticulous visual analyst. Describe what is verifiably present "
        "in the image, separate observation from inference, and quantify uncertainty.",
    ),
]


@contextmanager
def get_db() -> Iterator[sqlite3.Connection]:
    """Yield a connection with foreign keys on and Row access by column name."""
    conn = sqlite3.connect(DB_PATH, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_db() as conn:
        conn.executescript(SCHEMA)
        existing = conn.execute("SELECT COUNT(*) AS n FROM prompt_templates").fetchone()["n"]
        if existing == 0:
            now = time.time()
            conn.executemany(
                "INSERT INTO prompt_templates (id, name, body, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                [(str(uuid.uuid4()), name, body, now, now) for name, body in SEED_TEMPLATES],
            )


# --------------------------------------------------------------------------- #
# Pydantic schemas
# --------------------------------------------------------------------------- #


class MessageIn(BaseModel):
    role: str = Field(..., pattern="^(system|user|assistant)$")
    content: str
    tokens_per_sec: Optional[float] = None
    has_image: bool = False
    has_audio: bool = False


class ConversationCreate(BaseModel):
    title: str = "Untitled session"
    model_id: Optional[str] = None
    messages: list[MessageIn] = Field(default_factory=list)


class TemplateIn(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    body: str = Field(..., min_length=1)


# --------------------------------------------------------------------------- #
# App + lifecycle
# --------------------------------------------------------------------------- #

app = FastAPI(
    title="Llama-Nexus",
    description="Self-hosted, $0.00, browser-WebGPU multimodal inference for Meta's open models.",
    version="1.0.0",
)


@app.on_event("startup")
def _startup() -> None:
    init_db()


class CrossOriginIsolationMiddleware:
    """
    WebGPU and the multi-threaded WASM fallback require the document to be
    *cross-origin isolated*. That means every response — including the HTML and
    the worker script — must carry COOP/COEP headers. We add them globally here.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message):
            if message["type"] == "http.response.start":
                headers = message.setdefault("headers", [])
                headers.append((b"cross-origin-opener-policy", b"same-origin"))
                headers.append((b"cross-origin-embedder-policy", b"require-corp"))
                # Allow the worker to pull model weights / WASM from the HF + jsDelivr CDNs.
                headers.append((b"cross-origin-resource-policy", b"cross-origin"))
            await send(message)

        await self.app(scope, receive, send_with_headers)


app.add_middleware(CrossOriginIsolationMiddleware)

# Static assets (the worker, app shell JS, etc.).
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --------------------------------------------------------------------------- #
# Routes — UI
# --------------------------------------------------------------------------- #


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "db": str(DB_PATH), "ts": time.time()}


# --------------------------------------------------------------------------- #
# Routes — conversations
# --------------------------------------------------------------------------- #


@app.get("/api/conversations")
def list_conversations() -> JSONResponse:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, title, model_id, created_at, updated_at "
            "FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return JSONResponse([dict(r) for r in rows])


@app.post("/api/conversations", status_code=201)
def create_conversation(payload: ConversationCreate) -> JSONResponse:
    conv_id = str(uuid.uuid4())
    now = time.time()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, model_id, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, payload.title, payload.model_id, now, now),
        )
        for m in payload.messages:
            _insert_message(conn, conv_id, m, now)
    return JSONResponse({"id": conv_id}, status_code=201)


@app.get("/api/conversations/{conv_id}")
def get_conversation(conv_id: str) -> JSONResponse:
    with get_db() as conn:
        conv = conn.execute(
            "SELECT id, title, model_id, created_at, updated_at FROM conversations WHERE id = ?",
            (conv_id,),
        ).fetchone()
        if conv is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        msgs = conn.execute(
            "SELECT id, role, content, tokens_per_sec, has_image, has_audio, created_at "
            "FROM messages WHERE conversation_id = ? ORDER BY created_at",
            (conv_id,),
        ).fetchall()
    out = dict(conv)
    out["messages"] = [dict(m) for m in msgs]
    return JSONResponse(out)


@app.post("/api/conversations/{conv_id}/messages", status_code=201)
def append_message(conv_id: str, message: MessageIn) -> JSONResponse:
    now = time.time()
    with get_db() as conn:
        conv = conn.execute(
            "SELECT id FROM conversations WHERE id = ?", (conv_id,)
        ).fetchone()
        if conv is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        msg_id = _insert_message(conn, conv_id, message, now)
        conn.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conv_id)
        )
    return JSONResponse({"id": msg_id}, status_code=201)


@app.delete("/api/conversations/{conv_id}", status_code=204)
def delete_conversation(conv_id: str) -> JSONResponse:
    with get_db() as conn:
        cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="conversation not found")
    return JSONResponse(None, status_code=204)


def _insert_message(
    conn: sqlite3.Connection, conv_id: str, m: MessageIn, ts: float
) -> str:
    msg_id = str(uuid.uuid4())
    conn.execute(
        "INSERT INTO messages "
        "(id, conversation_id, role, content, tokens_per_sec, has_image, has_audio, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            msg_id,
            conv_id,
            m.role,
            m.content,
            m.tokens_per_sec,
            int(m.has_image),
            int(m.has_audio),
            ts,
        ),
    )
    return msg_id


# --------------------------------------------------------------------------- #
# Routes — prompt templates
# --------------------------------------------------------------------------- #


@app.get("/api/templates")
def list_templates() -> JSONResponse:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, name, body, created_at, updated_at "
            "FROM prompt_templates ORDER BY name"
        ).fetchall()
    return JSONResponse([dict(r) for r in rows])


@app.post("/api/templates", status_code=201)
def create_template(payload: TemplateIn) -> JSONResponse:
    tpl_id = str(uuid.uuid4())
    now = time.time()
    with get_db() as conn:
        try:
            conn.execute(
                "INSERT INTO prompt_templates (id, name, body, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (tpl_id, payload.name, payload.body, now, now),
            )
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="template name already exists")
    return JSONResponse({"id": tpl_id}, status_code=201)


@app.delete("/api/templates/{tpl_id}", status_code=204)
def delete_template(tpl_id: str) -> JSONResponse:
    with get_db() as conn:
        cur = conn.execute("DELETE FROM prompt_templates WHERE id = ?", (tpl_id,))
        if cur.rowcount == 0:
            raise HTTPException(status_code=404, detail="template not found")
    return JSONResponse(None, status_code=204)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=os.environ.get("NEXUS_HOST", "127.0.0.1"),
        port=int(os.environ.get("NEXUS_PORT", "8000")),
        reload=bool(os.environ.get("NEXUS_RELOAD")),
    )
