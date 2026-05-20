"""
Lance's AI API  v2.0
────────────────────
Endpoints : GET /          HTML landing page
            GET /health    health check (JSON)
            GET /models    show available providers
            POST /ask      single-turn Q&A  (Claude or OpenAI)
            POST /chat     multi-turn chat   (Claude or OpenAI)
            GET /docs      auto-generated Swagger UI

Auth       : X-API-Key header  (set SERVICE_API_KEY env var; empty = open in dev)
Rate limit : 20 requests / minute per IP
Providers  : claude (claude-haiku-4-5)  |  openai (gpt-4o-mini)
"""

import os
import uuid
import pathlib
from datetime import datetime, timezone
from dotenv import load_dotenv

_BASE = pathlib.Path(__file__).parent

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import anthropic
import openai

load_dotenv()

# ── Clients ────────────────────────────────────────────────────────────────────
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ── Auth ───────────────────────────────────────────────────────────────────────
_SERVICE_KEY = os.getenv("SERVICE_API_KEY", "")          # empty = no auth (local dev)
_key_header  = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_key(key: str = Depends(_key_header)):
    if _SERVICE_KEY and key != _SERVICE_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")
    return key

# ── Rate limiting ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["20/minute"])

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Lance's AI API",
    description=(
        "A production-ready AI API supporting Claude (Anthropic) and GPT (OpenAI). "
        "Built by Lance Galicia — AI Engineer & RAG Systems Builder."
    ),
    version="2.0",
    docs_url=None,    # we serve our own branded /docs
    redoc_url=None,   # disable redoc
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS — allow any frontend to call the API from the browser ─────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── In-memory chat sessions ────────────────────────────────────────────────────
_sessions: dict[str, list] = {}
_MAX_HISTORY = 20

# ── Schemas ────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    context:  str = ""
    provider: str = "claude"   # "claude" | "openai"
    system:   str = ""         # optional custom system prompt

    @validator("question")
    def validate_question(cls, v):
        v = v.strip()
        if not v:            raise ValueError("question cannot be empty")
        if len(v) > 4000:    raise ValueError("question too long (max 4000 chars)")
        return v

    @validator("provider")
    def validate_provider(cls, v):
        if v not in ("claude", "openai"):
            raise ValueError("provider must be 'claude' or 'openai'")
        return v

    @validator("system")
    def validate_system(cls, v):
        if len(v) > 2000:    raise ValueError("system prompt too long (max 2000 chars)")
        return v.strip()


class AskResponse(BaseModel):
    answer:      str
    provider:    str
    tokens_used: int


class ChatRequest(BaseModel):
    message:    str
    session_id: str = ""       # omit to start a new session
    provider:   str = "claude"
    system:     str = ""       # optional custom system prompt (applied to whole session)

    @validator("message")
    def validate_message(cls, v):
        v = v.strip()
        if not v:            raise ValueError("message cannot be empty")
        if len(v) > 4000:    raise ValueError("message too long (max 4000 chars)")
        return v

    @validator("provider")
    def validate_provider(cls, v):
        if v not in ("claude", "openai"):
            raise ValueError("provider must be 'claude' or 'openai'")
        return v

    @validator("system")
    def validate_system(cls, v):
        if len(v) > 2000:    raise ValueError("system prompt too long (max 2000 chars)")
        return v.strip()


class ChatResponse(BaseModel):
    reply:       str
    session_id:  str
    provider:    str
    tokens_used: int


# ── AI helpers ─────────────────────────────────────────────────────────────────
_SYSTEM = "You are a helpful, precise AI assistant. Answer clearly and concisely."


def _ask_claude(messages: list, system: str = _SYSTEM) -> tuple[str, int]:
    try:
        r = claude_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=800,
            system=system,
            messages=messages,
        )
        tokens = r.usage.input_tokens + r.usage.output_tokens
        return r.content[0].text.strip(), tokens
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=502, detail="Claude API key is invalid or missing.")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Claude rate limit reached. Try again shortly.")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error calling Claude: {str(e)}")


def _ask_openai(messages: list, system: str = _SYSTEM) -> tuple[str, int]:
    try:
        full = [{"role": "system", "content": system}] + messages
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=800,
            messages=full,
        )
        tokens = r.usage.prompt_tokens + r.usage.completion_tokens
        return r.choices[0].message.content.strip(), tokens
    except openai.AuthenticationError:
        raise HTTPException(status_code=502, detail="OpenAI API key is invalid or missing.")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI rate limit reached. Try again shortly.")
    except openai.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error calling OpenAI: {str(e)}")


def _route(provider: str, messages: list, system: str = _SYSTEM) -> tuple[str, int]:
    if provider == "openai":
        return _ask_openai(messages, system)
    return _ask_claude(messages, system)


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Health"])
def health():
    return {
        "status":    "live",
        "version":   "2.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "providers": {
            "claude": bool(os.getenv("ANTHROPIC_API_KEY")),
            "openai": bool(os.getenv("OPENAI_API_KEY")),
        },
        "sessions_active": len(_sessions),
        "endpoints": {
            "GET  /":              "landing page",
            "GET  /health":        "health check (JSON)",
            "GET  /models":        "available providers",
            "POST /ask":           "single-turn Q&A",
            "POST /chat":          "multi-turn conversation",
            "GET  /session/{id}":  "view conversation history",
            "DELETE /session/{id}":"clear a conversation",
            "GET  /docs":          "interactive API docs",
        },
    }


# ── Custom Swagger UI ──────────────────────────────────────────────────────────
_DOCS_HTML = (_BASE / "templates" / "docs.html").read_text(encoding="utf-8")


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def custom_docs():
    return HTMLResponse(content=_DOCS_HTML)


# ── Landing page ───────────────────────────────────────────────────────────────
_LANDING_HTML = (_BASE / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse, tags=["Landing"], include_in_schema=False)
def root():
    return HTMLResponse(content=_LANDING_HTML)


@app.get("/models", tags=["Info"])
def models():
    return {
        "claude": {
            "model":     "claude-haiku-4-5",
            "available": bool(os.getenv("ANTHROPIC_API_KEY")),
        },
        "openai": {
            "model":     "gpt-4o-mini",
            "available": bool(os.getenv("OPENAI_API_KEY")),
        },
    }


@app.post("/ask", response_model=AskResponse, tags=["AI"])
@limiter.limit("20/minute")
def ask(
    request: Request,
    payload: AskRequest,
    _: str = Depends(require_key),
):
    """
    Single-turn Q&A. Send a question, get an answer.
    Optionally pass `context` to ground the answer in specific information.
    Choose `provider`: **claude** (default) or **openai**.
    """
    system = payload.system if payload.system else _SYSTEM
    if payload.context:
        system += f"\n\nUse only this context to answer:\n{payload.context}"

    answer, tokens = _route(
        payload.provider,
        [{"role": "user", "content": payload.question}],
        system,
    )
    return AskResponse(answer=answer, provider=payload.provider, tokens_used=tokens)


@app.post("/chat", response_model=ChatResponse, tags=["AI"])
@limiter.limit("20/minute")
def chat(
    request: Request,
    payload: ChatRequest,
    _: str = Depends(require_key),
):
    """
    Multi-turn conversation with memory.
    Omit `session_id` to start a new session — the ID is returned in the response.
    Pass the same `session_id` on follow-up messages to continue the conversation.
    History is capped at the last 20 messages (in-memory, resets on redeploy).
    """
    session_id = payload.session_id or str(uuid.uuid4())

    history = _sessions.get(session_id, [])
    history.append({"role": "user", "content": payload.message})

    # Rolling window
    if len(history) > _MAX_HISTORY:
        history = history[-_MAX_HISTORY:]

    system = payload.system if payload.system else _SYSTEM
    reply, tokens = _route(payload.provider, history, system)

    history.append({"role": "assistant", "content": reply})
    _sessions[session_id] = history

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        provider=payload.provider,
        tokens_used=tokens,
    )


@app.get("/session/{session_id}", tags=["Sessions"])
def get_session(
    session_id: str,
    _: str = Depends(require_key),
):
    """
    Retrieve the full conversation history for a session.
    Returns a list of {role, content} message objects.
    """
    history = _sessions.get(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {
        "session_id":     session_id,
        "message_count":  len(history),
        "messages":       history,
    }


@app.delete("/session/{session_id}", tags=["Sessions"])
def delete_session(
    session_id: str,
    _: str = Depends(require_key),
):
    """
    Clear a conversation session from memory.
    Useful to reset context without starting a new session ID.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del _sessions[session_id]
    return {"deleted": True, "session_id": session_id}
