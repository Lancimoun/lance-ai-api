"""
Axiom AI  v3.0
──────────────
Endpoints : GET  /              HTML landing page
            GET  /ping          ultra-light liveness probe (no auth)
            GET  /health        health check + uptime + usage stats (JSON)
            GET  /models        available providers + metadata
            GET  /usage         cumulative usage stats
            POST /ask           single-turn Q&A  (Claude or OpenAI)
            POST /chat          multi-turn chat   (Claude or OpenAI)
            POST /stream        streaming Q&A via Server-Sent Events (SSE)
            GET  /session/{id}  view conversation history
            DELETE /session/{id} clear a conversation
            GET  /docs          interactive API docs (branded Swagger UI)

Auth       : X-API-Key header  (set SERVICE_API_KEY env var; empty = open in dev)
Rate limit : 20 requests / minute per IP
Providers  : claude (claude-haiku-4-5)  |  openai (gpt-4o-mini)
Streaming  : /stream returns SSE  →  data: {"token":"..."}  …  data: {"done":true}
"""

import os
import uuid
import pathlib
import time
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

_BASE = pathlib.Path(__file__).parent

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse, RedirectResponse
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
async_claude  = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
async_openai  = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ── Auth ───────────────────────────────────────────────────────────────────────
_SERVICE_KEY = os.getenv("SERVICE_API_KEY", "")
_key_header  = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_key(key: str = Depends(_key_header)):
    if _SERVICE_KEY and key != _SERVICE_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")
    return key

# ── Rate limiting ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["20/minute"])

# ── Uptime + Usage tracking ────────────────────────────────────────────────────
_START_TIME: float = time.time()
_usage: dict = {
    "total_requests": 0,
    "total_tokens":   0,
    "by_provider":    {"claude": 0, "openai": 0},
    "by_endpoint":    {"ask": 0, "chat": 0, "stream": 0},
}

def _uptime() -> str:
    up = int(time.time() - _START_TIME)
    h, r = divmod(up, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m}m {s}s"

def _record(provider: str, endpoint: str, tokens: int = 0) -> None:
    _usage["total_requests"] += 1
    _usage["total_tokens"]   += tokens
    if provider in _usage["by_provider"]:
        _usage["by_provider"][provider] += 1
    if endpoint in _usage["by_endpoint"]:
        _usage["by_endpoint"][endpoint] += 1

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Axiom AI",
    description=(
        "Production-grade AI infrastructure supporting Claude (Anthropic) and GPT (OpenAI). "
        "One unified API. Two world-class models. Built to ship."
    ),
    version="3.0",
    docs_url=None,
    redoc_url=None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    provider: str = "claude"
    system:   str = ""

    @validator("question")
    def validate_question(cls, v):
        v = v.strip()
        if not v:           raise ValueError("question cannot be empty")
        if len(v) > 4000:   raise ValueError("question too long (max 4000 chars)")
        return v

    @validator("provider")
    def validate_provider(cls, v):
        if v not in ("claude", "openai"):
            raise ValueError("provider must be 'claude' or 'openai'")
        return v

    @validator("system")
    def validate_system(cls, v):
        if len(v) > 2000:   raise ValueError("system prompt too long (max 2000 chars)")
        return v.strip()


class AskResponse(BaseModel):
    answer:      str
    provider:    str
    tokens_used: int


class ChatRequest(BaseModel):
    message:    str
    session_id: str = ""
    provider:   str = "claude"
    system:     str = ""

    @validator("message")
    def validate_message(cls, v):
        v = v.strip()
        if not v:           raise ValueError("message cannot be empty")
        if len(v) > 4000:   raise ValueError("message too long (max 4000 chars)")
        return v

    @validator("provider")
    def validate_provider(cls, v):
        if v not in ("claude", "openai"):
            raise ValueError("provider must be 'claude' or 'openai'")
        return v

    @validator("system")
    def validate_system(cls, v):
        if len(v) > 2000:   raise ValueError("system prompt too long (max 2000 chars)")
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

@app.get("/ping", tags=["Health"])
def ping():
    """Ultra-light liveness probe. No auth required. Use for uptime monitors."""
    return {"pong": True, "uptime_seconds": int(time.time() - _START_TIME)}


@app.get("/health", tags=["Health"])
def health(request: Request):
    """Full health check — browser → beautiful status page · API call → raw JSON."""
    # Browsers send text/html in Accept; redirect them to the pretty status dashboard
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url="/status", status_code=302)
    return {
        "status":          "live",
        "version":         "3.0",
        "environment":     "production" if _SERVICE_KEY else "development",
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "uptime":          _uptime(),
        "uptime_seconds":  int(time.time() - _START_TIME),
        "rate_limit":      "20 requests/minute per IP",
        "providers": {
            "claude": {
                "available": bool(os.getenv("ANTHROPIC_API_KEY")),
                "model":     "claude-haiku-4-5",
            },
            "openai": {
                "available": bool(os.getenv("OPENAI_API_KEY")),
                "model":     "gpt-4o-mini",
            },
        },
        "sessions_active": len(_sessions),
        "usage":           _usage,
        "endpoints": {
            "GET  /":               "landing page",
            "GET  /ping":           "liveness probe (no auth)",
            "GET  /health":         "health check (JSON)",
            "GET  /models":         "available providers + metadata",
            "GET  /usage":          "cumulative usage stats",
            "POST /ask":            "single-turn Q&A",
            "POST /chat":           "multi-turn conversation",
            "POST /stream":         "streaming Q&A via SSE",
            "GET  /session/{id}":   "view conversation history",
            "DELETE /session/{id}": "clear a conversation",
            "GET  /docs":           "interactive API docs",
        },
    }


@app.get("/usage", tags=["Info"])
def usage_endpoint(_: str = Depends(require_key)):
    """Cumulative request and token usage since last deploy."""
    return {
        **_usage,
        "sessions_active": len(_sessions),
        "uptime_seconds":  int(time.time() - _START_TIME),
        "uptime":          _uptime(),
    }


# ── Status page ────────────────────────────────────────────────────────────────
@app.get("/status", tags=["Health"], response_class=HTMLResponse, include_in_schema=False)
def status_page():
    """Beautiful system status dashboard."""
    return HTMLResponse(content=(_BASE / "templates" / "status.html").read_text(encoding="utf-8"))


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
    """Available AI providers, their models, capabilities, and context windows."""
    return {
        "claude": {
            "model":          "claude-haiku-4-5",
            "provider":       "Anthropic",
            "available":      bool(os.getenv("ANTHROPIC_API_KEY")),
            "context_window": 200_000,
            "max_output":     800,
            "strengths":      ["fast", "cost-efficient", "RAG-ready", "200K context"],
        },
        "openai": {
            "model":          "gpt-4o-mini",
            "provider":       "OpenAI",
            "available":      bool(os.getenv("OPENAI_API_KEY")),
            "context_window": 128_000,
            "max_output":     800,
            "strengths":      ["code generation", "reasoning", "tool-use", "JSON mode"],
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

    - Optionally pass `context` to ground the answer in specific information (RAG-style).
    - Optionally pass `system` to override the default system prompt.
    - Choose `provider`: **claude** (default) or **openai**.
    """
    system = payload.system if payload.system else _SYSTEM
    if payload.context:
        system += f"\n\nUse only this context to answer:\n{payload.context}"

    answer, tokens = _route(
        payload.provider,
        [{"role": "user", "content": payload.question}],
        system,
    )
    _record(payload.provider, "ask", tokens)
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

    - Omit `session_id` to start a new session — the ID is returned in the response.
    - Pass the same `session_id` on follow-up messages to continue the conversation.
    - History is capped at the last 20 messages (in-memory, resets on redeploy).
    - Optionally pass `system` to set a custom persona for the whole session.
    """
    session_id = payload.session_id or str(uuid.uuid4())
    history    = _sessions.get(session_id, [])
    history.append({"role": "user", "content": payload.message})

    if len(history) > _MAX_HISTORY:
        history = history[-_MAX_HISTORY:]

    system = payload.system if payload.system else _SYSTEM
    reply, tokens = _route(payload.provider, history, system)

    history.append({"role": "assistant", "content": reply})
    _sessions[session_id] = history

    _record(payload.provider, "chat", tokens)
    return ChatResponse(
        reply=reply,
        session_id=session_id,
        provider=payload.provider,
        tokens_used=tokens,
    )


@app.post("/stream", tags=["AI"])
@limiter.limit("20/minute")
async def stream_ask(
    request: Request,
    payload: AskRequest,
    _: str = Depends(require_key),
):
    """
    Streaming single-turn Q&A via **Server-Sent Events (SSE)**.

    Returns tokens in real-time as they are generated — no waiting for the full response.

    **Event stream format:**
    ```
    data: {"token": "Hello"}
    data: {"token": " world"}
    data: {"done": true, "tokens_used": 42}
    ```

    **JavaScript fetch example:**
    ```js
    const res = await fetch('/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-API-Key': 'YOUR_KEY' },
      body: JSON.stringify({ question: 'What is RAG?' })
    });
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const lines = decoder.decode(value).split('\\n');
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const d = JSON.parse(line.slice(6));
          if (d.token) process.stdout.write(d.token);
        }
      }
    }
    ```
    """
    system = payload.system if payload.system else _SYSTEM
    if payload.context:
        system += f"\n\nUse only this context to answer:\n{payload.context}"

    # Count the request now; tokens added at end of stream
    _record(payload.provider, "stream", 0)

    async def generate_claude():
        try:
            async with async_claude.messages.stream(
                model="claude-haiku-4-5",
                max_tokens=800,
                system=system,
                messages=[{"role": "user", "content": payload.question}],
            ) as stream:
                async for text in stream.text_stream:
                    yield f"data: {json.dumps({'token': text})}\n\n"
                final  = await stream.get_final_message()
                tokens = final.usage.input_tokens + final.usage.output_tokens
                _usage["total_tokens"] += tokens
            yield f"data: {json.dumps({'done': True, 'tokens_used': tokens})}\n\n"
        except anthropic.AuthenticationError:
            yield f"data: {json.dumps({'error': 'Claude API key is invalid or missing.'})}\n\n"
        except anthropic.RateLimitError:
            yield f"data: {json.dumps({'error': 'Claude rate limit reached. Try again shortly.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    async def generate_openai():
        try:
            stream = await async_openai.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=800,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": payload.question},
                ],
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield f"data: {json.dumps({'token': delta})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except openai.AuthenticationError:
            yield f"data: {json.dumps({'error': 'OpenAI API key is invalid or missing.'})}\n\n"
        except openai.RateLimitError:
            yield f"data: {json.dumps({'error': 'OpenAI rate limit reached. Try again shortly.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    gen = generate_claude() if payload.provider == "claude" else generate_openai()
    return StreamingResponse(
        gen,
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
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
        "session_id":    session_id,
        "message_count": len(history),
        "messages":      history,
    }


@app.delete("/session/{session_id}", tags=["Sessions"])
def delete_session(
    session_id: str,
    _: str = Depends(require_key),
):
    """
    Clear a conversation session from memory.
    Useful to reset context without changing the session ID.
    """
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del _sessions[session_id]
    return {"deleted": True, "session_id": session_id}
