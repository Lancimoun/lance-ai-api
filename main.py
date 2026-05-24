"""
Axiom AI  v4.0
──────────────
Providers  : claude  → claude-haiku-4-5 | claude-sonnet-4-6 | claude-opus-4-7
             openai  → gpt-4o-mini | gpt-4o
             gemini  → gemini-2.0-flash | gemini-2.5-pro
             groq    → llama-3.3-70b-versatile | mixtral-8x7b-32768

Endpoints  : GET  /              HTML landing page
             GET  /ping          ultra-light liveness probe (no auth)
             GET  /health        browser → /status redirect · API → JSON
             GET  /status        beautiful system status dashboard (HTML)
             GET  /models        browser → beautiful models page · API → JSON
             GET  /usage         cumulative usage stats (JSON, auth required)
             GET  /openapi.json  browser → /docs redirect · API → raw spec
             GET  /docs          interactive API docs (branded Swagger UI)
             POST /ask           single-turn Q&A  (any provider + model)
             POST /chat          multi-turn chat   (any provider + model)
             POST /stream        streaming Q&A via Server-Sent Events (SSE)
             GET  /session/{id}  view conversation history
             DELETE /session/{id} clear a conversation

Auth       : X-API-Key header  (set SERVICE_API_KEY env var; empty = open in dev)
Rate limit : 20 requests / minute per IP
"""

import os
import uuid
import pathlib
import time
import json
from datetime import datetime, timezone
from dotenv import load_dotenv

_BASE = pathlib.Path(__file__).parent
load_dotenv()

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, RedirectResponse
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import anthropic
import openai

# ── Model registry ─────────────────────────────────────────────────────────────
_MODELS: dict = {
    "claude": {
        "default": "claude-haiku-4-5",
        "provider_name": "Anthropic",
        "models": {
            "claude-haiku-4-5":  {"name": "Haiku 4.5",  "tier": "fast",     "ctx": 200_000},
            "claude-sonnet-4-6": {"name": "Sonnet 4.6", "tier": "balanced", "ctx": 200_000},
            "claude-opus-4-7":   {"name": "Opus 4.7",   "tier": "premium",  "ctx": 200_000},
        },
    },
    "openai": {
        "default": "gpt-4o-mini",
        "provider_name": "OpenAI",
        "models": {
            "gpt-4o-mini": {"name": "GPT-4o Mini", "tier": "fast",    "ctx": 128_000},
            "gpt-4o":      {"name": "GPT-4o",      "tier": "premium", "ctx": 128_000},
        },
    },
    "gemini": {
        "default": "gemini-2.0-flash",
        "provider_name": "Google",
        "models": {
            "gemini-2.0-flash": {"name": "Gemini 2.0 Flash", "tier": "fast",    "ctx": 1_000_000},
            "gemini-2.5-pro":   {"name": "Gemini 2.5 Pro",   "tier": "premium", "ctx": 1_000_000},
        },
    },
    "groq": {
        "default": "llama-3.3-70b-versatile",
        "provider_name": "Groq",
        "models": {
            "llama-3.3-70b-versatile": {"name": "Llama 3.3 70B", "tier": "fast",     "ctx": 128_000},
            "mixtral-8x7b-32768":      {"name": "Mixtral 8x7B",  "tier": "balanced", "ctx": 32_000},
        },
    },
}

def _resolve_model(provider: str, model: str) -> str:
    if not model:
        return _MODELS[provider]["default"]
    return model

# ── Clients ─────────────────────────────────────────────────────────────────────
claude_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
async_claude  = anthropic.AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
async_openai  = openai.AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# ── Gemini (optional) ───────────────────────────────────────────────────────────
_gemini_available = False
_GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
if _GEMINI_KEY:
    try:
        import google.generativeai as _genai
        _genai.configure(api_key=_GEMINI_KEY)
        _gemini_available = True
        print("[GEMINI] ✅ Client ready")
    except ImportError:
        print("[GEMINI] ⚠️  google-generativeai not installed")
    except Exception as _e:
        print(f"[GEMINI] ❌ Init error: {_e}")

# ── Groq (optional) ─────────────────────────────────────────────────────────────
_groq_available = False
_groq_client    = None
_async_groq     = None
_GROQ_KEY = os.getenv("GROQ_API_KEY", "")
if _GROQ_KEY:
    try:
        from groq import Groq as _GroqSync, AsyncGroq as _GroqAsync
        _groq_client    = _GroqSync(api_key=_GROQ_KEY)
        _async_groq     = _GroqAsync(api_key=_GROQ_KEY)
        _groq_available = True
        print("[GROQ] ✅ Client ready")
    except ImportError:
        print("[GROQ] ⚠️  groq not installed")
    except Exception as _e:
        print(f"[GROQ] ❌ Init error: {_e}")

# ── Auth ─────────────────────────────────────────────────────────────────────────
_SERVICE_KEY = os.getenv("SERVICE_API_KEY", "")
_key_header  = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_key(key: str = Depends(_key_header)):
    if _SERVICE_KEY and key != _SERVICE_KEY:
        raise HTTPException(status_code=403, detail="Invalid or missing API key.")
    return key

# ── Rate limiting ────────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, default_limits=["20/minute"])

# ── Uptime + Usage tracking ──────────────────────────────────────────────────────
_START_TIME: float = time.time()
_usage: dict = {
    "total_requests": 0,
    "total_tokens":   0,
    "by_provider":    {"claude": 0, "openai": 0, "gemini": 0, "groq": 0},
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

# ── App ──────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Axiom AI",
    description=(
        "Production-grade AI infrastructure — Claude, GPT-4o, Gemini, and Groq behind one unified API. "
        "9 models. 4 providers. One endpoint. Built to ship."
    ),
    version="4.0",
    docs_url=None,
    redoc_url=None,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ── In-memory chat sessions ───────────────────────────────────────────────────────
_sessions: dict[str, list] = {}
_MAX_HISTORY = 20

# ── Schemas ───────────────────────────────────────────────────────────────────────
_ALL_PROVIDERS = list(_MODELS.keys())

class AskRequest(BaseModel):
    question: str
    context:  str = ""
    provider: str = "claude"
    model:    str = ""
    system:   str = ""

    @validator("question")
    def validate_question(cls, v):
        v = v.strip()
        if not v:          raise ValueError("question cannot be empty")
        if len(v) > 4000:  raise ValueError("question too long (max 4000 chars)")
        return v

    @validator("provider")
    def validate_provider(cls, v):
        if v not in _MODELS:
            raise ValueError(f"provider must be one of: {_ALL_PROVIDERS}")
        return v

    @validator("model")
    def validate_model(cls, v, values):
        if not v:
            return v
        provider = values.get("provider", "claude")
        if provider in _MODELS:
            valid = list(_MODELS[provider]["models"].keys())
            if v not in valid:
                raise ValueError(f"invalid model for '{provider}'. Valid: {valid}")
        return v

    @validator("system")
    def validate_system(cls, v):
        if len(v) > 2000: raise ValueError("system prompt too long (max 2000 chars)")
        return v.strip()


class AskResponse(BaseModel):
    answer:      str
    provider:    str
    model:       str
    tokens_used: int


class ChatRequest(BaseModel):
    message:    str
    session_id: str = ""
    provider:   str = "claude"
    model:      str = ""
    system:     str = ""

    @validator("message")
    def validate_message(cls, v):
        v = v.strip()
        if not v:          raise ValueError("message cannot be empty")
        if len(v) > 4000:  raise ValueError("message too long (max 4000 chars)")
        return v

    @validator("provider")
    def validate_provider(cls, v):
        if v not in _MODELS:
            raise ValueError(f"provider must be one of: {_ALL_PROVIDERS}")
        return v

    @validator("model")
    def validate_model(cls, v, values):
        if not v:
            return v
        provider = values.get("provider", "claude")
        if provider in _MODELS:
            valid = list(_MODELS[provider]["models"].keys())
            if v not in valid:
                raise ValueError(f"invalid model for '{provider}'. Valid: {valid}")
        return v

    @validator("system")
    def validate_system(cls, v):
        if len(v) > 2000: raise ValueError("system prompt too long (max 2000 chars)")
        return v.strip()


class ChatResponse(BaseModel):
    reply:       str
    session_id:  str
    provider:    str
    model:       str
    tokens_used: int


# ── AI helpers ────────────────────────────────────────────────────────────────────
_SYSTEM = "You are a helpful, precise AI assistant. Answer clearly and concisely."


def _ask_claude(messages: list, model_id: str, system: str) -> tuple[str, int]:
    try:
        r = claude_client.messages.create(
            model=model_id, max_tokens=800, system=system, messages=messages,
        )
        return r.content[0].text.strip(), r.usage.input_tokens + r.usage.output_tokens
    except anthropic.AuthenticationError:
        raise HTTPException(status_code=502, detail="Claude API key is invalid or missing.")
    except anthropic.RateLimitError:
        raise HTTPException(status_code=429, detail="Claude rate limit reached. Try again shortly.")
    except anthropic.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"Claude API error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error calling Claude: {str(e)}")


def _ask_openai(messages: list, model_id: str, system: str) -> tuple[str, int]:
    try:
        full = [{"role": "system", "content": system}] + messages
        r = openai_client.chat.completions.create(model=model_id, max_tokens=800, messages=full)
        return r.choices[0].message.content.strip(), r.usage.prompt_tokens + r.usage.completion_tokens
    except openai.AuthenticationError:
        raise HTTPException(status_code=502, detail="OpenAI API key is invalid or missing.")
    except openai.RateLimitError:
        raise HTTPException(status_code=429, detail="OpenAI rate limit reached. Try again shortly.")
    except openai.APIStatusError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {e.message}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error calling OpenAI: {str(e)}")


def _ask_gemini(messages: list, model_id: str, system: str) -> tuple[str, int]:
    if not _gemini_available:
        raise HTTPException(status_code=503, detail="Gemini not configured. Add GEMINI_API_KEY to environment.")
    try:
        import google.generativeai as genai  # already configured at startup
        gm = genai.GenerativeModel(
            model_id,
            system_instruction=system,
            generation_config={"max_output_tokens": 800},
        )
        if len(messages) <= 1:
            response = gm.generate_content(messages[-1]["content"])
        else:
            history = [
                {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
                for m in messages[:-1]
            ]
            response = gm.start_chat(history=history).send_message(messages[-1]["content"])
        text = response.text.strip()
        try:    tokens = response.usage_metadata.total_token_count
        except Exception: tokens = len(text) // 4
        return text, tokens
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {str(e)}")


def _ask_groq(messages: list, model_id: str, system: str) -> tuple[str, int]:
    if not _groq_available or _groq_client is None:
        raise HTTPException(status_code=503, detail="Groq not configured. Add GROQ_API_KEY to environment.")
    try:
        full = [{"role": "system", "content": system}] + messages
        r = _groq_client.chat.completions.create(model=model_id, max_tokens=800, messages=full)
        return r.choices[0].message.content.strip(), r.usage.prompt_tokens + r.usage.completion_tokens
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Groq API error: {str(e)}")


def _route(provider: str, messages: list, system: str, model: str = "") -> tuple[str, str, int]:
    """Returns (text, actual_model_id, tokens)."""
    mid = _resolve_model(provider, model)
    if provider == "openai":
        text, tokens = _ask_openai(messages, mid, system)
    elif provider == "gemini":
        text, tokens = _ask_gemini(messages, mid, system)
    elif provider == "groq":
        text, tokens = _ask_groq(messages, mid, system)
    else:
        text, tokens = _ask_claude(messages, mid, system)
    return text, mid, tokens


# ── Routes ─────────────────────────────────────────────────────────────────────────

@app.get("/ping", tags=["Health"])
def ping():
    """Ultra-light liveness probe. No auth required. Use for uptime monitors."""
    return {"pong": True, "uptime_seconds": int(time.time() - _START_TIME)}


@app.get("/health", tags=["Health"])
def health(request: Request):
    """Full health check — browser → beautiful status page · API → JSON."""
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url="/status", status_code=302)
    return {
        "status":          "live",
        "version":         "4.0",
        "environment":     "production" if _SERVICE_KEY else "development",
        "timestamp":       datetime.now(timezone.utc).isoformat(),
        "uptime":          _uptime(),
        "uptime_seconds":  int(time.time() - _START_TIME),
        "rate_limit":      "20 requests/minute per IP",
        "providers": {
            "claude": {
                "available": bool(os.getenv("ANTHROPIC_API_KEY")),
                "model":     _MODELS["claude"]["default"],
                "models":    list(_MODELS["claude"]["models"].keys()),
            },
            "openai": {
                "available": bool(os.getenv("OPENAI_API_KEY")),
                "model":     _MODELS["openai"]["default"],
                "models":    list(_MODELS["openai"]["models"].keys()),
            },
            "gemini": {
                "available": _gemini_available,
                "model":     _MODELS["gemini"]["default"],
                "models":    list(_MODELS["gemini"]["models"].keys()),
            },
            "groq": {
                "available": _groq_available,
                "model":     _MODELS["groq"]["default"],
                "models":    list(_MODELS["groq"]["models"].keys()),
            },
        },
        "sessions_active": len(_sessions),
        "usage":           _usage,
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


# ── Status page ─────────────────────────────────────────────────────────────────
@app.get("/status", tags=["Health"], response_class=HTMLResponse, include_in_schema=False)
def status_page():
    return HTMLResponse(content=(_BASE / "templates" / "status.html").read_text(encoding="utf-8"))


# ── Custom Swagger UI ──────────────────────────────────────────────────────────
@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def custom_docs():
    return HTMLResponse(content=(_BASE / "templates" / "docs.html").read_text(encoding="utf-8"))


# ── Landing page ───────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, tags=["Landing"], include_in_schema=False)
def root():
    return HTMLResponse(content=(_BASE / "templates" / "index.html").read_text(encoding="utf-8"))


# ── Models page ────────────────────────────────────────────────────────────────
@app.get("/models", tags=["Info"])
def models_endpoint(request: Request):
    """Provider + model registry — browser → beautiful page · API → JSON."""
    if "text/html" in request.headers.get("accept", ""):
        return HTMLResponse(content=(_BASE / "templates" / "models.html").read_text(encoding="utf-8"))
    return {
        provider: {
            "provider_name": info["provider_name"],
            "available": bool(
                os.getenv("ANTHROPIC_API_KEY") if provider == "claude" else
                os.getenv("OPENAI_API_KEY")    if provider == "openai" else
                _gemini_available              if provider == "gemini" else
                _groq_available
            ),
            "default": info["default"],
            "models": {
                mid: {
                    "name":           minfo["name"],
                    "tier":           minfo["tier"],
                    "context_window": minfo["ctx"],
                    "max_output":     800,
                }
                for mid, minfo in info["models"].items()
            },
        }
        for provider, info in _MODELS.items()
    }


# ── OpenAPI spec ───────────────────────────────────────────────────────────────
@app.get("/openapi.json", include_in_schema=False)
def openapi_schema(request: Request):
    if "text/html" in request.headers.get("accept", ""):
        return RedirectResponse(url="/docs", status_code=302)
    return JSONResponse(app.openapi())


# ── Ask ─────────────────────────────────────────────────────────────────────────
@app.post("/ask", response_model=AskResponse, tags=["AI"])
@limiter.limit("20/minute")
def ask(request: Request, payload: AskRequest, _: str = Depends(require_key)):
    """
    Single-turn Q&A. Send a question, get an answer.

    - Choose `provider`: **claude** · **openai** · **gemini** · **groq**
    - Optionally specify `model` (e.g. `claude-sonnet-4-6`). Defaults to fastest per provider.
    - Pass `context` to ground the answer in your data (RAG-style).
    - Pass `system` to override the default assistant persona.
    """
    system = payload.system if payload.system else _SYSTEM
    if payload.context:
        system += f"\n\nUse only this context to answer:\n{payload.context}"

    answer, mid, tokens = _route(
        payload.provider,
        [{"role": "user", "content": payload.question}],
        system,
        payload.model,
    )
    _record(payload.provider, "ask", tokens)
    return AskResponse(answer=answer, provider=payload.provider, model=mid, tokens_used=tokens)


# ── Chat ────────────────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse, tags=["AI"])
@limiter.limit("20/minute")
def chat(request: Request, payload: ChatRequest, _: str = Depends(require_key)):
    """
    Multi-turn conversation with memory.

    - Omit `session_id` to start a new session — the ID is returned in the response.
    - Pass the same `session_id` on follow-up messages to continue the conversation.
    - History is capped at the last 20 messages (in-memory, resets on redeploy).
    - Works with all 4 providers. Switch provider mid-conversation freely.
    """
    session_id = payload.session_id or str(uuid.uuid4())
    history    = _sessions.get(session_id, [])
    history.append({"role": "user", "content": payload.message})
    if len(history) > _MAX_HISTORY:
        history = history[-_MAX_HISTORY:]

    system = payload.system if payload.system else _SYSTEM
    reply, mid, tokens = _route(payload.provider, history, system, payload.model)

    history.append({"role": "assistant", "content": reply})
    _sessions[session_id] = history

    _record(payload.provider, "chat", tokens)
    return ChatResponse(
        reply=reply, session_id=session_id,
        provider=payload.provider, model=mid, tokens_used=tokens,
    )


# ── Stream ──────────────────────────────────────────────────────────────────────
@app.post("/stream", tags=["AI"])
@limiter.limit("20/minute")
async def stream_ask(request: Request, payload: AskRequest, _: str = Depends(require_key)):
    """
    Streaming single-turn Q&A via **Server-Sent Events (SSE)**.

    Supports all 4 providers. Returns tokens in real-time.

    **Event stream format:**
    ```
    data: {"token": "Hello"}
    data: {"token": " world"}
    data: {"done": true, "tokens_used": 42, "model": "claude-haiku-4-5"}
    ```
    """
    system = payload.system if payload.system else _SYSTEM
    if payload.context:
        system += f"\n\nUse only this context to answer:\n{payload.context}"

    model_id = _resolve_model(payload.provider, payload.model)
    _record(payload.provider, "stream", 0)

    async def generate_claude():
        try:
            async with async_claude.messages.stream(
                model=model_id, max_tokens=800, system=system,
                messages=[{"role": "user", "content": payload.question}],
            ) as stream:
                async for text in stream.text_stream:
                    yield f"data: {json.dumps({'token': text})}\n\n"
                final = await stream.get_final_message()
                tokens = final.usage.input_tokens + final.usage.output_tokens
                _usage["total_tokens"] += tokens
            yield f"data: {json.dumps({'done': True, 'tokens_used': tokens, 'model': model_id})}\n\n"
        except anthropic.AuthenticationError:
            yield f"data: {json.dumps({'error': 'Claude API key invalid or missing.'})}\n\n"
        except anthropic.RateLimitError:
            yield f"data: {json.dumps({'error': 'Claude rate limit reached.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    async def generate_openai():
        try:
            stream = await async_openai.chat.completions.create(
                model=model_id, max_tokens=800, stream=True,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": payload.question}],
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield f"data: {json.dumps({'token': delta})}\n\n"
            yield f"data: {json.dumps({'done': True, 'model': model_id})}\n\n"
        except openai.AuthenticationError:
            yield f"data: {json.dumps({'error': 'OpenAI API key invalid or missing.'})}\n\n"
        except openai.RateLimitError:
            yield f"data: {json.dumps({'error': 'OpenAI rate limit reached.'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    async def generate_gemini():
        if not _gemini_available:
            yield f"data: {json.dumps({'error': 'Gemini not configured. Add GEMINI_API_KEY.'})}\n\n"
            return
        try:
            import google.generativeai as genai
            gm = genai.GenerativeModel(
                model_id, system_instruction=system,
                generation_config={"max_output_tokens": 800},
            )
            response = await gm.generate_content_async(payload.question, stream=True)
            tokens = 0
            async for chunk in response:
                if chunk.text:
                    yield f"data: {json.dumps({'token': chunk.text})}\n\n"
            try:    tokens = response.usage_metadata.total_token_count
            except Exception: pass
            _usage["total_tokens"] += tokens
            yield f"data: {json.dumps({'done': True, 'tokens_used': tokens, 'model': model_id})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': f'Gemini error: {str(e)}'})}\n\n"

    async def generate_groq():
        if not _groq_available or _async_groq is None:
            yield f"data: {json.dumps({'error': 'Groq not configured. Add GROQ_API_KEY.'})}\n\n"
            return
        try:
            stream = await _async_groq.chat.completions.create(
                model=model_id, max_tokens=800, stream=True,
                messages=[{"role": "system", "content": system},
                          {"role": "user",   "content": payload.question}],
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield f"data: {json.dumps({'token': delta})}\n\n"
            yield f"data: {json.dumps({'done': True, 'model': model_id})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': f'Groq error: {str(e)}'})}\n\n"

    generators = {
        "claude": generate_claude,
        "openai": generate_openai,
        "gemini": generate_gemini,
        "groq":   generate_groq,
    }
    gen = generators.get(payload.provider, generate_claude)()
    return StreamingResponse(
        gen, media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Sessions ────────────────────────────────────────────────────────────────────
@app.get("/session/{session_id}", tags=["Sessions"])
def get_session(session_id: str, _: str = Depends(require_key)):
    """Retrieve the full conversation history for a session."""
    history = _sessions.get(session_id)
    if history is None:
        raise HTTPException(status_code=404, detail="Session not found.")
    return {"session_id": session_id, "message_count": len(history), "messages": history}


@app.delete("/session/{session_id}", tags=["Sessions"])
def delete_session(session_id: str, _: str = Depends(require_key)):
    """Clear a conversation session from memory."""
    if session_id not in _sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    del _sessions[session_id]
    return {"deleted": True, "session_id": session_id}
