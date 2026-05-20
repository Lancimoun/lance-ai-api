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
from datetime import datetime, timezone
from dotenv import load_dotenv

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


# ── Landing page ───────────────────────────────────────────────────────────────
_LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Lance AI API — Production-Ready AI Infrastructure</title>
  <meta name="description" content="Dual-provider AI API powering Claude and GPT-4o. Built for developers who ship."/>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg:       #030712;
      --bg2:      #060d1f;
      --surface:  rgba(255,255,255,0.04);
      --surface2: rgba(255,255,255,0.07);
      --border:   rgba(255,255,255,0.08);
      --border2:  rgba(255,255,255,0.15);
      --primary:  #6366f1;
      --primary-l:#818cf8;
      --cyan:     #06b6d4;
      --green:    #10b981;
      --text:     #f1f5f9;
      --text2:    #94a3b8;
      --muted:    #475569;
    }

    html { scroll-behavior: smooth; }

    body {
      font-family: 'Inter', -apple-system, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* BG */
    .bg-grid {
      position: fixed; inset: 0; z-index: 0; pointer-events: none;
      background-image: radial-gradient(circle, rgba(255,255,255,0.045) 1px, transparent 1px);
      background-size: 32px 32px;
    }
    .bg-orb {
      position: fixed; border-radius: 50%;
      pointer-events: none; z-index: 0; filter: blur(140px);
    }
    .orb1 { width: 700px; height: 700px; background: var(--primary);  top: -280px; left: -200px; opacity: 0.13; }
    .orb2 { width: 500px; height: 500px; background: var(--cyan);     top: 80px;   right: -180px; opacity: 0.07; }
    .orb3 { width: 600px; height: 600px; background: #8b5cf6;         bottom: -200px; left: 25%; opacity: 0.09; }

    .container { position: relative; z-index: 1; max-width: 1160px; margin: 0 auto; padding: 0 24px; }

    /* ── NAV ── */
    nav {
      position: sticky; top: 0; z-index: 100;
      backdrop-filter: blur(24px) saturate(180%);
      background: rgba(3,7,18,0.75);
      border-bottom: 1px solid var(--border);
    }
    .nav-inner {
      display: flex; align-items: center; justify-content: space-between;
      height: 64px; max-width: 1160px; margin: 0 auto; padding: 0 24px;
    }
    .logo { display: flex; align-items: center; gap: 10px; text-decoration: none; color: var(--text); }
    .logo-icon {
      width: 32px; height: 32px; border-radius: 8px;
      background: linear-gradient(135deg, var(--primary), var(--cyan));
      display: flex; align-items: center; justify-content: center;
      flex-shrink: 0;
    }
    .logo-wordmark { font-size: 15px; font-weight: 800; letter-spacing: -0.02em; }
    .logo-tag {
      font-size: 9px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase;
      background: rgba(99,102,241,0.18); border: 1px solid rgba(99,102,241,0.3);
      color: var(--primary-l); padding: 2px 7px; border-radius: 5px;
    }
    .nav-links { display: flex; align-items: center; gap: 2px; }
    .nav-links a {
      color: var(--text2); text-decoration: none; font-size: 13px; font-weight: 500;
      padding: 6px 13px; border-radius: 8px; transition: all .18s;
    }
    .nav-links a:hover { background: var(--surface2); color: var(--text); }
    .nav-right { display: flex; align-items: center; gap: 12px; }
    .status-pill {
      display: flex; align-items: center; gap: 6px; padding: 5px 13px;
      background: rgba(16,185,129,0.1); border: 1px solid rgba(16,185,129,0.22);
      border-radius: 999px; font-size: 11px; font-weight: 600; color: var(--green);
    }
    .pulse-dot {
      width: 6px; height: 6px; border-radius: 50%; background: var(--green);
      animation: blink 2.2s ease-in-out infinite;
    }
    @keyframes blink {
      0%,100% { box-shadow: 0 0 0 0 rgba(16,185,129,0.6); }
      50%      { box-shadow: 0 0 0 5px rgba(16,185,129,0); }
    }
    .btn-nav {
      display: inline-flex; align-items: center; gap: 6px;
      padding: 7px 18px; border-radius: 9px; font-size: 13px; font-weight: 600;
      text-decoration: none; color: #fff;
      background: var(--primary); transition: all .18s;
    }
    .btn-nav:hover { background: var(--primary-l); transform: translateY(-1px); }

    /* ── HERO ── */
    .hero {
      display: grid; grid-template-columns: 1fr 1fr;
      gap: 56px; align-items: center; padding: 100px 0 88px;
    }
    .hero-eyebrow {
      display: inline-flex; align-items: center; gap: 8px;
      background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.22);
      border-radius: 999px; padding: 5px 14px;
      font-size: 11px; font-weight: 700; letter-spacing: 0.05em; color: var(--primary-l);
      margin-bottom: 28px;
    }
    .eyebrow-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--primary-l); }
    h1 {
      font-size: clamp(38px, 4.8vw, 60px);
      font-weight: 900; line-height: 1.07; letter-spacing: -0.04em;
      margin-bottom: 20px;
    }
    .grad-text {
      background: linear-gradient(135deg, var(--primary-l) 0%, var(--cyan) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    }
    .hero-sub {
      font-size: 16px; color: var(--text2); line-height: 1.72;
      max-width: 440px; margin-bottom: 36px;
    }
    .hero-cta { display: flex; gap: 10px; flex-wrap: wrap; margin-bottom: 36px; }
    .btn-primary {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 13px 26px; border-radius: 11px; font-size: 14px; font-weight: 600;
      text-decoration: none; color: #fff;
      background: linear-gradient(135deg, var(--primary), #7c3aed);
      box-shadow: 0 4px 28px rgba(99,102,241,0.35); transition: all .2s;
    }
    .btn-primary:hover { transform: translateY(-2px); box-shadow: 0 8px 36px rgba(99,102,241,0.45); }
    .btn-outline {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 13px 26px; border-radius: 11px; font-size: 14px; font-weight: 600;
      text-decoration: none; color: var(--text2);
      background: var(--surface); border: 1px solid var(--border2); transition: all .2s;
    }
    .btn-outline:hover { color: var(--text); background: var(--surface2); transform: translateY(-2px); }
    .hero-trust { display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
    .trust-item {
      display: flex; align-items: center; gap: 6px;
      font-size: 12px; color: var(--muted); font-weight: 500;
    }
    .trust-check {
      width: 14px; height: 14px; border-radius: 50%;
      background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.3);
      display: flex; align-items: center; justify-content: center;
      font-size: 8px; color: var(--green);
    }

    /* ── DEMO CARD ── */
    .demo-wrap { position: relative; display: flex; justify-content: center; align-items: center; }
    .demo-glow {
      position: absolute; inset: -24px;
      background: radial-gradient(ellipse at center, rgba(99,102,241,0.18) 0%, transparent 68%);
    }
    .demo-card {
      position: relative; background: var(--bg2);
      border: 1px solid var(--border2); border-radius: 18px;
      overflow: hidden; width: 100%; max-width: 460px;
      box-shadow: 0 32px 80px rgba(0,0,0,0.65), 0 0 0 1px rgba(255,255,255,0.04);
    }
    .demo-card::after {
      content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
      background: linear-gradient(90deg, transparent 0%, rgba(99,102,241,0.7) 35%, rgba(6,182,212,0.7) 65%, transparent 100%);
    }
    .demo-topbar {
      display: flex; align-items: center; gap: 8px;
      padding: 13px 16px; border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,0.025);
    }
    .win-dots { display: flex; gap: 5px; }
    .wd { width: 10px; height: 10px; border-radius: 50%; }
    .wd-r { background: #ff5f57; } .wd-y { background: #ffbd2e; } .wd-g { background: #28c840; }
    .demo-url {
      flex: 1; text-align: center; font-size: 11px; color: var(--muted);
      font-family: 'SF Mono', monospace; letter-spacing: 0.01em;
    }
    .demo-body { padding: 20px 22px; font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12.5px; line-height: 1.75; }
    .demo-sep {
      font-size: 10px; letter-spacing: 0.1em; text-transform: uppercase;
      color: var(--muted); font-weight: 600; margin: 14px 0 10px;
      display: flex; align-items: center; gap: 8px;
    }
    .demo-sep::after { content: ''; flex: 1; height: 1px; background: var(--border); }
    .req-block {
      background: rgba(99,102,241,0.07); border: 1px solid rgba(99,102,241,0.14);
      border-radius: 9px; padding: 14px 16px;
    }
    .res-block {
      background: rgba(16,185,129,0.05); border: 1px solid rgba(16,185,129,0.12);
      border-radius: 9px; padding: 14px 16px;
      animation: fadeUp .5s ease 1.1s both;
    }
    .ok-row {
      display: flex; align-items: center; gap: 8px; margin-bottom: 10px;
      animation: fadeDown .4s ease .85s both;
    }
    .ok-badge {
      display: inline-flex; align-items: center; gap: 5px;
      background: rgba(16,185,129,0.14); border: 1px solid rgba(16,185,129,0.28);
      border-radius: 5px; padding: 2px 9px; font-size: 11px; font-weight: 700; color: var(--green);
    }
    .ok-ms { font-size: 11px; color: var(--muted); }
    @keyframes fadeDown { from { opacity:0; transform:translateY(-5px); } to { opacity:1; transform:none; } }
    @keyframes fadeUp   { from { opacity:0; transform:translateY(7px);  } to { opacity:1; transform:none; } }
    .dm  { color: #475569; }
    .dp  { color: #818cf8; }
    .ds  { color: #34d399; }
    .dk  { color: #06b6d4; }
    .dnum{ color: #fb923c; }

    /* ── STATS BAR ── */
    .stats-bar {
      border: 1px solid var(--border); border-radius: 18px;
      display: grid; grid-template-columns: repeat(4,1fr);
      margin-bottom: 100px; overflow: hidden;
    }
    .stat {
      padding: 30px 20px; text-align: center;
      border-right: 1px solid var(--border); position: relative;
    }
    .stat:last-child { border-right: none; }
    .stat-val { font-size: 28px; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 5px; }
    .c-indigo  { color: var(--primary-l); }
    .c-cyan    { color: var(--cyan); }
    .c-green   { color: var(--green); }
    .c-violet  { color: #a78bfa; }
    .stat-lbl  { font-size: 12px; color: var(--muted); font-weight: 500; }

    /* ── SECTION HEADER ── */
    .sec-hd { text-align: center; margin-bottom: 60px; }
    .sec-pill {
      display: inline-block; font-size: 11px; font-weight: 700; letter-spacing: 0.1em;
      text-transform: uppercase; color: var(--primary-l);
      background: rgba(99,102,241,0.1); border: 1px solid rgba(99,102,241,0.22);
      border-radius: 999px; padding: 4px 13px; margin-bottom: 16px;
    }
    .sec-hd h2 { font-size: 38px; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 10px; }
    .sec-hd p  { font-size: 15px; color: var(--text2); max-width: 490px; margin: 0 auto; line-height: 1.65; }

    /* ── ENDPOINT CARDS ── */
    .ep-grid {
      display: grid; grid-template-columns: repeat(3,1fr);
      gap: 12px; margin-bottom: 100px;
    }
    .ep-card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 15px; padding: 24px; transition: all .22s;
      position: relative; overflow: hidden;
    }
    .ep-card:hover { border-color: var(--border2); background: var(--surface2); transform: translateY(-3px); box-shadow: 0 16px 48px rgba(0,0,0,0.35); }
    .ep-card:hover .ep-arr { opacity: 1; transform: translateX(0); }
    .ep-top { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
    .ep-badges { display: flex; align-items: center; gap: 6px; }
    .ep-method {
      font-size: 9px; font-weight: 800; letter-spacing: 0.1em; text-transform: uppercase;
      padding: 3px 9px; border-radius: 5px;
    }
    .m-get  { background: rgba(16,185,129,0.12);  color: var(--green);     border: 1px solid rgba(16,185,129,0.22); }
    .m-post { background: rgba(99,102,241,0.12); color: var(--primary-l); border: 1px solid rgba(99,102,241,0.22); }
    .ep-path { font-size: 14px; font-weight: 700; font-family: 'SF Mono', monospace; }
    .ep-arr  { color: var(--muted); opacity: 0; transform: translateX(-5px); transition: all .2s; }
    .ep-icon { font-size: 26px; margin-bottom: 12px; }
    .ep-title { font-size: 14px; font-weight: 700; margin-bottom: 7px; }
    .ep-desc  { font-size: 12.5px; color: var(--text2); line-height: 1.6; }

    /* ── HOW IT WORKS ── */
    .steps-wrap { display: grid; grid-template-columns: repeat(3,1fr); gap: 0; margin-bottom: 100px; position: relative; }
    .steps-wrap::before {
      content: ''; position: absolute;
      top: 27px; left: calc(16.67% + 12px); right: calc(16.67% + 12px);
      height: 1px;
      background: linear-gradient(90deg, var(--primary) 0%, var(--cyan) 100%);
      opacity: 0.25;
    }
    .step { text-align: center; padding: 0 28px; }
    .step-num {
      width: 54px; height: 54px; border-radius: 14px; margin: 0 auto 22px;
      display: flex; align-items: center; justify-content: center;
      font-size: 18px; font-weight: 800;
    }
    .sn1 { background: rgba(99,102,241,0.12);  border: 1px solid rgba(99,102,241,0.25);  color: var(--primary-l); }
    .sn2 { background: rgba(139,92,246,0.12);  border: 1px solid rgba(139,92,246,0.25);  color: #a78bfa; }
    .sn3 { background: rgba(6,182,212,0.12);   border: 1px solid rgba(6,182,212,0.25);   color: var(--cyan); }
    .step h3 { font-size: 16px; font-weight: 700; margin-bottom: 8px; }
    .step p   { font-size: 13px; color: var(--text2); line-height: 1.65; }
    code { font-family: 'SF Mono', monospace; font-size: 11.5px; }

    /* ── PROVIDERS ── */
    .prov-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 100px; }
    .prov-card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 18px; padding: 34px; transition: all .22s;
    }
    .prov-card:hover { border-color: var(--border2); transform: translateY(-3px); box-shadow: 0 16px 48px rgba(0,0,0,0.3); }
    .prov-hd { display: flex; align-items: center; gap: 14px; margin-bottom: 18px; }
    .prov-logo {
      width: 50px; height: 50px; border-radius: 13px;
      display: flex; align-items: center; justify-content: center; font-size: 24px; flex-shrink: 0;
    }
    .pl-claude { background: rgba(255,154,76,0.12); border: 1px solid rgba(255,154,76,0.22); }
    .pl-openai { background: rgba(16,185,129,0.12); border: 1px solid rgba(16,185,129,0.22); }
    .prov-name { font-size: 18px; font-weight: 700; margin-bottom: 3px; }
    .prov-model{ font-size: 11px; color: var(--muted); font-family: monospace; font-weight: 500; }
    .prov-desc { font-size: 13px; color: var(--text2); line-height: 1.72; margin-bottom: 22px; }
    .prov-tags { display: flex; flex-wrap: wrap; gap: 6px; }
    .ptag {
      font-size: 11px; font-weight: 600; padding: 3px 10px; border-radius: 6px;
    }
    .pt-c { background: rgba(255,154,76,0.1);  color: #fb923c; border: 1px solid rgba(255,154,76,0.18); }
    .pt-o { background: rgba(16,185,129,0.1);  color: var(--green); border: 1px solid rgba(16,185,129,0.18); }

    /* ── CODE BLOCK ── */
    .code-section { margin-bottom: 100px; }
    .code-card {
      background: #060d1f; border: 1px solid var(--border2);
      border-radius: 18px; overflow: hidden;
      box-shadow: 0 28px 72px rgba(0,0,0,0.5);
    }
    .code-topbar {
      display: flex; align-items: center; justify-content: space-between;
      padding: 0 20px; border-bottom: 1px solid var(--border);
      background: rgba(255,255,255,0.02);
    }
    .code-tabs { display: flex; }
    .ctab {
      padding: 13px 16px; font-size: 12px; font-weight: 600;
      color: var(--muted); border-bottom: 2px solid transparent; transition: all .18s;
      cursor: default;
    }
    .ctab.on { color: var(--text); border-bottom-color: var(--primary); }
    .code-badge {
      font-size: 11px; font-weight: 600; color: var(--muted);
      background: var(--surface); border: 1px solid var(--border);
      padding: 5px 13px; border-radius: 7px;
    }
    .code-body { padding: 30px 32px; overflow-x: auto; }
    pre { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; font-size: 13px; line-height: 1.85; }
    .cc { color: #374151; }
    .cm { color: #10b981; }
    .ch { color: #f59e0b; }
    .cv { color: #34d399; }
    .cj { color: #6b7280; }
    .ck { color: #c084fc; }
    .cs { color: #fbbf24; }
    .cu { color: #818cf8; }

    /* ── CTA ── */
    .cta-wrap {
      background: linear-gradient(135deg, rgba(99,102,241,0.1), rgba(6,182,212,0.06));
      border: 1px solid rgba(99,102,241,0.22); border-radius: 22px;
      padding: 64px 48px; text-align: center; margin-bottom: 80px; position: relative; overflow: hidden;
    }
    .cta-wrap::before {
      content: ''; position: absolute; top: 0; left: 0; right: 0; height: 1px;
      background: linear-gradient(90deg, transparent, var(--primary), var(--cyan), transparent);
    }
    .cta-wrap h2 { font-size: 38px; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 12px; }
    .cta-wrap p  { font-size: 15px; color: var(--text2); margin-bottom: 36px; }
    .cta-btns { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }

    /* ── FOOTER ── */
    footer { border-top: 1px solid var(--border); padding: 44px 0; }
    .foot-inner {
      display: flex; justify-content: space-between; align-items: center;
      flex-wrap: wrap; gap: 20px;
    }
    .foot-left {}
    .foot-brand { display: flex; align-items: center; gap: 9px; margin-bottom: 7px; }
    .foot-brand-name { font-size: 14px; font-weight: 700; }
    .foot-copy { font-size: 12px; color: var(--muted); }
    .foot-links { display: flex; gap: 24px; flex-wrap: wrap; }
    .foot-links a {
      font-size: 13px; color: var(--muted); text-decoration: none;
      font-weight: 500; transition: color .18s;
    }
    .foot-links a:hover { color: var(--text); }

    /* ── RESPONSIVE ── */
    @media (max-width: 960px) {
      .hero { grid-template-columns: 1fr; }
      .demo-wrap { display: none; }
      .ep-grid { grid-template-columns: 1fr 1fr; }
      .prov-grid { grid-template-columns: 1fr; }
      .steps-wrap { grid-template-columns: 1fr; gap: 36px; }
      .steps-wrap::before { display: none; }
      .stats-bar { grid-template-columns: 1fr 1fr; }
      .stat:nth-child(2) { border-right: none; }
      .stat:nth-child(1),
      .stat:nth-child(2) { border-bottom: 1px solid var(--border); }
      .nav-links { display: none; }
    }
    @media (max-width: 580px) {
      .ep-grid { grid-template-columns: 1fr; }
      .cta-wrap { padding: 40px 24px; }
      h1 { font-size: 36px; }
    }
  </style>
</head>
<body>
  <div class="bg-grid"></div>
  <div class="bg-orb orb1"></div>
  <div class="bg-orb orb2"></div>
  <div class="bg-orb orb3"></div>

  <!-- ── NAV ── -->
  <nav>
    <div class="nav-inner">
      <a href="/" class="logo">
        <div class="logo-icon">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
            <path d="M13 2L3 14h8l-1 8 10-12h-8l1-8z" fill="white"/>
          </svg>
        </div>
        <span class="logo-wordmark">Lance AI</span>
        <span class="logo-tag">API</span>
      </a>
      <div class="nav-links">
        <a href="#endpoints">Endpoints</a>
        <a href="#providers">Providers</a>
        <a href="#quickstart">Code</a>
        <a href="/health">Status</a>
      </div>
      <div class="nav-right">
        <div class="status-pill"><div class="pulse-dot"></div>All Systems Online</div>
        <a href="/docs" class="btn-nav">Open Docs &#8594;</a>
      </div>
    </div>
  </nav>

  <div class="container">

    <!-- ── HERO ── -->
    <section class="hero">
      <div>
        <div class="hero-eyebrow"><div class="eyebrow-dot"></div>Production-Ready &nbsp;&#183;&nbsp; v2.0</div>
        <h1>AI Infrastructure<br/>for Modern <span class="grad-text">Builders.</span></h1>
        <p class="hero-sub">
          A dual-provider AI API backed by Claude and GPT-4o.
          One endpoint. Instant responses. No setup friction.
          Built to ship.
        </p>
        <div class="hero-cta">
          <a href="/docs" class="btn-primary">&#9889; Try in Swagger</a>
          <a href="/health" class="btn-outline">&#128314; Health Check</a>
        </div>
        <div class="hero-trust">
          <div class="trust-item"><div class="trust-check">&#10003;</div>FastAPI + Docker</div>
          <div class="trust-item"><div class="trust-check">&#10003;</div>Deployed on Railway</div>
          <div class="trust-item"><div class="trust-check">&#10003;</div>API Key Auth</div>
          <div class="trust-item"><div class="trust-check">&#10003;</div>Rate Limited</div>
        </div>
      </div>

      <div class="demo-wrap">
        <div class="demo-glow"></div>
        <div class="demo-card">
          <div class="demo-topbar">
            <div class="win-dots">
              <div class="wd wd-r"></div><div class="wd wd-y"></div><div class="wd wd-g"></div>
            </div>
            <div class="demo-url">lance-ai-api-production.up.railway.app</div>
          </div>
          <div class="demo-body">
            <div class="demo-sep">Request</div>
            <div class="req-block">
<span class="dk">POST</span> <span class="dp">/ask</span>
<span class="dm">X-API-Key: </span><span class="ds">&#8226;&#8226;&#8226;&#8226;&#8226;&#8226;&#8226;&#8226;</span>

<span class="dm">{</span>
  <span class="dp">"question"</span><span class="dm">:</span> <span class="ds">"What is RAG?"</span><span class="dm">,</span>
  <span class="dp">"provider"</span><span class="dm">:</span> <span class="ds">"claude"</span>
<span class="dm">}</span>
            </div>
            <div class="demo-sep">Response</div>
            <div class="ok-row">
              <span class="ok-badge">&#10003; 200 OK</span>
              <span class="ok-ms">318ms</span>
            </div>
            <div class="res-block">
<span class="dm">{</span>
  <span class="dp">"answer"</span><span class="dm">:</span> <span class="ds">"RAG combines retrieval
  with generation to ground
  AI answers in real data..."</span><span class="dm">,</span>
  <span class="dp">"provider"</span><span class="dm">:</span> <span class="ds">"claude"</span><span class="dm">,</span>
  <span class="dp">"tokens_used"</span><span class="dm">:</span> <span class="dnum">187</span>
<span class="dm">}</span>
            </div>
          </div>
        </div>
      </div>
    </section>

    <!-- ── STATS ── -->
    <div class="stats-bar">
      <div class="stat">
        <div class="stat-val c-indigo">2</div>
        <div class="stat-lbl">AI Providers</div>
      </div>
      <div class="stat">
        <div class="stat-val c-cyan">20 / min</div>
        <div class="stat-lbl">Rate Limit</div>
      </div>
      <div class="stat">
        <div class="stat-val c-green">20 msg</div>
        <div class="stat-lbl">Chat Memory</div>
      </div>
      <div class="stat">
        <div class="stat-val c-violet">HTTPS</div>
        <div class="stat-lbl">End-to-End Encrypted</div>
      </div>
    </div>

    <!-- ── ENDPOINTS ── -->
    <div id="endpoints">
      <div class="sec-hd">
        <div class="sec-pill">Endpoints</div>
        <h2>Everything you need to build</h2>
        <p>Five focused endpoints. No bloat. No setup friction. Swap providers per request.</p>
      </div>
      <div class="ep-grid">

        <div class="ep-card">
          <div class="ep-top">
            <div class="ep-badges"><span class="ep-method m-get">GET</span><span class="ep-path">/health</span></div>
            <span class="ep-arr">&#8594;</span>
          </div>
          <div class="ep-icon">&#128314;</div>
          <div class="ep-title">Health Check</div>
          <div class="ep-desc">Live status endpoint. Returns version and the full endpoint map. No auth required.</div>
        </div>

        <div class="ep-card">
          <div class="ep-top">
            <div class="ep-badges"><span class="ep-method m-post">POST</span><span class="ep-path">/ask</span></div>
            <span class="ep-arr">&#8594;</span>
          </div>
          <div class="ep-icon">&#129504;</div>
          <div class="ep-title">Single-Turn Q&amp;A</div>
          <div class="ep-desc">Send a question and optional context. Get a precise AI answer back in one round trip.</div>
        </div>

        <div class="ep-card">
          <div class="ep-top">
            <div class="ep-badges"><span class="ep-method m-post">POST</span><span class="ep-path">/chat</span></div>
            <span class="ep-arr">&#8594;</span>
          </div>
          <div class="ep-icon">&#128172;</div>
          <div class="ep-title">Multi-Turn Chat</div>
          <div class="ep-desc">Stateful conversations with 20-message rolling memory. Use session_id to continue across requests.</div>
        </div>

        <div class="ep-card">
          <div class="ep-top">
            <div class="ep-badges"><span class="ep-method m-get">GET</span><span class="ep-path">/models</span></div>
            <span class="ep-arr">&#8594;</span>
          </div>
          <div class="ep-icon">&#128225;</div>
          <div class="ep-title">Model Registry</div>
          <div class="ep-desc">Lists available providers, model IDs, and live availability status for each.</div>
        </div>

        <div class="ep-card">
          <div class="ep-top">
            <div class="ep-badges"><span class="ep-method m-get">GET</span><span class="ep-path">/docs</span></div>
            <span class="ep-arr">&#8594;</span>
          </div>
          <div class="ep-icon">&#128218;</div>
          <div class="ep-title">Swagger UI</div>
          <div class="ep-desc">Interactive API explorer. Test every endpoint live in your browser — no tools needed.</div>
        </div>

        <div class="ep-card" style="background:linear-gradient(135deg,rgba(99,102,241,0.08),rgba(6,182,212,0.04));border-color:rgba(99,102,241,0.22);">
          <div class="ep-top">
            <div class="ep-badges"><span class="ep-method m-post">POST</span><span class="ep-path">/ask &amp; /chat</span></div>
          </div>
          <div class="ep-icon">&#128256;</div>
          <div class="ep-title">Provider Switching</div>
          <div class="ep-desc">Pass <code style="color:var(--primary-l)">"provider":"claude"</code> or <code style="color:var(--primary-l)">"openai"</code> per request. Zero re-implementation.</div>
        </div>

      </div>
    </div>

    <!-- ── HOW IT WORKS ── -->
    <div style="margin-bottom:100px;">
      <div class="sec-hd">
        <div class="sec-pill">How It Works</div>
        <h2>Three steps. First response in &lt;60s.</h2>
        <p>From zero to live AI responses — faster than you can brew coffee.</p>
      </div>
      <div class="steps-wrap">
        <div class="step">
          <div class="step-num sn1">1</div>
          <h3>Get Your API Key</h3>
          <p>Request access. Set <code style="color:var(--primary-l)">X-API-Key</code> in your header. Done in 10 seconds.</p>
        </div>
        <div class="step">
          <div class="step-num sn2">2</div>
          <h3>Pick Your Provider</h3>
          <p>Pass <code style="color:#a78bfa">"provider":"claude"</code> or <code style="color:#a78bfa">"openai"</code>. Switch any time, per request.</p>
        </div>
        <div class="step">
          <div class="step-num sn3">3</div>
          <h3>Ship Your Product</h3>
          <p>Hit <code style="color:var(--cyan)">/ask</code> or <code style="color:var(--cyan)">/chat</code>. Get structured AI responses in milliseconds.</p>
        </div>
      </div>
    </div>

    <!-- ── PROVIDERS ── -->
    <div id="providers">
      <div class="sec-hd">
        <div class="sec-pill">Providers</div>
        <h2>Two models. One API.</h2>
        <p>Switch between the world's best AI models with a single parameter — no re-implementation, no SDK swaps.</p>
      </div>
      <div class="prov-grid">
        <div class="prov-card">
          <div class="prov-hd">
            <div class="prov-logo pl-claude">&#129000;</div>
            <div>
              <div class="prov-name">Anthropic Claude</div>
              <div class="prov-model">claude-haiku-4-5</div>
            </div>
          </div>
          <p class="prov-desc">
            Fast, context-aware, and precise. The default provider.
            Exceptional at complex reasoning, structured outputs, and
            RAG pipelines. Low latency. High consistency.
          </p>
          <div class="prov-tags">
            <span class="ptag pt-c">Default Provider</span>
            <span class="ptag pt-c">RAG-Ready</span>
            <span class="ptag pt-c">Long Context</span>
            <span class="ptag pt-c">Low Latency</span>
          </div>
        </div>
        <div class="prov-card">
          <div class="prov-hd">
            <div class="prov-logo pl-openai">&#129001;</div>
            <div>
              <div class="prov-name">OpenAI GPT-4o</div>
              <div class="prov-model">gpt-4o-mini</div>
            </div>
          </div>
          <p class="prov-desc">
            Cost-efficient and industry-standard. Pass
            <code style="color:var(--green);font-size:12px">"provider":"openai"</code> to switch instantly.
            Ideal when clients prefer the OpenAI ecosystem — zero extra config.
          </p>
          <div class="prov-tags">
            <span class="ptag pt-o">Drop-in Switch</span>
            <span class="ptag pt-o">Cost Efficient</span>
            <span class="ptag pt-o">GPT-4o Quality</span>
            <span class="ptag pt-o">Industry Standard</span>
          </div>
        </div>
      </div>
    </div>

    <!-- ── CODE ── -->
    <div class="code-section" id="quickstart">
      <div class="sec-hd">
        <div class="sec-pill">Quick Start</div>
        <h2>Hit the API in seconds</h2>
        <p>Production-ready examples. Copy. Paste. Ship.</p>
      </div>
      <div class="code-card">
        <div class="code-topbar">
          <div class="code-tabs">
            <div class="ctab on">cURL</div>
            <div class="ctab">Python</div>
            <div class="ctab">JavaScript</div>
          </div>
          <div class="code-badge">&#9109; REST API</div>
        </div>
        <div class="code-body">
<pre><span class="cc"># Single-turn Q&amp;A ── Claude (default) ─────────────────────────────────</span>
<span class="cm">curl</span> -X POST <span class="cu">https://lance-ai-api-production.up.railway.app/ask</span> \
  -H <span class="ch">"Content-Type"</span>: <span class="cv">"application/json"</span> \
  -H <span class="ch">"X-API-Key"</span>: <span class="cv">"YOUR_SERVICE_KEY"</span> \
  -d <span class="cj">'</span><span class="cj">{</span>
    <span class="ck">"question"</span>: <span class="cs">"Explain RAG in 3 sentences"</span>,
    <span class="ck">"provider"</span>: <span class="cs">"claude"</span>
  <span class="cj">}'</span>

<span class="cc"># Multi-turn chat ── start a new session ────────────────────────────────</span>
<span class="cm">curl</span> -X POST <span class="cu">https://lance-ai-api-production.up.railway.app/chat</span> \
  -H <span class="ch">"Content-Type"</span>: <span class="cv">"application/json"</span> \
  -H <span class="ch">"X-API-Key"</span>: <span class="cv">"YOUR_SERVICE_KEY"</span> \
  -d <span class="cj">'</span><span class="cj">{</span>
    <span class="ck">"message"</span>: <span class="cs">"What can you help me build?"</span>,
    <span class="ck">"provider"</span>: <span class="cs">"openai"</span>
  <span class="cj">}'</span>

<span class="cc"># Continue the same conversation ─────────────────────────────────────────</span>
<span class="cm">curl</span> -X POST <span class="cu">https://lance-ai-api-production.up.railway.app/chat</span> \
  -H <span class="ch">"Content-Type"</span>: <span class="cv">"application/json"</span> \
  -H <span class="ch">"X-API-Key"</span>: <span class="cv">"YOUR_SERVICE_KEY"</span> \
  -d <span class="cj">'</span><span class="cj">{</span>
    <span class="ck">"message"</span>: <span class="cs">"Tell me more about option 2"</span>,
    <span class="ck">"session_id"</span>: <span class="cs">"&lt;returned-session-id&gt;"</span>,
    <span class="ck">"provider"</span>: <span class="cs">"openai"</span>
  <span class="cj">}'</span></pre>
        </div>
      </div>
    </div>

    <!-- ── CTA ── -->
    <div class="cta-wrap">
      <h2>Ready to build?</h2>
      <p>Explore every endpoint, fire live requests, and integrate in minutes.</p>
      <div class="cta-btns">
        <a href="/docs" class="btn-primary">&#9889; Open Swagger UI</a>
        <a href="/models" class="btn-outline">View Available Models</a>
      </div>
    </div>

    <!-- ── FOOTER ── -->
    <footer>
      <div class="foot-inner">
        <div class="foot-left">
          <div class="foot-brand">
            <div class="logo-icon" style="width:26px;height:26px;border-radius:7px;">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                <path d="M13 2L3 14h8l-1 8 10-12h-8l1-8z" fill="white"/>
              </svg>
            </div>
            <span class="foot-brand-name">Lance AI API</span>
          </div>
          <div class="foot-copy">&#169; 2026 Lance Galicia &nbsp;&#183;&nbsp; AI Engineer &amp; RAG Systems Builder</div>
        </div>
        <div class="foot-links">
          <a href="/docs">Swagger UI</a>
          <a href="/health">Health</a>
          <a href="/models">Models</a>
          <a href="#endpoints">Endpoints</a>
          <a href="#quickstart">Quick Start</a>
        </div>
      </div>
    </footer>

  </div>
</body>
</html>"""


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
