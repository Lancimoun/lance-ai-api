"""
Lance's AI API  v2.0
────────────────────
Endpoints : GET /          health check + endpoint map
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
from dotenv import load_dotenv

from fastapi import FastAPI, Depends, HTTPException, Request
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

# ── In-memory chat sessions ────────────────────────────────────────────────────
_sessions: dict[str, list] = {}
_MAX_HISTORY = 20

# ── Schemas ────────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    context:  str = ""
    provider: str = "claude"   # "claude" | "openai"

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


class AskResponse(BaseModel):
    answer:      str
    provider:    str
    tokens_used: int


class ChatRequest(BaseModel):
    message:    str
    session_id: str = ""       # omit to start a new session
    provider:   str = "claude"

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


class ChatResponse(BaseModel):
    reply:       str
    session_id:  str
    provider:    str
    tokens_used: int


# ── AI helpers ─────────────────────────────────────────────────────────────────
_SYSTEM = "You are a helpful, precise AI assistant. Answer clearly and concisely."


def _ask_claude(messages: list, system: str = _SYSTEM) -> tuple[str, int]:
    r = claude_client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=800,
        system=system,
        messages=messages,
    )
    tokens = r.usage.input_tokens + r.usage.output_tokens
    return r.content[0].text.strip(), tokens


def _ask_openai(messages: list, system: str = _SYSTEM) -> tuple[str, int]:
    full = [{"role": "system", "content": system}] + messages
    r = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=800,
        messages=full,
    )
    tokens = r.usage.prompt_tokens + r.usage.completion_tokens
    return r.choices[0].message.content.strip(), tokens


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
        "endpoints": {
            "GET  /":       "landing page",
            "GET  /health": "health check (JSON)",
            "GET  /models": "available providers",
            "POST /ask":    "single-turn Q&A",
            "POST /chat":   "multi-turn conversation",
            "GET  /docs":   "interactive API docs",
        },
    }


_LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Lance's AI API</title>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap" rel="stylesheet"/>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

    :root {
      --bg: #07080f;
      --surface: rgba(255,255,255,0.04);
      --border: rgba(255,255,255,0.08);
      --accent: #7c6dfa;
      --accent2: #e96bff;
      --green: #22d3a5;
      --text: #e8eaf2;
      --muted: #6b7089;
    }

    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      overflow-x: hidden;
    }

    /* ── Background orbs ── */
    .orb {
      position: fixed;
      border-radius: 50%;
      filter: blur(120px);
      opacity: 0.18;
      pointer-events: none;
      z-index: 0;
    }
    .orb-1 { width: 600px; height: 600px; background: var(--accent); top: -200px; left: -150px; }
    .orb-2 { width: 500px; height: 500px; background: var(--accent2); top: 200px; right: -200px; }
    .orb-3 { width: 400px; height: 400px; background: var(--green); bottom: -100px; left: 30%; }

    .wrapper { position: relative; z-index: 1; max-width: 1100px; margin: 0 auto; padding: 0 24px; }

    /* ── Nav ── */
    nav {
      display: flex; align-items: center; justify-content: space-between;
      padding: 24px 0; border-bottom: 1px solid var(--border);
    }
    .nav-logo { font-size: 15px; font-weight: 700; letter-spacing: 0.05em; color: var(--text); }
    .nav-logo span { color: var(--accent); }
    .nav-links { display: flex; gap: 24px; }
    .nav-links a {
      color: var(--muted); text-decoration: none; font-size: 14px; font-weight: 500;
      transition: color .2s;
    }
    .nav-links a:hover { color: var(--text); }
    .nav-badge {
      display: flex; align-items: center; gap: 6px;
      background: rgba(34,211,165,0.12); border: 1px solid rgba(34,211,165,0.25);
      border-radius: 999px; padding: 5px 12px; font-size: 12px; font-weight: 600; color: var(--green);
    }
    .pulse {
      width: 7px; height: 7px; border-radius: 50%; background: var(--green);
      animation: pulse 2s infinite;
    }
    @keyframes pulse {
      0%,100% { box-shadow: 0 0 0 0 rgba(34,211,165,0.5); }
      50% { box-shadow: 0 0 0 6px rgba(34,211,165,0); }
    }

    /* ── Hero ── */
    .hero { text-align: center; padding: 96px 0 64px; }
    .hero-pill {
      display: inline-flex; align-items: center; gap: 8px;
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 999px; padding: 6px 16px; font-size: 12px;
      color: var(--muted); margin-bottom: 32px; letter-spacing: 0.05em; text-transform: uppercase;
    }
    .hero-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--accent2); }
    h1 {
      font-size: clamp(42px, 7vw, 80px);
      font-weight: 900;
      line-height: 1.05;
      letter-spacing: -0.03em;
      margin-bottom: 24px;
    }
    h1 .grad {
      background: linear-gradient(135deg, var(--accent) 0%, var(--accent2) 100%);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-sub {
      font-size: 18px; color: var(--muted); max-width: 540px; margin: 0 auto 48px;
      line-height: 1.6; font-weight: 400;
    }
    .hero-actions { display: flex; gap: 12px; justify-content: center; flex-wrap: wrap; }
    .btn {
      display: inline-flex; align-items: center; gap: 8px;
      padding: 12px 24px; border-radius: 10px; font-size: 14px; font-weight: 600;
      text-decoration: none; transition: all .2s; cursor: pointer; border: none;
    }
    .btn-primary {
      background: linear-gradient(135deg, var(--accent), var(--accent2));
      color: #fff;
    }
    .btn-primary:hover { opacity: 0.85; transform: translateY(-1px); }
    .btn-ghost {
      background: var(--surface); border: 1px solid var(--border); color: var(--text);
    }
    .btn-ghost:hover { background: rgba(255,255,255,0.08); transform: translateY(-1px); }

    /* ── Stats row ── */
    .stats {
      display: flex; justify-content: center; gap: 48px; flex-wrap: wrap;
      padding: 48px 0; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border);
      margin-bottom: 80px;
    }
    .stat { text-align: center; }
    .stat-num { font-size: 32px; font-weight: 800; letter-spacing: -0.02em; }
    .stat-num.purple { color: var(--accent); }
    .stat-num.pink   { color: var(--accent2); }
    .stat-num.green  { color: var(--green); }
    .stat-label { font-size: 13px; color: var(--muted); margin-top: 4px; font-weight: 500; }

    /* ── Section title ── */
    .section-title { text-align: center; margin-bottom: 48px; }
    .section-title h2 { font-size: 32px; font-weight: 800; letter-spacing: -0.02em; margin-bottom: 8px; }
    .section-title p { color: var(--muted); font-size: 15px; }

    /* ── Endpoint cards ── */
    .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 16px; margin-bottom: 80px; }
    .card {
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 16px; padding: 24px;
      transition: all .25s; position: relative; overflow: hidden;
    }
    .card::before {
      content: ''; position: absolute; inset: 0;
      background: linear-gradient(135deg, var(--card-color, var(--accent)) 0%, transparent 60%);
      opacity: 0; transition: opacity .25s; border-radius: 16px;
    }
    .card:hover { border-color: rgba(255,255,255,0.15); transform: translateY(-3px); }
    .card:hover::before { opacity: 0.06; }
    .card-header { display: flex; align-items: center; gap: 12px; margin-bottom: 12px; }
    .card-icon {
      width: 42px; height: 42px; border-radius: 10px;
      display: flex; align-items: center; justify-content: center;
      font-size: 20px; flex-shrink: 0;
    }
    .card-method {
      font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
      padding: 3px 8px; border-radius: 4px;
    }
    .get  { background: rgba(34,211,165,0.15); color: var(--green); }
    .post { background: rgba(124,109,250,0.15); color: var(--accent); }
    .card-path { font-size: 15px; font-weight: 700; color: var(--text); }
    .card-desc { font-size: 13px; color: var(--muted); line-height: 1.5; }

    /* ── Providers ── */
    .providers { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 80px; }
    .provider-card {
      background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 28px;
      display: flex; flex-direction: column; gap: 16px;
    }
    .provider-logo { font-size: 36px; }
    .provider-name { font-size: 20px; font-weight: 700; }
    .provider-model {
      display: inline-block; font-size: 12px; font-weight: 600; letter-spacing: 0.05em;
      padding: 4px 10px; border-radius: 6px; background: rgba(255,255,255,0.06);
      border: 1px solid var(--border); color: var(--muted);
    }
    .provider-desc { font-size: 13px; color: var(--muted); line-height: 1.6; }

    /* ── Code block ── */
    .code-section { margin-bottom: 80px; }
    .code-tabs { display: flex; gap: 4px; margin-bottom: -1px; }
    .code-tab {
      padding: 8px 16px; font-size: 12px; font-weight: 600; border-radius: 8px 8px 0 0;
      background: var(--surface); border: 1px solid var(--border); border-bottom: none;
      color: var(--muted); cursor: pointer; transition: all .2s;
    }
    .code-tab.active { background: #131520; color: var(--text); border-color: rgba(255,255,255,0.12); }
    .code-box {
      background: #131520; border: 1px solid rgba(255,255,255,0.12);
      border-radius: 0 12px 12px 12px; padding: 28px; overflow-x: auto;
    }
    pre { font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace; font-size: 13px; line-height: 1.7; }
    .c-dim    { color: #4a5068; }
    .c-green  { color: #22d3a5; }
    .c-blue   { color: #79b8ff; }
    .c-yellow { color: #ffd680; }
    .c-purple { color: #c792ea; }
    .c-orange { color: #ffab76; }
    .c-pink   { color: #e96bff; }

    /* ── Auth section ── */
    .auth-section {
      background: var(--surface); border: 1px solid var(--border); border-radius: 20px;
      padding: 40px; margin-bottom: 80px; display: flex; gap: 40px; align-items: flex-start;
      flex-wrap: wrap;
    }
    .auth-text { flex: 1; min-width: 240px; }
    .auth-text h3 { font-size: 22px; font-weight: 700; margin-bottom: 10px; }
    .auth-text p { font-size: 14px; color: var(--muted); line-height: 1.7; }
    .auth-rules { flex: 1; min-width: 240px; display: flex; flex-direction: column; gap: 12px; }
    .auth-rule {
      display: flex; gap: 12px; align-items: flex-start;
      background: rgba(255,255,255,0.03); border: 1px solid var(--border);
      border-radius: 10px; padding: 14px;
    }
    .auth-rule-icon { font-size: 18px; flex-shrink: 0; margin-top: 1px; }
    .auth-rule-text { font-size: 13px; line-height: 1.5; }
    .auth-rule-text strong { color: var(--text); display: block; margin-bottom: 2px; }
    .auth-rule-text span { color: var(--muted); }

    /* ── Footer ── */
    footer {
      border-top: 1px solid var(--border); padding: 32px 0;
      display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 16px;
    }
    .footer-left { font-size: 13px; color: var(--muted); }
    .footer-left strong { color: var(--text); }
    .footer-right { display: flex; gap: 16px; }
    .footer-right a {
      font-size: 13px; color: var(--muted); text-decoration: none; font-weight: 500;
      transition: color .2s;
    }
    .footer-right a:hover { color: var(--text); }

    @media (max-width: 640px) {
      .providers { grid-template-columns: 1fr; }
      .stats { gap: 28px; }
      nav { flex-wrap: wrap; gap: 12px; }
      .nav-links { display: none; }
    }
  </style>
</head>
<body>
  <div class="orb orb-1"></div>
  <div class="orb orb-2"></div>
  <div class="orb orb-3"></div>

  <div class="wrapper">

    <!-- Nav -->
    <nav>
      <div class="nav-logo">LANCE<span>.</span>AI</div>
      <div class="nav-links">
        <a href="/docs">Docs</a>
        <a href="/models">Models</a>
        <a href="/health">Health</a>
      </div>
      <div class="nav-badge"><div class="pulse"></div> Online · v2.0</div>
    </nav>

    <!-- Hero -->
    <section class="hero">
      <div class="hero-pill"><div class="dot"></div> Production API · Railway</div>
      <h1>Your AI.<br/><span class="grad">Your API.</span></h1>
      <p class="hero-sub">
        A production-ready AI backend supporting Claude (Anthropic) and GPT-4o (OpenAI).
        Built by <strong>Lance Galicia</strong> — AI Engineer &amp; RAG Systems Builder.
      </p>
      <div class="hero-actions">
        <a href="/docs" class="btn btn-primary">⚡ Try it Live</a>
        <a href="/health" class="btn btn-ghost">🔍 Health Check</a>
      </div>
    </section>

    <!-- Stats -->
    <div class="stats">
      <div class="stat">
        <div class="stat-num purple">2</div>
        <div class="stat-label">AI Providers</div>
      </div>
      <div class="stat">
        <div class="stat-num pink">20/min</div>
        <div class="stat-label">Rate Limit</div>
      </div>
      <div class="stat">
        <div class="stat-num green">20</div>
        <div class="stat-label">Message Memory</div>
      </div>
      <div class="stat">
        <div class="stat-num purple">v2.0</div>
        <div class="stat-label">Current Version</div>
      </div>
    </div>

    <!-- Endpoints -->
    <div class="section-title">
      <h2>Endpoints</h2>
      <p>Everything you need to build AI-powered products</p>
    </div>
    <div class="cards">

      <div class="card" style="--card-color:#22d3a5">
        <div class="card-header">
          <div class="card-icon" style="background:rgba(34,211,165,0.12)">🩺</div>
          <div>
            <div style="display:flex;gap:6px;align-items:center;margin-bottom:4px">
              <span class="card-method get">GET</span>
              <span class="card-path">/health</span>
            </div>
          </div>
        </div>
        <div class="card-desc">Live status check. Returns version info and the full endpoint map in JSON.</div>
      </div>

      <div class="card" style="--card-color:#7c6dfa">
        <div class="card-header">
          <div class="card-icon" style="background:rgba(124,109,250,0.12)">🧠</div>
          <div>
            <div style="display:flex;gap:6px;align-items:center;margin-bottom:4px">
              <span class="card-method post">POST</span>
              <span class="card-path">/ask</span>
            </div>
          </div>
        </div>
        <div class="card-desc">Single-turn Q&amp;A. Send a question and optional context, get a precise AI answer back instantly.</div>
      </div>

      <div class="card" style="--card-color:#e96bff">
        <div class="card-header">
          <div class="card-icon" style="background:rgba(233,107,255,0.12)">💬</div>
          <div>
            <div style="display:flex;gap:6px;align-items:center;margin-bottom:4px">
              <span class="card-method post">POST</span>
              <span class="card-path">/chat</span>
            </div>
          </div>
        </div>
        <div class="card-desc">Multi-turn conversation with memory. Pass a session_id to continue a conversation across requests.</div>
      </div>

      <div class="card" style="--card-color:#22d3a5">
        <div class="card-header">
          <div class="card-icon" style="background:rgba(34,211,165,0.12)">📡</div>
          <div>
            <div style="display:flex;gap:6px;align-items:center;margin-bottom:4px">
              <span class="card-method get">GET</span>
              <span class="card-path">/models</span>
            </div>
          </div>
        </div>
        <div class="card-desc">Lists available AI providers and their underlying models with live availability status.</div>
      </div>

      <div class="card" style="--card-color:#7c6dfa" style="grid-column: span 2">
        <div class="card-header">
          <div class="card-icon" style="background:rgba(124,109,250,0.12)">📖</div>
          <div>
            <div style="display:flex;gap:6px;align-items:center;margin-bottom:4px">
              <span class="card-method get">GET</span>
              <span class="card-path">/docs</span>
            </div>
          </div>
        </div>
        <div class="card-desc">Interactive Swagger UI. Test every endpoint live in your browser — no code needed.</div>
      </div>

    </div>

    <!-- Providers -->
    <div class="section-title">
      <h2>Providers</h2>
      <p>Switch between AI providers per request with a single field</p>
    </div>
    <div class="providers">
      <div class="provider-card">
        <div class="provider-logo">🟠</div>
        <div>
          <div class="provider-name">Anthropic Claude</div>
          <span class="provider-model">claude-haiku-4-5</span>
        </div>
        <div class="provider-desc">
          Fast, capable, and context-aware. Default provider. Excellent for RAG pipelines, structured outputs, and complex reasoning.
        </div>
      </div>
      <div class="provider-card">
        <div class="provider-logo">🟢</div>
        <div>
          <div class="provider-name">OpenAI GPT</div>
          <span class="provider-model">gpt-4o-mini</span>
        </div>
        <div class="provider-desc">
          Cost-efficient powerhouse. Pass <code style="font-size:12px;color:var(--accent)">"provider":"openai"</code> in any request body to switch instantly.
        </div>
      </div>
    </div>

    <!-- Code example -->
    <div class="code-section">
      <div class="section-title">
        <h2>Quick Start</h2>
        <p>Hit the API in seconds</p>
      </div>
      <div class="code-tabs">
        <div class="code-tab active">cURL</div>
      </div>
      <div class="code-box">
<pre><span class="c-dim"># Single-turn Q&amp;A — Claude (default)</span>
<span class="c-green">curl</span> <span class="c-blue">-X POST</span> https://lance-ai-api-production.up.railway.app/ask \\
  <span class="c-blue">-H</span> <span class="c-yellow">"Content-Type: application/json"</span> \\
  <span class="c-blue">-H</span> <span class="c-yellow">"X-API-Key: YOUR_KEY"</span> \\
  <span class="c-blue">-d</span> <span class="c-yellow">'{</span>
    <span class="c-pink">"question"</span><span class="c-yellow">:</span> <span class="c-orange">"What is RAG?"</span><span class="c-yellow">,</span>
    <span class="c-pink">"provider"</span><span class="c-yellow">:</span> <span class="c-orange">"claude"</span>
  <span class="c-yellow">}'</span>

<span class="c-dim"># Multi-turn chat — start a session</span>
<span class="c-green">curl</span> <span class="c-blue">-X POST</span> https://lance-ai-api-production.up.railway.app/chat \\
  <span class="c-blue">-H</span> <span class="c-yellow">"Content-Type: application/json"</span> \\
  <span class="c-blue">-H</span> <span class="c-yellow">"X-API-Key: YOUR_KEY"</span> \\
  <span class="c-blue">-d</span> <span class="c-yellow">'{</span>
    <span class="c-pink">"message"</span><span class="c-yellow">:</span> <span class="c-orange">"Hello! What can you help me with?"</span><span class="c-yellow">,</span>
    <span class="c-pink">"provider"</span><span class="c-yellow">:</span> <span class="c-orange">"openai"</span>
  <span class="c-yellow">}'</span></pre>
      </div>
    </div>

    <!-- Auth section -->
    <div class="auth-section">
      <div class="auth-text">
        <h3>🔐 Authentication</h3>
        <p>
          Every request to <code style="color:var(--accent);font-size:13px">/ask</code> and
          <code style="color:var(--accent);font-size:13px">/chat</code> requires an API key
          in the <code style="color:var(--accent2);font-size:13px">X-API-Key</code> header.
          Rate limiting is enforced per IP to ensure fair usage.
        </p>
      </div>
      <div class="auth-rules">
        <div class="auth-rule">
          <div class="auth-rule-icon">🔑</div>
          <div class="auth-rule-text">
            <strong>Header Required</strong>
            <span>Pass <code style="color:var(--accent2)">X-API-Key: &lt;key&gt;</code> on every request</span>
          </div>
        </div>
        <div class="auth-rule">
          <div class="auth-rule-icon">⚡</div>
          <div class="auth-rule-text">
            <strong>Rate Limit: 20/min</strong>
            <span>Per IP address. Returns HTTP 429 when exceeded</span>
          </div>
        </div>
        <div class="auth-rule">
          <div class="auth-rule-icon">🚫</div>
          <div class="auth-rule-text">
            <strong>Missing Key → 403</strong>
            <span>Invalid or absent key returns Forbidden immediately</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Footer -->
    <footer>
      <div class="footer-left">
        Built by <strong>Lance Galicia</strong> — AI Engineer &amp; RAG Systems Builder · v2.0
      </div>
      <div class="footer-right">
        <a href="/docs">Swagger UI</a>
        <a href="/health">Health</a>
        <a href="/models">Models</a>
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
    system = _SYSTEM
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

    reply, tokens = _route(payload.provider, history)

    history.append({"role": "assistant", "content": reply})
    _sessions[session_id] = history

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        provider=payload.provider,
        tokens_used=tokens,
    )
