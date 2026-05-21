# Lance's AI API

**Production-ready AI API** supporting Claude (Anthropic) and GPT-4o (OpenAI) — deployed on Railway, built with FastAPI.

[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Claude](https://img.shields.io/badge/Claude-Haiku_4.5-cc785c?style=flat-square)](https://anthropic.com)
[![Railway](https://img.shields.io/badge/Deployed_on-Railway-0B0D0E?style=flat-square&logo=railway)](https://railway.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**Live:** [lance-ai-api-production.up.railway.app](https://lance-ai-api-production.up.railway.app)  
**Docs:** [lance-ai-api-production.up.railway.app/docs](https://lance-ai-api-production.up.railway.app/docs)  
**Health:** [lance-ai-api-production.up.railway.app/health](https://lance-ai-api-production.up.railway.app/health)

---

## What This Is

A dual-provider AI API that routes requests to either **Claude (Anthropic)** or **GPT-4o Mini (OpenAI)** behind a single, clean REST interface. Built with production concerns in mind: API key auth, per-IP rate limiting, CORS, structured error handling, real-time streaming, and full usage analytics.

This is the backend I'd wire up for any client that needs to add AI to their product without building the provider integration from scratch.

---

## Features

- **Dual provider** — Claude Haiku 4.5 and GPT-4o Mini behind one API
- **Streaming** — real-time token-by-token output via Server-Sent Events (SSE)
- **Multi-turn chat** — session memory with rolling 20-message window
- **RAG-ready** — pass `context` to any `/ask` call to ground answers in your data
- **Custom system prompts** — override the default persona per request or per session
- **Usage analytics** — track total requests, tokens, and breakdown by provider/endpoint
- **Rate limiting** — 20 req/min per IP via SlowAPI
- **API key auth** — `X-API-Key` header (open in dev, enforced in production)
- **CORS** — browser-callable from any frontend
- **Branded Swagger UI** — dark-themed interactive docs at `/docs`
- **Dockerized** — single `docker run` for local or self-hosted deployment

---

## Endpoints

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| `GET` | `/` | ✗ | Landing page |
| `GET` | `/ping` | ✗ | Liveness probe — use for uptime monitors |
| `GET` | `/health` | ✗ | Full health check with uptime + usage stats |
| `GET` | `/models` | ✗ | Available providers, context windows, capabilities |
| `GET` | `/usage` | ✓ | Cumulative request and token usage |
| `POST` | `/ask` | ✓ | Single-turn Q&A (Claude or OpenAI) |
| `POST` | `/chat` | ✓ | Multi-turn conversation with session memory |
| `POST` | `/stream` | ✓ | Streaming Q&A via SSE |
| `GET` | `/session/{id}` | ✓ | View conversation history |
| `DELETE` | `/session/{id}` | ✓ | Clear a session |
| `GET` | `/docs` | ✗ | Interactive API docs |

---

## Quick Start

### Ask a question (Claude)
```bash
curl -X POST https://lance-ai-api-production.up.railway.app/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"question": "What is retrieval-augmented generation?"}'
```

**Response:**
```json
{
  "answer": "Retrieval-Augmented Generation (RAG) is a technique that combines...",
  "provider": "claude",
  "tokens_used": 312
}
```

### Multi-turn chat with memory
```bash
# Start a session
curl -X POST .../chat \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"message": "My name is Lance.", "provider": "claude"}'

# Returns: {"reply": "...", "session_id": "abc-123", ...}

# Continue the same conversation
curl -X POST .../chat \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"message": "What is my name?", "session_id": "abc-123"}'

# Returns: {"reply": "Your name is Lance.", ...}
```

### RAG-style grounded answer
```bash
curl -X POST .../ask \
  -H "X-API-Key: YOUR_KEY" \
  -d '{
    "question": "What is the refund policy?",
    "context": "All purchases come with a 30-day money-back guarantee...",
    "provider": "claude"
  }'
```

### Streaming (SSE)
```javascript
const res = await fetch('https://lance-ai-api-production.up.railway.app/stream', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': 'YOUR_KEY'
  },
  body: JSON.stringify({ question: 'Explain RAG in 3 sentences.' })
});

const reader = res.body.getReader();
const decoder = new TextDecoder();

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  const lines = decoder.decode(value).split('\n');
  for (const line of lines) {
    if (line.startsWith('data: ')) {
      const data = JSON.parse(line.slice(6));
      if (data.token) process.stdout.write(data.token);
      if (data.done) console.log(`\nDone. Tokens: ${data.tokens_used}`);
    }
  }
}
```

### Python
```python
import requests

BASE = "https://lance-ai-api-production.up.railway.app"
HEADERS = {"X-API-Key": "YOUR_KEY", "Content-Type": "application/json"}

# Single-turn
r = requests.post(f"{BASE}/ask", headers=HEADERS, json={
    "question": "Summarise the FIRE movement in 2 sentences.",
    "provider": "claude"
})
print(r.json()["answer"])

# Check usage
r = requests.get(f"{BASE}/usage", headers=HEADERS)
print(r.json())
```

---

## Self-Host with Docker

```bash
git clone https://github.com/Lancimoun/lance-ai-api.git
cd lance-ai-api

cp .env.example .env
# Fill in your API keys in .env

docker build -t lance-ai-api .
docker run -p 8000:8000 --env-file .env lance-ai-api
```

Open `http://localhost:8000`

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | Yes (for Claude) | Your Anthropic API key |
| `OPENAI_API_KEY` | Yes (for OpenAI) | Your OpenAI API key |
| `SERVICE_API_KEY` | Optional | If set, all auth-required endpoints need `X-API-Key: <this>`. Leave empty for open access in local dev. |
| `PORT` | Auto | Set by Railway automatically |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Framework | FastAPI 0.100+ |
| AI Providers | Anthropic Claude Haiku 4.5, OpenAI GPT-4o Mini |
| Streaming | Server-Sent Events (SSE) via `AsyncAnthropic` + `AsyncOpenAI` |
| Rate Limiting | SlowAPI (20 req/min per IP) |
| Auth | `APIKeyHeader` via FastAPI Security |
| Deployment | Railway (auto-deploy from GitHub) |
| Container | Docker (python:3.11-slim) |
| Frontend | Vanilla JS + Three.js + GSAP + VanillaTilt |

---

## Project Structure

```
lance-ai-api/
├── main.py              # FastAPI app — all routes, schemas, AI helpers
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
├── .env.example         # Environment variable template
└── templates/
    ├── index.html       # 3D animated landing page (Three.js + GSAP)
    └── docs.html        # Custom dark-themed Swagger UI
```

---

## About

Built by **Lance Galicia** — AI Engineer based in the Philippines, specialising in Claude API integrations, RAG pipelines, and production AI systems.

- GitHub: [@Lancimoun](https://github.com/Lancimoun)
- Live API: [lance-ai-api-production.up.railway.app](https://lance-ai-api-production.up.railway.app)

> *"Build once. Leverage forever."*
