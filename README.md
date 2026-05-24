# Axiom AI

**Production-grade AI infrastructure** — Claude (Anthropic) and GPT-4o (OpenAI) behind one clean REST API.

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Claude](https://img.shields.io/badge/Claude-Haiku_4.5-cc785c?style=flat-square)](https://anthropic.com)
[![OpenAI](https://img.shields.io/badge/GPT--4o_Mini-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com)
[![Railway](https://img.shields.io/badge/Railway-0B0D0E?style=flat-square&logo=railway)](https://railway.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**Live:** [lance-ai-api-production.up.railway.app](https://lance-ai-api-production.up.railway.app)
**Docs:** [lance-ai-api-production.up.railway.app/docs](https://lance-ai-api-production.up.railway.app/docs)
**Health:** [lance-ai-api-production.up.railway.app/health](https://lance-ai-api-production.up.railway.app/health)

---

## What It Is

A dual-provider AI API that routes requests to either **Claude (Anthropic)** or **GPT-4o Mini (OpenAI)** behind a single, unified interface. Switch models per-request with one field. No SDK swaps. No re-implementation.

Built with production concerns from day one: API key auth, per-IP rate limiting, CORS, real-time SSE streaming, multi-turn session memory, and full usage analytics.

---

## Endpoints

| Method | Path | Auth | Description |
|---|---|---|---|
| `GET` | `/` | — | Landing page |
| `GET` | `/ping` | — | Ultra-light liveness probe |
| `GET` | `/health` | — | Status, uptime, usage stats (JSON) |
| `GET` | `/models` | ✓ | Available providers + model metadata |
| `GET` | `/usage` | ✓ | Cumulative token + request analytics |
| `POST` | `/ask` | ✓ | Single-turn Q&A |
| `POST` | `/chat` | ✓ | Multi-turn conversation with session memory |
| `POST` | `/stream` | ✓ | Real-time streaming via Server-Sent Events |
| `GET` | `/session/{id}` | ✓ | View conversation history |
| `DELETE` | `/session/{id}` | ✓ | Clear a conversation |
| `GET` | `/docs` | — | Interactive API reference |

---

## Quickstart

```bash
# Single-turn Q&A — Claude
curl -X POST https://lance-ai-api-production.up.railway.app/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"question": "What is RAG?", "provider": "claude"}'

# Multi-turn chat — OpenAI
curl -X POST https://lance-ai-api-production.up.railway.app/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"message": "Hello!", "provider": "openai"}'

# Real-time streaming
curl -N -X POST https://lance-ai-api-production.up.railway.app/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"question": "Explain transformers", "provider": "claude"}'
```

---

## Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI + Uvicorn |
| AI Providers | Claude Haiku 4.5 (Anthropic) · GPT-4o Mini (OpenAI) |
| Streaming | Server-Sent Events (SSE) |
| Auth | API Key (`X-API-Key` header) |
| Rate Limiting | slowapi — 20 req/min per IP |
| Deployment | Railway · Docker |
| Language | Python 3.11 |

---

## Features

- **Dual provider** — Claude and GPT-4o behind one API, switchable per request
- **Streaming** — real-time token-by-token output via SSE
- **Multi-turn chat** — session memory with rolling 20-message window
- **RAG-ready** — pass `context` to any `/ask` call to ground answers in your data
- **Custom system prompts** — override persona per request
- **Usage analytics** — total requests, tokens, breakdown by provider and endpoint
- **Auth + rate limiting** — production-safe out of the box

---

## Environment Variables

```env
ANTHROPIC_API_KEY=your_anthropic_key
OPENAI_API_KEY=your_openai_key
SERVICE_API_KEY=your_service_key   # leave empty for open dev access
```

---

> Built with Claude Code 💛⚡
