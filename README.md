# Axiom AI

**Production-grade AI infrastructure** — Claude, GPT-4.1, Gemini 2.5, and Groq behind one clean REST API. 4 providers. 9 models. One endpoint.

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Claude](https://img.shields.io/badge/Claude-Haiku_4.5_·_Sonnet_4.6_·_Opus_4.7-cc785c?style=flat-square)](https://anthropic.com)
[![OpenAI](https://img.shields.io/badge/GPT--4.1_Mini_·_GPT--4.1-412991?style=flat-square&logo=openai&logoColor=white)](https://openai.com)
[![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash_·_2.5_Pro-4285F4?style=flat-square&logo=google&logoColor=white)](https://deepmind.google/technologies/gemini/)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.3_70B_·_Llama_3.1_8B-F55036?style=flat-square)](https://groq.com)
[![Railway](https://img.shields.io/badge/Railway-0B0D0E?style=flat-square&logo=railway)](https://railway.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](LICENSE)

**Live:** [axiom-ai-production-aaec.up.railway.app](https://axiom-ai-production-aaec.up.railway.app)
**Docs:** [axiom-ai-production-aaec.up.railway.app/docs](https://axiom-ai-production-aaec.up.railway.app/docs)
**Health:** [axiom-ai-production-aaec.up.railway.app/health](https://axiom-ai-production-aaec.up.railway.app/health)

---

## What It Is

A quad-provider AI API that routes requests to **Claude (Anthropic)**, **GPT-4o (OpenAI)**, **Gemini (Google)**, or **Groq (LPU)** behind a single, unified interface. Switch providers and models per-request with one field. No SDK swaps. No re-implementation. 9 models total.

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
curl -X POST https://axiom-ai-production-aaec.up.railway.app/ask \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"question": "What is RAG?", "provider": "claude"}'

# Multi-turn chat — OpenAI
curl -X POST https://axiom-ai-production-aaec.up.railway.app/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"message": "Hello!", "provider": "openai"}'

# Real-time streaming
curl -N -X POST https://axiom-ai-production-aaec.up.railway.app/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: YOUR_KEY" \
  -d '{"question": "Explain transformers", "provider": "claude"}'
```

---

## Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI + Uvicorn |
| AI Providers | Claude (Anthropic) · GPT-4o (OpenAI) · Gemini (Google) · Groq LPU |
| Models | 9 total across 4 providers |
| Streaming | Server-Sent Events (SSE) |
| Auth | API Key (`X-API-Key` header) |
| Rate Limiting | slowapi — 20 req/min per IP |
| Deployment | Railway · Docker |
| Language | Python 3.11 |

---

## Features

- **Quad provider** — Claude, GPT-4o, Gemini, and Groq behind one API, switchable per request
- **9 models** — Haiku 4.5 / Sonnet 4.6 / Opus 4.7 · GPT-4o Mini / GPT-4o · Gemini 2.0 Flash / 2.5 Pro · Llama 3.3 70B / Mixtral 8x7B
- **Streaming** — real-time token-by-token output via SSE from all 4 providers
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
GEMINI_API_KEY=your_gemini_key     # optional — enables Google Gemini
GROQ_API_KEY=your_groq_key        # optional — enables Groq LPU inference
SERVICE_API_KEY=your_service_key   # leave empty for open dev access
```

---

> Built with Claude Code 💛⚡
