from fastapi import FastAPI
from pydantic import BaseModel
import anthropic
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

app = FastAPI(title="Lance's AI API", description="Powered by Claude", version="1.0")
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


# --- What the user sends in ---
class Question(BaseModel):
    question: str
    context: str = ""   # optional background info


# --- What we send back ---
class Answer(BaseModel):
    answer: str
    tokens_used: int


@app.get("/")
def root():
    return {"status": "live", "message": "Lance's AI API is running"}


@app.post("/ask", response_model=Answer)
def ask(payload: Question):
    system = "You are a helpful AI assistant. Answer clearly and concisely."
    if payload.context:
        system += f"\n\nContext you should use:\n{payload.context}"

    response = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=500,
        system=system,
        messages=[{"role": "user", "content": payload.question}]
    )

    return Answer(
        answer=response.content[0].text,
        tokens_used=response.usage.input_tokens + response.usage.output_tokens
    )
