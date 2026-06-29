"""
PDF Extractor Server

Receives bank statement text and returns structured transactions
using a locally running LLM via Ollama (OpenAI-compatible API).

Endpoints:
  GET  /health   → {"status": "ok", "model": "<model>"}
  POST /extract  → {"text": str, "bank": str} → {"transactions": [...]}

Each transaction:
  {"date": "YYYY-MM-DD", "description": str, "amount": "0.00"}
"""
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import date

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8001"))

client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"Starting pdf-extractor-server — model: {MODEL}, ollama: {OLLAMA_BASE_URL}")
    yield


app = FastAPI(title="pdf-extractor-server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ExtractRequest(BaseModel):
    text: str
    bank: str


class Transaction(BaseModel):
    date: str
    description: str
    amount: str


class ExtractResponse(BaseModel):
    transactions: list[Transaction]


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.post("/extract", response_model=ExtractResponse)
async def extract(req: ExtractRequest):
    transactions = await _extract_with_llm(req.text, req.bank)
    return ExtractResponse(transactions=transactions)


_SYSTEM_PROMPT = """\
You are a financial data extractor. You receive raw text from a Brazilian bank statement
and must return a JSON array of all financial transactions you find.

Rules:
- Only include actual debits or charges (purchases, transfers, withdrawals, fees).
- Do NOT include: totals, balances, credit payments ("Pagamento da fatura"), interest summaries, page numbers, or headers.
- date: use YYYY-MM-DD format. If only day/month appear, infer the year from context.
- description: merchant or transfer name, clean and concise. Include installment info if present (e.g. "Parcela 2/3").
- amount: positive decimal string with 2 decimal places (e.g. "123.45"). Never negative.

Respond with ONLY a valid JSON array. No markdown, no explanation, no extra text.
Example: [{"date": "2025-03-15", "description": "SUPERMERCADO ABC", "amount": "89.90"}]
"""


async def _extract_with_llm(text: str, bank: str) -> list[Transaction]:
    today = date.today().isoformat()
    user_prompt = f"Bank: {bank}\nToday: {today}\n\nStatement text:\n{text}"

    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=4096,
    )

    raw = response.choices[0].message.content or ""
    return _parse_response(raw)


def _parse_response(raw: str) -> list[Transaction]:
    # Strip markdown code fences if the model wraps output
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    # Find the first JSON array in the response
    m = re.search(r"\[.*\]", raw, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(0))
    except json.JSONDecodeError:
        return []

    result = []
    for item in items:
        if not isinstance(item, dict):
            continue
        txn_date = _normalize_date(item.get("date", ""))
        description = str(item.get("description", "")).strip()
        amount = _normalize_amount(item.get("amount", ""))
        if txn_date and description and amount:
            result.append(Transaction(date=txn_date, description=description, amount=amount))

    return result


def _normalize_date(value: str) -> str:
    value = str(value).strip()
    # Already ISO
    if re.match(r"^\d{4}-\d{2}-\d{2}$", value):
        return value
    # DD/MM/YYYY or DD/MM/YY
    m = re.match(r"^(\d{2})/(\d{2})/(\d{2,4})$", value)
    if m:
        day, month, year = m.groups()
        if len(year) == 2:
            year = "20" + year
        return f"{year}-{month}-{day}"
    return ""


def _normalize_amount(value) -> str:
    s = str(value).strip()
    # Remove currency symbols
    s = re.sub(r"[R$\s]", "", s)
    # Handle BR format: 1.234,56 → 1234.56
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        f = float(s)
        if f <= 0:
            return ""
        return f"{abs(f):.2f}"
    except (ValueError, TypeError):
        return ""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
