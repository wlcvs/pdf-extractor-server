"""
bank-statement-extractor

Receives a PDF file and returns structured transactions using a local LLM (Ollama).
Each bank gets its own text extraction strategy to give the LLM clean, readable input —
see the `banks/` package.

Endpoints:
  GET  /health   → {"status": "ok", "model": "..."}
  POST /extract  multipart: pdf=<file>, bank=<str optional hint>
               → {"bank": "...", "transactions": [...]}
"""
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import banks
from banks import Transaction
from config import HOST, MODEL, OLLAMA_BASE_URL, PORT


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"bank-statement-extractor — model: {MODEL}, ollama: {OLLAMA_BASE_URL}")
    yield


app = FastAPI(title="bank-statement-extractor", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ExtractResponse(BaseModel):
    bank: str
    transactions: list[Transaction]
    extracted_text: str = ""


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.post("/extract", response_model=ExtractResponse)
async def extract(pdf: UploadFile = File(...), bank: str = Form(""), corrections: str = Form("[]")):
    pdf_bytes = await pdf.read()
    detected_bank = bank or banks.detect_bank(pdf_bytes)
    try:
        corrections_list = json.loads(corrections) if corrections else []
    except (json.JSONDecodeError, ValueError):
        corrections_list = []
    transactions, extracted_text = await banks.extract(pdf_bytes, detected_bank, corrections_list)
    return ExtractResponse(bank=detected_bank, transactions=transactions, extracted_text=extracted_text)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=HOST, port=PORT, reload=False)
