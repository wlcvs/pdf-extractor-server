# CLAUDE.md

## What this is

HTTP sidecar for [debt-tracker-django](https://github.com/wlcvs/debt-tracker-django). Receives a PDF bank statement and returns structured transactions using a local LLM via Ollama.

## Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI (Python 3.12+) |
| LLM runtime | Ollama (OpenAI-compatible API) |
| Default model | `qwen2.5:3b` |
| PDF parsing | pdfplumber |

## Endpoints

```
GET  /health   → {"status": "ok", "model": "..."}
POST /extract  multipart: pdf=<file>, bank=<str hint>, corrections=<json>
             → {"bank": "...", "transactions": [...], "extracted_text": "..."}
```

## Running

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8001

# or via env vars
OLLAMA_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:3b PORT=8001 python main.py
```

Requires Ollama running locally with the target model pulled:
```bash
ollama pull qwen2.5:3b
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible base URL |
| `LLM_MODEL` | `qwen2.5:3b` | Model name to use for extraction |
| `PORT` | `8001` | Server port |
| `HOST` | `0.0.0.0` | Bind address |

## Architecture

```
config.py     # env vars, Ollama client (client, MODEL, OLLAMA_BASE_URL, PORT, HOST)
main.py       # FastAPI app: /health, /extract — no extraction logic
banks/
  __init__.py     # detect_bank(pdf_bytes), extract(pdf_bytes, bank, corrections) dispatch
  base.py         # Transaction model, call_llm(), parse_response()/norm_date()/norm_amount(), extract_generic() fallback
  itau.py         # Itaú extractor + prompt hint
  nubank.py       # Nubank extractor (cartão + extrato) + prompt hints
  bradesco.py     # Bradesco extractor + system prompt
  mercadopago.py  # Mercado Pago extractor + prompt hint
```

Each bank module is self-contained: its PDF pre-processing, its prompt (as a module-level constant), and a single `async def extract(pdf_bytes, corrections) -> (transactions, extracted_text)`. **Adding a new bank = adding a new module to `banks/`** and wiring it into the `if/elif` dispatch in `banks/__init__.py:extract()` — nothing else changes.

## `corrections` parameter

Optional JSON array of previously missed transactions injected as few-shot examples into the system prompt. Used by debt-tracker-django to improve recall over time via user feedback.

```json
[{"date": "2026-05-11", "description": "SUPERMERCADO ABC", "amount": "89.90"}]
```

## Supported banks

| Bank | Strategy |
|---|---|
| Nubank (cartão) | Per-page extraction on `TRANSAÇÕES` pages |
| Nubank (extrato) | Per-page extraction on all pages, dedup by (date, desc, amount) |
| Itaú | Extracts only the left column of the `DATA / ESTABELECIMENTO` table |
| Bradesco | Rule-based pre-processing into `YYYY-MM-DD DESCRIPTION AMOUNT` lines, then LLM |
| Mercado Pago | Extracts only the `Detalhes de consumo` section |
| Unknown | Generic full-text LLM extraction |

## Rules

- **Commits:** Conventional Commits in English.
- Bank-specific extractors pre-process PDFs to give the LLM clean, minimal input — avoid sending raw full-page text to the LLM when possible.
- The `_CREDIT_RE` regex in `_parse()` is the last line of defense against credits/payments slipping through — update it when new false positives appear.
- Test by running the server and hitting `/extract` with a real PDF from `extratos/` in debt-tracker-django.
