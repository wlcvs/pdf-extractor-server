# bank-statement-extractor

FastAPI sidecar that extracts structured transactions from Brazilian bank statement PDFs using a local LLM served through [Ollama](https://ollama.com). Built as a companion service for [debt-tracker-django](https://github.com/wlcvs/debt-tracker-django), but usable standalone.

Each supported bank gets its own text pre-processing strategy — the PDF is trimmed down to clean, minimal input before it's sent to the LLM, instead of dumping raw full-page text.

## Endpoints

```
GET  /health   → {"status": "ok", "model": "..."}
POST /extract  multipart: pdf=<file>, bank=<str optional hint>, corrections=<json>
             → {"bank": "...", "transactions": [...], "extracted_text": "..."}
```

`corrections` is an optional JSON array of previously missed transactions, injected as few-shot examples into the prompt to improve recall over time:

```json
[{"date": "2026-05-11", "description": "SUPERMERCADO ABC", "amount": "89.90"}]
```

## Supported banks

| Bank | Strategy |
|---|---|
| Nubank (cartão) | Per-page extraction on `TRANSAÇÕES` pages |
| Nubank (extrato) | Per-page extraction on all pages, dedup by (date, description, amount) |
| Itaú | Extracts only the left column of the `DATA / ESTABELECIMENTO` table |
| Bradesco | Rule-based pre-processing into `YYYY-MM-DD DESCRIPTION AMOUNT` lines, then LLM |
| Mercado Pago | Extracts only the `Detalhes de consumo` section |
| Unknown | Generic full-text LLM extraction |

## Running

Requires Ollama running locally with the target model pulled:

```bash
ollama pull qwen2.5:3b
```

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8001

# or via env vars
OLLAMA_BASE_URL=http://localhost:11434/v1 LLM_MODEL=qwen2.5:3b PORT=8001 python main.py
```

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible base URL |
| `LLM_MODEL` | `qwen2.5:3b` | Model name to use for extraction |
| `PORT` | `8001` | Server port |
| `HOST` | `0.0.0.0` | Bind address |

## Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI (Python 3.12+) |
| LLM runtime | Ollama (OpenAI-compatible API) |
| PDF parsing | pdfplumber |

## Architecture

All logic lives in `main.py`:

- `_detect_bank(pdf_bytes)` — keyword-based bank detection from plain text
- `_extract(pdf_bytes, bank, corrections)` — dispatches to a bank-specific extractor
- Bank extractors (`_extract_itau`, `_extract_nubank`, `_extract_bradesco`, `_extract_mercadopago`) — each pre-processes the PDF into clean text before calling the LLM
- `_call_llm(text, bank, ...)` — sends text + system prompt to Ollama, returns `list[Transaction]`
- `_parse(raw)` — extracts the JSON array from the LLM response, normalizes dates/amounts, filters out credits/payments
