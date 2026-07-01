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

```
config.py     # env vars, Ollama client
main.py       # FastAPI app: /health, /extract
banks/
  __init__.py     # detect_bank(pdf_bytes), extract(pdf_bytes, bank, corrections) dispatch
  base.py         # Transaction model, call_llm(), response parsing, generic fallback extractor
  itau.py         # Itaú extractor + prompt
  nubank.py       # Nubank extractor (cartão + extrato) + prompts
  bradesco.py     # Bradesco extractor + prompt
  mercadopago.py  # Mercado Pago extractor + prompt
```

Each bank module owns its extraction strategy (PDF pre-processing) and its prompt hint, and exposes a single `async def extract(pdf_bytes, corrections) -> (transactions, extracted_text)`. Adding a new bank means adding a new module to `banks/` and wiring it into the dispatcher in `banks/__init__.py` — no changes needed elsewhere.
