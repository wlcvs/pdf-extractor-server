"""
PDF Extractor Server

Receives a PDF file and returns structured transactions using a local LLM (Ollama).
Each bank gets its own text extraction strategy to give the LLM clean, readable input.

Endpoints:
  GET  /health   → {"status": "ok", "model": "..."}
  POST /extract  multipart: pdf=<file>, bank=<str optional hint>
               → {"bank": "...", "transactions": [...]}
"""
import io
import json
import os
import re
from contextlib import asynccontextmanager
from datetime import date

import pdfplumber
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.getenv("LLM_MODEL", "qwen2.5:3b")
PORT = int(os.getenv("PORT", "8001"))

client = AsyncOpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")


@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"pdf-extractor-server — model: {MODEL}, ollama: {OLLAMA_BASE_URL}")
    yield


app = FastAPI(title="pdf-extractor-server", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class Transaction(BaseModel):
    date: str
    description: str
    amount: str


class ExtractResponse(BaseModel):
    bank: str
    transactions: list[Transaction]


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.post("/extract", response_model=ExtractResponse)
async def extract(pdf: UploadFile = File(...), bank: str = Form("")):
    pdf_bytes = await pdf.read()
    detected_bank = bank or _detect_bank(pdf_bytes)
    transactions = await _extract(pdf_bytes, detected_bank)
    return ExtractResponse(bank=detected_bank, transactions=transactions)


# ── Bank detection ─────────────────────────────────────────────────────────────

def _detect_bank(pdf_bytes: bytes) -> str:
    text = _plain_text(pdf_bytes).lower()
    if "nubank" in text or "nu pagamentos" in text:
        return "Nubank"
    if "bradesco celular" in text or "banco bradesco" in text:
        return "Bradesco"
    if "itaú" in text or "banco itaú" in text or "itauunibanco" in text:
        return "Itaú"
    if "mercado pago" in text or "mercadopago" in text:
        return "Mercado Pago"
    if "bradesco" in text:
        return "Bradesco"
    return "Desconhecido"


def _plain_text(pdf_bytes: bytes) -> str:
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


# ── Bank-specific extraction dispatch ─────────────────────────────────────────

async def _extract(pdf_bytes: bytes, bank: str) -> list[Transaction]:
    if bank == "Itaú":
        return await _extract_itau(pdf_bytes)
    if bank == "Nubank":
        return await _extract_nubank(pdf_bytes)
    if bank == "Bradesco":
        return await _extract_bradesco(pdf_bytes)
    if bank == "Mercado Pago":
        return await _extract_mercadopago(pdf_bytes)
    # Unknown: generic extraction
    return await _call_llm(_plain_text(pdf_bytes), bank)


# ── Itaú ───────────────────────────────────────────────────────────────────────

async def _extract_itau(pdf_bytes: bytes) -> list[Transaction]:
    """
    Itaú fatura: extract only the transaction table rows (DATA | ESTABELECIMENTO | VALOR).
    The PDF has billing slips and installment simulations on other pages — skip those.
    """
    text = _itau_transaction_rows(pdf_bytes)
    if not text:
        return []
    hint = (
        "\n\nItaú fatura transaction table (DATA | ESTABELECIMENTO | VALOR EM R$):\n"
        "- First line is the header: DATA  ESTABELECIMENTO  VALOREMR$\n"
        "- Transaction line: DD/MM  CODE  amount  (e.g. '27/03 DISTRIBUIDOR-CTEI03/03 156,68')\n"
        "- Continuation line: merchant name on the next line (e.g. 'MORADIA.FRANCODAROC')\n"
        "- Combine code + continuation as description.\n"
        "- Skip: 'Lançamentosnocartão', 'LTotaldos', totals."
    )
    return await _call_llm(text, "Itaú", extra_hint=hint, max_tokens=512)


def _itau_transaction_rows(pdf_bytes: bytes) -> str:
    """
    Find the page with DATA/ESTABELECIMENTO header and extract only the left column
    from that header down to the totals line. Ignores billing/simulation pages.
    """
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            words = page.extract_words(x_tolerance=3, y_tolerance=3)
            word_texts = [w["text"] for w in words]
            if "DATA" not in word_texts or "ESTABELECIMENTO" not in word_texts:
                continue

            split_x = page.width * 0.60
            rows: dict[int, list] = {}
            for w in words:
                if w["x0"] > split_x:
                    continue
                key = round(w["top"])
                rows.setdefault(key, []).append(w)

            lines = []
            in_table = False
            for key in sorted(rows):
                row_words = sorted(rows[key], key=lambda w: w["x0"])
                line = " ".join(w["text"] for w in row_words)

                if not in_table:
                    if "DATA" in line and "ESTABELECIMENTO" in line:
                        in_table = True
                        lines.append(line)
                    continue

                # Stop before totals and credit limit section
                if any(t in line for t in ["Totaldos", "LTotaldos", "Limitesdecr", "Fiqueaten"]):
                    break

                lines.append(line)

            if lines:
                return "\n".join(lines)
    return ""


# ── Nubank ─────────────────────────────────────────────────────────────────────

async def _extract_nubank(pdf_bytes: bytes) -> list[Transaction]:
    plain = _plain_text(pdf_bytes)
    if "Movimentações" in plain:
        return await _extract_nubank_extrato(pdf_bytes)
    return await _extract_nubank_cartao(pdf_bytes)


async def _extract_nubank_extrato(pdf_bytes: bytes) -> list[Transaction]:
    """
    Nubank extrato has many transactions spread across pages.
    Process page by page to keep each LLM call small and fast.
    """
    pages_text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")

    hint = (
        "\n\nNubank current account (extrato) format:\n"
        "Day headers look like: '01 MAI 2026 Total de saídas - 92,49' — these are NOT transactions.\n"
        "Transaction lines come after a day header and end with a BR amount (e.g. '1.234,56').\n"
        "Skip: 'Saldo inicial', 'Saldo final', 'Rendimento', 'Nu Pagamentos', header/footer lines.\n"
        "Each transaction line: description ending with the amount."
    )

    all_transactions: list[Transaction] = []
    seen: set[tuple] = set()

    for page_text in pages_text:
        if not page_text.strip():
            continue
        page_txns = await _call_llm(page_text, "Nubank", extra_hint=hint)
        for t in page_txns:
            key = (t.date, t.description, t.amount)
            if key not in seen:
                seen.add(key)
                all_transactions.append(t)

    return all_transactions


async def _extract_nubank_cartao(pdf_bytes: bytes) -> list[Transaction]:
    """
    Nubank credit card (fatura): transactions are in a table.
    Extract table rows and feed clean text to the LLM.
    """
    lines: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            for table in page.extract_tables():
                for row in table:
                    if row and row[0]:
                        cell = str(row[0]).split("\n")[0].strip()
                        if cell:
                            lines.append(cell)

    text = "\n".join(lines)
    hint = (
        "\n\nNubank credit card (fatura) table format:\n"
        "Each row: 'DD MMM •••• NNNN  MERCHANT NAME  R$ 68,59'\n"
        "The date is 'DD MMM' (month abbreviated in Portuguese: JAN FEB MAR ABR MAI JUN JUL AGO SET OUT NOV DEZ).\n"
        "Skip: 'IOF de', rows without a date prefix."
    )
    return await _call_llm(text, "Nubank", extra_hint=hint)


# ── Bradesco ───────────────────────────────────────────────────────────────────

_BRADESCO_SKIP = re.compile(
    r"TED-TRANSF ELET DISPON|PIX RECEBIDO|COD\. LANC\. 0|RENTAB\.INVEST",
    re.IGNORECASE,
)
_BR_AMOUNT = re.compile(r"(\d{1,3}(?:\.\d{3})*,\d{2})")
_DATE_PREFIX = re.compile(r"^(\d{2}/\d{2}/(\d{4}))\s*")


async def _extract_bradesco(pdf_bytes: bytes) -> list[Transaction]:
    text = _bradesco_clean_lines(pdf_bytes)
    if not text:
        return []
    hint = (
        "\n\nEach input line is already a clean debit transaction:\n"
        "YYYY-MM-DD DESCRIPTION R$AMOUNT\n"
        "Just convert each line to JSON. Skip nothing — they are already filtered."
    )
    return await _call_llm(text, "Bradesco", extra_hint=hint, max_tokens=2048)


def _bradesco_clean_lines(pdf_bytes: bytes) -> str:
    """
    Pre-process Bradesco extrato into unambiguous 'YYYY-MM-DD DESCRIPTION R$AMOUNT' lines.

    Column structure: DocNum | Credit(R$) | Debit(R$) | Balance(R$)
    Second-to-last amount = debit (what we want). Last amount = running balance (ignore).
    Credit-only entries (incoming TED/PIX) are skipped via _BRADESCO_SKIP.
    """
    full_text = _plain_text(pdf_bytes)
    lines = full_text.splitlines()

    section_lines = []
    in_section = False
    for line in lines:
        stripped = line.strip()
        if not in_section:
            if "Histórico" in stripped and "Débito" in stripped:
                in_section = True
            continue
        if "Saldo Final" in stripped or stripped.startswith("S Saldo Final"):
            break
        if stripped:
            section_lines.append(stripped)

    result = []
    current_date = None
    current_type = None
    current_desc = None
    skip_next = False

    for line in section_lines:
        dm = _DATE_PREFIX.match(line)
        if dm:
            current_date = dm.group(1)  # DD/MM/YYYY
            line = line[dm.end():].strip()

        amounts = _BR_AMOUNT.findall(line)

        if line.startswith("DES:") or line.startswith("REMET."):
            raw = line.split(":", 1)[-1] if ":" in line else line[5:]
            current_desc = re.sub(r"\s+\d{2}/\d{2}$", "", raw).strip()
            continue

        if line.startswith("CONTR") and not amounts:
            if current_desc:
                current_desc += " " + line
            continue

        if len(amounts) >= 2 and current_date:
            if not skip_next:
                debit_str = amounts[-2]
                debit_val = float(debit_str.replace(".", "").replace(",", "."))
                if debit_val > 0:
                    desc = current_desc or current_type or "DÉBITO"
                    d, m, y = current_date.split("/")
                    result.append(f"{y}-{m}-{d} {desc} R${debit_str}")
            current_type = None
            current_desc = None
            skip_next = False
            continue

        if re.match(r"^[A-Z][A-Z\s\-\.\*]+$", line) and not amounts:
            current_type = line
            skip_next = bool(_BRADESCO_SKIP.search(line))

    return "\n".join(result)


# ── Mercado Pago ───────────────────────────────────────────────────────────────

async def _extract_mercadopago(pdf_bytes: bytes) -> list[Transaction]:
    """Extract only the 'Detalhes de consumo' transaction section from Mercado Pago."""
    text = _mercadopago_transaction_section(pdf_bytes)
    hint = (
        "\n\nMercado Pago transaction lines format:\n"
        "'DD/MM  MERCHANT  R$ 111,23' or 'DD/MM  MERCHANT  Parcela 2 de 3  R$ 111,23'.\n"
        "Skip: 'Pagamento da fatura', 'Total R$' lines."
    )
    return await _call_llm(text, "Mercado Pago", extra_hint=hint, max_tokens=512)


def _mercadopago_transaction_section(pdf_bytes: bytes) -> str:
    """Extract only lines from 'Data Movimentações' header to 'Total R$'."""
    full_text = _plain_text(pdf_bytes)
    lines = full_text.splitlines()
    result = []
    in_section = False

    for line in lines:
        stripped = line.strip()
        if not in_section:
            if "Data" in stripped and "Movimenta" in stripped:
                in_section = True
            continue
        if stripped.startswith("Total R$") or stripped.startswith("Total\xa0R$"):
            break
        if stripped:
            result.append(stripped)

    return "\n".join(result)


# ── LLM call ───────────────────────────────────────────────────────────────────

_SYSTEM = """\
You are a financial data extractor. Extract all purchase and debit transactions from this Brazilian bank statement.

INCLUDE: purchases, PIX sent, TED sent, withdrawals, fees, loan installments, debit card purchases.
EXCLUDE: incoming payments ("PIX recebido", "TED recebida", "Pagamento recebido"), credits, "Pagamento da fatura",
         balance lines, interest summaries, totals, opening/closing balance entries.

For each transaction output:
- date: YYYY-MM-DD  (infer year from the statement header)
- description: concise merchant or counterpart name; include "Parcela X/Y" if present
- amount: positive decimal string with 2 decimal places, e.g. "123.45"

Respond with ONLY a valid JSON array — no markdown, no explanation before or after.
Example: [{"date":"2026-05-11","description":"SUPERMERCADO ABC","amount":"89.90"}]"""


async def _call_llm(
    text: str, bank: str, extra_hint: str = "", max_tokens: int = 2048
) -> list[Transaction]:
    system = _SYSTEM + extra_hint
    today = date.today().isoformat()
    response = await client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Bank: {bank}\nToday: {today}\n\nStatement:\n{text}"},
        ],
        temperature=0.1,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content or ""
    return _parse(raw)


# ── Response parsing ────────────────────────────────────────────────────────────

_CREDIT_RE = re.compile(
    r"pagamento\s+da\s+fatura|pagamento\s+recebido|pix\s+recebido|ted\s+recebida?|"
    r"transf(?:er[eê]ncia)?\s+recebida?|estorno|devolu[cç][aã]o|reembolso|"
    r"cr[eé]dito\s+em\s+conta|rendimento|saldo\s+(anterior|final|inicial)|"
    r"total\s+d[ao]s?\s+(fatura|lançamentos)|cod\.\s*lanc",
    re.IGNORECASE,
)


def _parse(raw: str) -> list[Transaction]:
    raw = re.sub(r"```(?:json)?", "", raw).strip()
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
        txn_date = _norm_date(item.get("date", ""))
        desc = str(item.get("description", "")).strip()
        amount = _norm_amount(item.get("amount", ""))
        if txn_date and desc and amount and not _CREDIT_RE.search(desc):
            result.append(Transaction(date=txn_date, description=desc, amount=amount))
    return result


def _norm_date(value: str) -> str:
    v = str(value).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
        return v
    m = re.match(r"^(\d{2})/(\d{2})/(\d{2,4})$", v)
    if m:
        d, mo, y = m.groups()
        return f"{'20'+y if len(y)==2 else y}-{mo}-{d}"
    return ""


def _norm_amount(value) -> str:
    s = re.sub(r"[R$\s]", "", str(value).strip())
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        f = float(s)
        return f"{abs(f):.2f}" if f > 0 else ""
    except (ValueError, TypeError):
        return ""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=os.getenv("HOST", "0.0.0.0"), port=PORT, reload=False)
