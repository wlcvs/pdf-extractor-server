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
    extracted_text: str = ""


@app.get("/health")
async def health():
    return {"status": "ok", "model": MODEL}


@app.post("/extract", response_model=ExtractResponse)
async def extract(pdf: UploadFile = File(...), bank: str = Form(""), corrections: str = Form("[]")):
    pdf_bytes = await pdf.read()
    detected_bank = bank or _detect_bank(pdf_bytes)
    try:
        corrections_list = json.loads(corrections) if corrections else []
    except (json.JSONDecodeError, ValueError):
        corrections_list = []
    transactions, extracted_text = await _extract(pdf_bytes, detected_bank, corrections_list)
    return ExtractResponse(bank=detected_bank, transactions=transactions, extracted_text=extracted_text)


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

async def _extract(pdf_bytes: bytes, bank: str, corrections: list[dict]) -> tuple[list[Transaction], str]:
    if bank == "Itaú":
        return await _extract_itau(pdf_bytes, corrections)
    if bank == "Nubank":
        return await _extract_nubank(pdf_bytes, corrections)
    if bank == "Bradesco":
        return await _extract_bradesco(pdf_bytes, corrections)
    if bank == "Mercado Pago":
        return await _extract_mercadopago(pdf_bytes, corrections)
    # Unknown: generic extraction
    text = _plain_text(pdf_bytes)
    txns = await _call_llm(text, bank, corrections=corrections)
    return txns, text


# ── Itaú ───────────────────────────────────────────────────────────────────────

async def _extract_itau(pdf_bytes: bytes, corrections: list[dict]) -> tuple[list[Transaction], str]:
    """
    Itaú fatura: extract only the transaction table rows (DATA | ESTABELECIMENTO | VALOR).
    The PDF has billing slips and installment simulations on other pages — skip those.
    """
    text = _itau_transaction_rows(pdf_bytes)
    if not text:
        return [], ""
    hint = (
        "\n\nItaú fatura transaction table (DATA | ESTABELECIMENTO | VALOR EM R$):\n"
        "- First line is the header: DATA  ESTABELECIMENTO  VALOREMR$\n"
        "- Transaction line: DD/MM  CODE  amount  (e.g. '27/03 DISTRIBUIDOR-CTEI03/03 156,68')\n"
        "- Continuation line: merchant name on the next line (e.g. 'MORADIA.FRANCODAROC')\n"
        "- Combine code + continuation as description.\n"
        "- Skip: 'Lançamentosnocartão', 'LTotaldos', totals."
    )
    txns = await _call_llm(text, "Itaú", extra_hint=hint, max_tokens=512, corrections=corrections)
    return txns, text


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

async def _extract_nubank(pdf_bytes: bytes, corrections: list[dict]) -> tuple[list[Transaction], str]:
    plain = _plain_text(pdf_bytes)
    if "Movimentações" in plain:
        return await _extract_nubank_extrato(pdf_bytes, corrections)
    return await _extract_nubank_cartao(pdf_bytes, corrections)


async def _extract_nubank_extrato(pdf_bytes: bytes, corrections: list[dict]) -> tuple[list[Transaction], str]:
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
    all_text_parts = []

    for page_text in pages_text:
        if not page_text.strip():
            continue
        all_text_parts.append(page_text)
        page_txns = await _call_llm(page_text, "Nubank", extra_hint=hint, corrections=corrections)
        for t in page_txns:
            key = (t.date, t.description, t.amount)
            if key not in seen:
                seen.add(key)
                all_transactions.append(t)

    return all_transactions, "\n\n---\n\n".join(all_text_parts)


async def _extract_nubank_cartao(pdf_bytes: bytes, corrections: list[dict]) -> tuple[list[Transaction], str]:
    """
    Nubank credit card (fatura): transactions appear on pages labelled 'TRANSAÇÕES'.
    Process those pages individually to keep each LLM call small.
    """
    hint = (
        "\n\nNubank credit card (fatura) format:\n"
        "Each transaction line: 'DD MMM •••• NNNN  MERCHANT NAME  R$ 68,59'\n"
        "Portuguese month → number: JAN=01 FEV=02 MAR=03 ABR=04 MAI=05 JUN=06 JUL=07 AGO=08 SET=09 OUT=10 NOV=11 DEZ=12\n"
        "Year is 2026 (from the statement header).\n"
        "Skip: lines starting with 'IOF de', lines with negative amounts (−R$), lines starting with 'Pagamento', totals."
    )
    all_transactions: list[Transaction] = []
    seen: set[tuple] = set()
    all_text_parts = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if "TRANSAÇÕES" not in text:
                continue
            all_text_parts.append(text)
            page_txns = await _call_llm(text, "Nubank", extra_hint=hint, max_tokens=2048, corrections=corrections)
            for t in page_txns:
                key = (t.date, t.description, t.amount)
                if key not in seen:
                    seen.add(key)
                    all_transactions.append(t)

    return all_transactions, "\n\n---\n\n".join(all_text_parts)


# ── Bradesco ───────────────────────────────────────────────────────────────────

_BRADESCO_SKIP = re.compile(
    r"TED-TRANSF ELET DISPON|PIX RECEBIDO|COD\. LANC\. 0|RENTAB\.INVEST",
    re.IGNORECASE,
)
_BR_AMOUNT = re.compile(r"(\d{1,3}(?:\.\d{3})*,\d{2})")
_DATE_PREFIX = re.compile(r"^(\d{2}/\d{2}/(\d{4}))\s*")


_BRADESCO_SYSTEM = """\
Convert each input line to a JSON object. Every line is a confirmed debit transaction — do NOT skip, filter, or deduplicate any of them.

Each line format: YYYY-MM-DD DESCRIPTION AMOUNT

- date: copy exactly as YYYY-MM-DD
- description: the text between the date and the last number on the line
- amount: the last number on the line, already in decimal format (e.g. 186.69), copy as a string with 2 decimal places

Output ONLY a valid JSON array:
[{"date":"2026-05-11","description":"PIX ENVIADO","amount":"186.69"},...]

Include ALL lines without exception."""


async def _extract_bradesco(pdf_bytes: bytes, corrections: list[dict]) -> tuple[list[Transaction], str]:
    text = _bradesco_clean_lines(pdf_bytes)
    if not text:
        return [], ""
    txns = await _call_llm(text, "Bradesco", system_override=_BRADESCO_SYSTEM, max_tokens=2048, corrections=corrections)
    return txns, text


def _bradesco_clean_lines(pdf_bytes: bytes) -> str:
    """
    Pre-process Bradesco extrato into unambiguous 'YYYY-MM-DD DESCRIPTION R$AMOUNT' lines.

    Entry structure (order in PDF):
      [date] TYPE_LABEL        ← optional type label (may share line with date)
      [date] DocNum D₁ D₂      ← amounts: second-to-last=debit, last=balance
      DES:/REMET.: name DD/MM  ← recipient name (comes AFTER amounts)

    We buffer each pending entry and only emit when the next entry starts, so we
    can capture the DES: that follows the amounts line.
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

    p_date = None
    p_type = None
    p_desc = None
    p_debit = None
    p_skip = False
    has_pending = False

    def flush():
        nonlocal has_pending, p_date, p_type, p_desc, p_debit, p_skip
        if has_pending and not p_skip and p_debit and p_date:
            val = float(p_debit.replace(".", "").replace(",", "."))
            if val > 0:
                d, m, y = p_date.split("/")
                desc = p_desc or p_type or "DÉBITO"
                result.append(f"{y}-{m}-{d} {desc} {val:.2f}")
        has_pending = False
        p_type = None
        p_desc = None
        p_debit = None
        p_skip = False

    for line in section_lines:
        dm = _DATE_PREFIX.match(line)
        if dm:
            current_date = dm.group(1)
            line = line[dm.end():].strip()

        if not line:
            continue

        amounts = _BR_AMOUNT.findall(line)

        # Stop at totals line
        if line.startswith("Total") and amounts:
            flush()
            continue

        # Recipient name line — update pending description
        if line.startswith("DES:") or line.startswith("REMET.") or line.startswith("REM:"):
            raw = line.split(":", 1)[-1] if ":" in line else line[5:]
            desc = re.sub(r"\s+\d{2}/\d{2}$", "", raw).strip()
            if has_pending and p_desc is None:
                p_desc = desc
            continue

        # CONTR line (loan installment ref) — no desc update needed
        if line.startswith("CONTR") and not amounts:
            continue

        # Amounts line: DocNum + debit + balance
        if len(amounts) >= 2 and current_date:
            if has_pending and p_debit is None:
                # Fill debit for the current pending entry; date may have changed on this line
                if _BRADESCO_SKIP.search(line):
                    p_skip = True
                p_debit = amounts[-2]
                p_date = current_date
            else:
                # Standalone amounts (type was inline with date, or two rows back-to-back)
                flush()
                p_date = current_date
                p_debit = amounts[-2]
                p_skip = bool(_BRADESCO_SKIP.search(line))
                has_pending = True
            continue

        # Type label line — start new pending entry
        if re.match(r"^[A-Z\*][A-Z\s\-\.\*\/]+$", line) and not amounts:
            flush()
            p_date = current_date
            p_type = line
            p_skip = bool(_BRADESCO_SKIP.search(line))
            has_pending = True

    flush()
    return "\n".join(result)


# ── Mercado Pago ───────────────────────────────────────────────────────────────

async def _extract_mercadopago(pdf_bytes: bytes, corrections: list[dict]) -> tuple[list[Transaction], str]:
    """Extract only the 'Detalhes de consumo' transaction section from Mercado Pago."""
    text = _mercadopago_transaction_section(pdf_bytes)
    hint = (
        "\n\nMercado Pago transaction lines format:\n"
        "'DD/MM  MERCHANT  R$ 111,23' or 'DD/MM  MERCHANT  Parcela 2 de 3  R$ 111,23'.\n"
        "Skip: 'Pagamento da fatura', 'Total R$' lines."
    )
    txns = await _call_llm(text, "Mercado Pago", extra_hint=hint, max_tokens=512, corrections=corrections)
    return txns, text


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
    text: str, bank: str, extra_hint: str = "", max_tokens: int = 2048,
    system_override: str = "", corrections: list[dict] | None = None,
) -> list[Transaction]:
    system = system_override if system_override else (_SYSTEM + extra_hint)
    if corrections:
        examples = "\n".join(
            f"- {c['date']} {c['description']} {c['amount']}"
            for c in corrections[:8]
        )
        system += f"\n\nPreviously missed transactions for {bank} that must always be included if they appear:\n{examples}"
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
