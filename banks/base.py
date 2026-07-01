"""Shared LLM plumbing, PDF text helpers, and response parsing used by every bank module."""
import io
import json
import re
from datetime import date

import pdfplumber
from pydantic import BaseModel

from config import MODEL, client


class Transaction(BaseModel):
    date: str
    description: str
    amount: str


SYSTEM_PROMPT = """\
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


_CREDIT_RE = re.compile(
    r"pagamento\s+da\s+fatura|pagamento\s+recebido|pix\s+recebido|ted\s+recebida?|"
    r"transf(?:er[eê]ncia)?\s+recebida?|estorno|devolu[cç][aã]o|reembolso|"
    r"cr[eé]dito\s+em\s+conta|rendimento|saldo\s+(anterior|final|inicial)|"
    r"total\s+d[ao]s?\s+(fatura|lançamentos)|cod\.\s*lanc",
    re.IGNORECASE,
)


def plain_text(pdf_bytes: bytes) -> str:
    pages = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")
    return "\n".join(pages)


async def call_llm(
    text: str, bank: str, extra_hint: str = "", max_tokens: int = 2048,
    system_override: str = "", corrections: list[dict] | None = None,
) -> list[Transaction]:
    system = system_override if system_override else (SYSTEM_PROMPT + extra_hint)
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
    return parse_response(raw)


def parse_response(raw: str) -> list[Transaction]:
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
        txn_date = norm_date(item.get("date", ""))
        desc = str(item.get("description", "")).strip()
        amount = norm_amount(item.get("amount", ""))
        if txn_date and desc and amount and not _CREDIT_RE.search(desc):
            result.append(Transaction(date=txn_date, description=desc, amount=amount))
    return result


def norm_date(value: str) -> str:
    v = str(value).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", v):
        return v
    m = re.match(r"^(\d{2})/(\d{2})/(\d{2,4})$", v)
    if m:
        d, mo, y = m.groups()
        return f"{'20'+y if len(y)==2 else y}-{mo}-{d}"
    return ""


def norm_amount(value) -> str:
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


async def extract_generic(pdf_bytes: bytes, bank: str, corrections: list[dict]) -> tuple[list[Transaction], str]:
    """Fallback for unrecognized banks: full-text LLM extraction, no pre-processing."""
    text = plain_text(pdf_bytes)
    txns = await call_llm(text, bank, corrections=corrections)
    return txns, text
