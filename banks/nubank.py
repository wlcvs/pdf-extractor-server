"""Nubank extraction: separate strategies for credit card (fatura) and current account (extrato) statements."""
import io

import pdfplumber

from .base import Transaction, call_llm, plain_text

CARTAO_HINT = (
    "\n\nNubank credit card (fatura) format:\n"
    "Each transaction line: 'DD MMM •••• NNNN  MERCHANT NAME  R$ 68,59'\n"
    "Portuguese month → number: JAN=01 FEV=02 MAR=03 ABR=04 MAI=05 JUN=06 JUL=07 AGO=08 SET=09 OUT=10 NOV=11 DEZ=12\n"
    "Year is 2026 (from the statement header).\n"
    "Skip: lines starting with 'IOF de', lines with negative amounts (−R$), lines starting with 'Pagamento', totals."
)

EXTRATO_HINT = (
    "\n\nNubank current account (extrato) format:\n"
    "Day headers look like: '01 MAI 2026 Total de saídas - 92,49' — these are NOT transactions.\n"
    "Transaction lines come after a day header and end with a BR amount (e.g. '1.234,56').\n"
    "Skip: 'Saldo inicial', 'Saldo final', 'Rendimento', 'Nu Pagamentos', header/footer lines.\n"
    "Each transaction line: description ending with the amount."
)


async def extract(pdf_bytes: bytes, corrections: list[dict]) -> tuple[list[Transaction], str]:
    if "Movimentações" in plain_text(pdf_bytes):
        return await _extract_extrato(pdf_bytes, corrections)
    return await _extract_cartao(pdf_bytes, corrections)


async def _extract_extrato(pdf_bytes: bytes, corrections: list[dict]) -> tuple[list[Transaction], str]:
    """Extrato has many transactions spread across pages — process page by page to keep each LLM call small and fast."""
    pages_text = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            pages_text.append(page.extract_text() or "")

    all_transactions: list[Transaction] = []
    seen: set[tuple] = set()
    all_text_parts = []

    for page_text in pages_text:
        if not page_text.strip():
            continue
        all_text_parts.append(page_text)
        page_txns = await call_llm(page_text, "Nubank", extra_hint=EXTRATO_HINT, corrections=corrections)
        for t in page_txns:
            key = (t.date, t.description, t.amount)
            if key not in seen:
                seen.add(key)
                all_transactions.append(t)

    return all_transactions, "\n\n---\n\n".join(all_text_parts)


async def _extract_cartao(pdf_bytes: bytes, corrections: list[dict]) -> tuple[list[Transaction], str]:
    """Fatura: transactions appear on pages labelled 'TRANSAÇÕES' — process those individually."""
    all_transactions: list[Transaction] = []
    seen: set[tuple] = set()
    all_text_parts = []

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            if "TRANSAÇÕES" not in text:
                continue
            all_text_parts.append(text)
            page_txns = await call_llm(text, "Nubank", extra_hint=CARTAO_HINT, max_tokens=2048, corrections=corrections)
            for t in page_txns:
                key = (t.date, t.description, t.amount)
                if key not in seen:
                    seen.add(key)
                    all_transactions.append(t)

    return all_transactions, "\n\n---\n\n".join(all_text_parts)
