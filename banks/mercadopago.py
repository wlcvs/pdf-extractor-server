"""Mercado Pago extraction: only the 'Detalhes de consumo' transaction section is sent to the LLM."""
from .base import Transaction, call_llm, plain_text

PROMPT_HINT = (
    "\n\nMercado Pago transaction lines format:\n"
    "'DD/MM  MERCHANT  R$ 111,23' or 'DD/MM  MERCHANT  Parcela 2 de 3  R$ 111,23'.\n"
    "Skip: 'Pagamento da fatura', 'Total R$' lines."
)


async def extract(pdf_bytes: bytes, corrections: list[dict]) -> tuple[list[Transaction], str]:
    text = _transaction_section(pdf_bytes)
    txns = await call_llm(text, "Mercado Pago", extra_hint=PROMPT_HINT, max_tokens=512, corrections=corrections)
    return txns, text


def _transaction_section(pdf_bytes: bytes) -> str:
    """Extract only lines from 'Data Movimentações' header to 'Total R$'."""
    full_text = plain_text(pdf_bytes)
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
