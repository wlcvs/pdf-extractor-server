"""Itaú fatura extraction: transaction table only (billing slips and installment simulations are skipped)."""
import io

import pdfplumber

from .base import Transaction, call_llm

PROMPT_HINT = (
    "\n\nItaú fatura transaction table (DATA | ESTABELECIMENTO | VALOR EM R$):\n"
    "- First line is the header: DATA  ESTABELECIMENTO  VALOREMR$\n"
    "- Transaction line: DD/MM  CODE  amount  (e.g. '27/03 DISTRIBUIDOR-CTEI03/03 156,68')\n"
    "- Continuation line: merchant name on the next line (e.g. 'MORADIA.FRANCODAROC')\n"
    "- Combine code + continuation as description.\n"
    "- Skip: 'Lançamentosnocartão', 'LTotaldos', totals."
)


async def extract(pdf_bytes: bytes, corrections: list[dict]) -> tuple[list[Transaction], str]:
    text = _transaction_rows(pdf_bytes)
    if not text:
        return [], ""
    txns = await call_llm(text, "Itaú", extra_hint=PROMPT_HINT, max_tokens=512, corrections=corrections)
    return txns, text


def _transaction_rows(pdf_bytes: bytes) -> str:
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
