"""
Bradesco extrato extraction: dates, descriptions and amounts are spread across separate,
ambiguous lines in the raw PDF text, so we rule-based pre-process into clean
'YYYY-MM-DD DESCRIPTION AMOUNT' lines before calling the LLM (which just does a strict
pass-through conversion to JSON — no interpretation needed).
"""
import re

from .base import Transaction, call_llm, plain_text

SYSTEM_PROMPT = """\
Convert each input line to a JSON object. Every line is a confirmed debit transaction — do NOT skip, filter, or deduplicate any of them.

Each line format: YYYY-MM-DD DESCRIPTION AMOUNT

- date: copy exactly as YYYY-MM-DD
- description: the text between the date and the last number on the line
- amount: the last number on the line, already in decimal format (e.g. 186.69), copy as a string with 2 decimal places

Output ONLY a valid JSON array:
[{"date":"2026-05-11","description":"PIX ENVIADO","amount":"186.69"},...]

Include ALL lines without exception."""

_SKIP_RE = re.compile(
    r"TED-TRANSF ELET DISPON|PIX RECEBIDO|COD\. LANC\. 0|RENTAB\.INVEST",
    re.IGNORECASE,
)
_AMOUNT_RE = re.compile(r"(\d{1,3}(?:\.\d{3})*,\d{2})")
_DATE_PREFIX_RE = re.compile(r"^(\d{2}/\d{2}/(\d{4}))\s*")


async def extract(pdf_bytes: bytes, corrections: list[dict]) -> tuple[list[Transaction], str]:
    text = _clean_lines(pdf_bytes)
    if not text:
        return [], ""
    txns = await call_llm(text, "Bradesco", system_override=SYSTEM_PROMPT, max_tokens=2048, corrections=corrections)
    return txns, text


def _clean_lines(pdf_bytes: bytes) -> str:
    """
    Pre-process Bradesco extrato into unambiguous 'YYYY-MM-DD DESCRIPTION R$AMOUNT' lines.

    Entry structure (order in PDF):
      [date] TYPE_LABEL        ← optional type label (may share line with date)
      [date] DocNum D₁ D₂      ← amounts: second-to-last=debit, last=balance
      DES:/REMET.: name DD/MM  ← recipient name (comes AFTER amounts)

    We buffer each pending entry and only emit when the next entry starts, so we
    can capture the DES: that follows the amounts line.
    """
    full_text = plain_text(pdf_bytes)
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
        dm = _DATE_PREFIX_RE.match(line)
        if dm:
            current_date = dm.group(1)
            line = line[dm.end():].strip()

        if not line:
            continue

        amounts = _AMOUNT_RE.findall(line)

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
                if _SKIP_RE.search(line):
                    p_skip = True
                p_debit = amounts[-2]
                p_date = current_date
            else:
                # Standalone amounts (type was inline with date, or two rows back-to-back)
                flush()
                p_date = current_date
                p_debit = amounts[-2]
                p_skip = bool(_SKIP_RE.search(line))
                has_pending = True
            continue

        # Type label line — start new pending entry
        if re.match(r"^[A-Z\*][A-Z\s\-\.\*\/]+$", line) and not amounts:
            flush()
            p_date = current_date
            p_type = line
            p_skip = bool(_SKIP_RE.search(line))
            has_pending = True

    flush()
    return "\n".join(result)
