"""Bank detection and extraction dispatch."""
from . import bradesco, itau, mercadopago, nubank
from .base import Transaction, extract_generic, plain_text

__all__ = ["Transaction", "detect_bank", "extract"]


def detect_bank(pdf_bytes: bytes) -> str:
    text = plain_text(pdf_bytes).lower()
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


async def extract(pdf_bytes: bytes, bank: str, corrections: list[dict]) -> tuple[list[Transaction], str]:
    if bank == "Itaú":
        return await itau.extract(pdf_bytes, corrections)
    if bank == "Nubank":
        return await nubank.extract(pdf_bytes, corrections)
    if bank == "Bradesco":
        return await bradesco.extract(pdf_bytes, corrections)
    if bank == "Mercado Pago":
        return await mercadopago.extract(pdf_bytes, corrections)
    return await extract_generic(pdf_bytes, bank, corrections)
