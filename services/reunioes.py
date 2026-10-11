"""Orquestração: motor analítico existente + persistência."""
from __future__ import annotations
from services.data_service import analyze_transcript
from storage.database import salvar_reuniao


def analisar_e_salvar(texto: str, *, cliente: str = "", assunto: str = "",
                      data_reuniao: str | None = None, origem: str = "dashboard") -> dict:
    texto = (texto or "").strip()
    if not texto:
        raise ValueError("Informe uma transcrição.")
    if len(texto) > 200_000:
        raise ValueError("A transcrição excede o limite de 200.000 caracteres.")
    analise = analyze_transcript(texto)
    return salvar_reuniao(texto=texto, analise=analise, cliente=cliente,
                          assunto=assunto, data_reuniao=data_reuniao, origem=origem)
