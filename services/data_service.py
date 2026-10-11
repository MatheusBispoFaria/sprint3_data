from __future__ import annotations

import math
from functools import lru_cache
from statistics import NormalDist

import pandas as pd

from analysis.core import load_dictionary, load_or_analyze_dataframe, load_source_data
from analysis.ml_pipeline import train_models
from analysis.sprint4_analysis import (
    build_sprint4_analysis,
    load_validation_set,
    select_final_model,
    validate_final_model,
)

RISK_THRESHOLD = 10
TEST_SIZE = 0.20


@lru_cache(maxsize=1)
def get_dictionary():
    return load_dictionary()


@lru_cache(maxsize=1)
def get_dataframe():
    base, warnings = load_source_data()
    df = load_or_analyze_dataframe(base, get_dictionary())
    return df, warnings


@lru_cache(maxsize=1)
def get_ml_bundle():
    df, _ = get_dataframe()
    try:
        return train_models(df, threshold=RISK_THRESHOLD, test_size=TEST_SIZE), None
    except Exception as exc:
        return None, str(exc)


def analyze_transcript(text: str) -> dict:
    bundle, ml_error = get_ml_bundle()
    result = build_sprint4_analysis(text, get_dictionary(), bundle)
    return {
        "classe_final": result.final_class,
        "risco_regras": result.rule_risk,
        "risco_ml": result.ml_risk,
        "risco_final": result.final_risk,
        "oportunidade": result.opportunity,
        "sentimento": result.sentiment_label,
        "sentimento_score": result.sentiment_score,
        "confianca_scoring": result.confidence,
        "modelo": result.final_model,
        "produtos_totvs": result.products,
        "concorrentes": result.competitors,
        "termos_tfidf": [{"termo": t, "peso": w} for t, w in result.top_terms],
        "padroes": result.patterns,
        "recomendacao": result.recommendation,
        "frequencias": result.rule_result.get("freq", {}),
        "n_palavras": result.rule_result.get("n_palavras", 0),
        "locutores": result.rule_result.get("locutores", 0),
        "ml_error": ml_error,
    }


def mean_confidence_interval(values, confidence: float = 0.95) -> tuple[float, float, float, int]:
    """IC aproximado da média usando erro-padrão e quantil normal."""
    s = pd.Series(values, dtype="float64").dropna()
    n = len(s)
    if n == 0:
        return math.nan, math.nan, math.nan, 0
    mean = float(s.mean())
    if n < 2:
        return mean, mean, mean, n
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    se = float(s.std(ddof=1) / math.sqrt(n))
    margin = z * se
    return mean, mean - margin, mean + margin, n


def group_mean_ci(df: pd.DataFrame, group_col: str, value_col: str, min_n: int = 2) -> pd.DataFrame:
    rows = []
    for group, part in df.groupby(group_col, dropna=True):
        mean, low, high, n = mean_confidence_interval(part[value_col])
        if n >= min_n:
            rows.append({group_col: str(group), "media": mean, "ci_low": low, "ci_high": high, "n": n})
    return pd.DataFrame(rows)


@lru_cache(maxsize=1)
def get_model_summary() -> dict:
    bundle, error = get_ml_bundle()
    if bundle is None:
        return {
            "modelo": None,
            "justificativa": error or "Modelo indisponível",
            "metricas": [],
            "matrizes": {},
            "class_counts": {},
            "train_shape": None,
        }

    model, justification = select_final_model(bundle)
    metrics = bundle.metrics.reset_index().to_dict("records")
    matrices = {
        name: info["cm"].astype(int).tolist()
        for name, info in bundle.predictions.items()
    }
    return {
        "modelo": model,
        "justificativa": justification,
        "metricas": metrics,
        "matrizes": matrices,
        "class_counts": bundle.class_counts,
        "train_shape": list(bundle.train_shape),
        "limiar_risco": RISK_THRESHOLD,
        "test_size": TEST_SIZE,
    }


@lru_cache(maxsize=1)
def get_validation_summary() -> dict:
    bundle, error = get_ml_bundle()
    validation = load_validation_set()
    if bundle is None:
        return {
            "metricas": {},
            "resultados": validation.to_dict("records"),
            "erro": error or "Modelo indisponível",
        }
    result, metrics = validate_final_model(bundle, validation, get_dictionary())
    return {
        "metricas": metrics,
        "resultados": result.to_dict("records"),
        "erro": None,
    }


def get_scoring_description() -> list[dict]:
    return [
        {
            "indicador": "Risco por regras",
            "escala": "0–100",
            "origem": "Motor detalhado do totvs_datascience",
            "descricao": "Densidade de termos de churn, concorrência e gatilhos financeiros.",
        },
        {
            "indicador": "Probabilidade ML",
            "escala": "0–100%",
            "origem": "Modelo final da Sprint 3",
            "descricao": "Probabilidade de risco prevista pelo classificador supervisionado selecionado.",
        },
        {
            "indicador": "Risco final",
            "escala": "0–100",
            "origem": "Scoring integrado Sprint 4",
            "descricao": "45% do risco por regras + 55% da probabilidade do modelo final.",
        },
        {
            "indicador": "Oportunidade",
            "escala": "0–100",
            "origem": "Motor detalhado do totvs_datascience",
            "descricao": "Sinais de upsell, produtos TOTVS e gatilhos financeiros.",
        },
        {
            "indicador": "Sentimento",
            "escala": "Positivo / Neutro / Negativo",
            "origem": "Heurística transparente",
            "descricao": "Sinal complementar baseado em vocabulário positivo/negativo e negações.",
        },
        {
            "indicador": "Confiança do scoring",
            "escala": "0–100",
            "origem": "Índice operacional",
            "descricao": "Combina concordância regra/ML e quantidade de evidências. Não é probabilidade calibrada.",
        },
    ]
