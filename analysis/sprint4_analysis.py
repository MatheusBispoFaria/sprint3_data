from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from analysis.core import analyze_text, clean_text
from analysis.ml_pipeline import MLBundle, predict_text

BASE_DIR = Path(__file__).resolve().parents[1]
VALIDATION_PATH = BASE_DIR / "data" / "reference" / "validation_sprint4.csv"
WORD_RE = re.compile(r"\b[\wÀ-ÿ]+\b", flags=re.UNICODE)
NEGATIONS = {"não", "nunca", "jamais", "nem", "sem"}

POSITIVE_WORDS = {
    "bom", "boa", "ótimo", "ótima", "excelente", "satisfeito", "satisfeita", "satisfação",
    "resolveu", "resolvido", "funcionando", "estável", "melhorou", "melhoria", "rápido", "rápida",
    "eficiente", "sucesso", "aprovado", "aprovada", "gostou", "gostamos", "confiável", "tranquilo",
    "positivo", "positiva", "benefício", "beneficios", "ganho", "ganhos", "aderente", "interesse",
    "interessado", "interessada", "oportunidade", "expansão", "expandir", "comprar", "adquirir",
    "upgrade", "integração", "treinamento", "consultoria", "parceria", "renovar", "renovação",
}

NEGATIVE_WORDS = {
    "ruim", "péssimo", "péssima", "insatisfeito", "insatisfeita", "reclamação", "reclamou", "problema",
    "problemas", "erro", "erros", "bug", "bugs", "instável", "instabilidade", "travado", "travou",
    "parado", "parou", "indisponível", "lento", "lenta", "demora", "atraso", "cancelar", "cancelamento",
    "churn", "caro", "encareceu", "multa", "jurídico", "procon", "concorrente", "frustração", "frustrado",
    "frustrada", "falha", "falhas", "crítico", "crítica", "urgente", "risco", "renegociar", "reajuste",
    "fora", "queda", "quedas", "inseguro", "insegura", "preocupado", "preocupada", "perda", "perder",
}


@dataclass
class Sprint4Result:
    rule_risk: float
    ml_risk: float | None
    final_risk: float
    final_class: str
    opportunity: float
    sentiment_label: str
    sentiment_score: float
    confidence: float
    final_model: str | None
    products: list[str]
    competitors: list[str]
    top_terms: list[tuple[str, float]]
    patterns: list[str]
    recommendation: str
    rule_result: dict


def analyze_sentiment(text: str) -> dict:
    """Sinal de sentimento transparente, complementar ao modelo de churn."""
    tokens = [t.lower() for t in WORD_RE.findall(clean_text(text))]
    positive_hits: list[str] = []
    negative_hits: list[str] = []
    score = 0

    for i, token in enumerate(tokens):
        window = tokens[max(0, i - 3):i]
        negated = any(w in NEGATIONS for w in window)
        if token in POSITIVE_WORDS:
            if negated:
                score -= 1
                negative_hits.append(f"não {token}")
            else:
                score += 1
                positive_hits.append(token)
        elif token in NEGATIVE_WORDS:
            if negated:
                score += 1
                positive_hits.append(f"não {token}")
            else:
                score -= 1
                negative_hits.append(token)

    evidence = len(positive_hits) + len(negative_hits)
    normalized = 0.0 if evidence == 0 else max(-100.0, min(100.0, score / max(1, evidence) * 100.0))
    if evidence == 0 or abs(normalized) < 20:
        label = "Neutro"
    elif normalized > 0:
        label = "Positivo"
    else:
        label = "Negativo"

    return {
        "label": label,
        "score": round(normalized, 1),
        "positive_hits": positive_hits,
        "negative_hits": negative_hits,
        "evidence": evidence,
    }


def select_final_model(bundle: MLBundle | None) -> tuple[str | None, str]:
    if bundle is None or bundle.metrics.empty:
        return None, "Modelo supervisionado indisponível; o sistema mantém o motor de regras como fallback."
    best_model = bundle.metrics["F1 (Risco)"].idxmax()
    row = bundle.metrics.loc[best_model]
    justification = (
        f"{best_model} foi selecionado como modelo final por apresentar o maior F1 para a classe de risco "
        f"({row['F1 (Risco)'] * 100:.1f}%). O F1 equilibra Precision ({row['Precision (Risco)'] * 100:.1f}%) "
        f"e Recall ({row['Recall (Risco)'] * 100:.1f}%). Para churn, o Recall também é acompanhado porque "
        "um falso negativo pode ocultar um cliente que necessita de ação de retenção."
    )
    return best_model, justification


def extract_top_terms(bundle: MLBundle | None, text: str, top_n: int = 10) -> list[tuple[str, float]]:
    if bundle is None or not text.strip():
        return []
    matrix = bundle.vectorizer.transform([clean_text(text)])
    if matrix.nnz == 0:
        return []
    names = np.asarray(bundle.vectorizer.get_feature_names_out())
    row = matrix.tocoo()
    ranked = sorted(zip(names[row.col], row.data), key=lambda x: x[1], reverse=True)[:top_n]
    return [(str(term), round(float(weight), 4)) for term, weight in ranked]


def _confidence(rule_risk: float, ml_risk: float | None, n_words: int, evidence_count: int) -> float:
    """Índice operacional de confiança; não representa probabilidade estatística calibrada."""
    if ml_risk is None:
        evidence = min(100.0, evidence_count * 10 + min(n_words, 800) / 800 * 35)
        return round(max(25.0, min(78.0, 30 + evidence * 0.48)), 1)
    agreement = max(0.0, 100.0 - abs(rule_risk - ml_risk))
    evidence = min(100.0, evidence_count * 10 + min(n_words, 800) / 800 * 35)
    return round(max(20.0, min(100.0, agreement * 0.68 + evidence * 0.32)), 1)


def _recommendation(
    final_risk: float,
    opportunity: float,
    sentiment_label: str,
    products: Iterable[str],
    competitors: Iterable[str],
) -> str:
    products = sorted({p for p in products if p})
    competitors = sorted({c for c in competitors if c})
    product_text = ", ".join(products) if products else "o portfólio adequado"
    competitor_text = ", ".join(competitors)

    if final_risk >= 60:
        extra = f" Há menção a {competitor_text}." if competitors else ""
        return (
            "Prioridade: retenção. Abrir plano de ação com responsável e prazo, validar a causa do risco e realizar follow-up executivo."
            + extra
            + " Evitar uma abordagem de expansão antes de estabilizar a relação."
        )
    if final_risk >= 35 and opportunity >= 45:
        return (
            "A reunião combina risco e potencial comercial. Tratar primeiro as objeções e pontos de atrito e, somente depois, "
            f"validar uma proposta relacionada a {product_text}."
        )
    if sentiment_label == "Negativo" and final_risk >= 25:
        return (
            "Há sinal negativo relevante. Recomenda-se contato consultivo, registro dos problemas citados e confirmação de uma data "
            "para retorno antes de avançar comercialmente."
        )
    if opportunity >= 60:
        return f"Prioridade: expansão. Confirmar decisores, escopo, orçamento e prazo e aprofundar a aderência de {product_text}."
    if opportunity >= 30:
        return "Existem indícios de oportunidade. Fazer follow-up para validar interesse, prioridade, orçamento e prazo."
    return (
        "Manter acompanhamento consultivo. Não há evidência suficiente para uma ação imediata de retenção ou expansão; "
        "registrar a reunião e acompanhar a evolução."
    )


def build_sprint4_analysis(text: str, dictionary: dict[str, list[str]], bundle: MLBundle | None) -> Sprint4Result:
    # Motor detalhado reaproveitado do projeto totvs_datascience.
    rule = analyze_text(text, dictionary)
    rule_risk = float(rule["idx_risco"])
    opportunity = float(rule["idx_oportunidade"])
    sentiment = analyze_sentiment(text)
    model_name, _ = select_final_model(bundle)

    ml_risk = None
    if bundle is not None and model_name:
        prediction = predict_text(bundle, text, model_name)
        ml_risk = float(prediction["probabilidade_risco"] * 100)

    # Scoring de integração da Sprint 4: o modelo supervisionado tem peso maior,
    # mas o motor de regras continua contribuindo com sinais de negócio explícitos.
    final_risk = rule_risk if ml_risk is None else round(rule_risk * 0.45 + ml_risk * 0.55, 1)
    final_class = "Risco" if final_risk >= 50 else "Neutro"
    products = sorted(set(rule["hits"].get("Produtos TOTVS", [])))
    competitors = sorted(set(rule["hits"].get("Concorrência", [])))
    evidence_count = sum(rule["freq"].values()) + int(sentiment["evidence"])
    confidence = _confidence(rule_risk, ml_risk, int(rule["n_palavras"]), evidence_count)
    top_terms = extract_top_terms(bundle, text, top_n=10)
    patterns = [label for _, label in rule["padroes"]]

    return Sprint4Result(
        rule_risk=round(rule_risk, 1),
        ml_risk=None if ml_risk is None else round(ml_risk, 1),
        final_risk=round(final_risk, 1),
        final_class=final_class,
        opportunity=round(opportunity, 1),
        sentiment_label=sentiment["label"],
        sentiment_score=float(sentiment["score"]),
        confidence=confidence,
        final_model=model_name,
        products=products,
        competitors=competitors,
        top_terms=top_terms,
        patterns=patterns,
        recommendation=_recommendation(final_risk, opportunity, sentiment["label"], products, competitors),
        rule_result=rule,
    )


def load_validation_set(path: Path = VALIDATION_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["id", "cenario", "classe_esperada", "texto"])
    return pd.read_csv(path, encoding="utf-8")


def validate_final_model(
    bundle: MLBundle | None,
    validation_df: pd.DataFrame,
    dictionary: dict[str, list[str]] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Valida o pipeline integrado em transcrições inéditas, separadas do treino."""
    if bundle is None or validation_df.empty:
        return validation_df.copy(), {}

    model_name, _ = select_final_model(bundle)
    rows = []
    for row in validation_df.itertuples(index=False):
        expected = int(row.classe_esperada)
        if dictionary is not None:
            analysis = build_sprint4_analysis(str(row.texto), dictionary, bundle)
            predicted = 1 if analysis.final_class == "Risco" else 0
            ml_probability = analysis.ml_risk if analysis.ml_risk is not None else 0.0
            final_score = analysis.final_risk
        else:
            prediction = predict_text(bundle, str(row.texto), model_name)
            predicted = 1 if prediction["classe"] == "Risco" else 0
            ml_probability = prediction["probabilidade_risco"] * 100
            final_score = ml_probability

        rows.append(
            {
                "id": getattr(row, "id", ""),
                "cenario": getattr(row, "cenario", ""),
                "classe_esperada": "Risco" if expected == 1 else "Neutro",
                "classe_prevista": "Risco" if predicted == 1 else "Neutro",
                "score_final_risco": round(float(final_score), 1),
                "probabilidade_ml": round(float(ml_probability), 1),
                "acerto": "Sim" if expected == predicted else "Não",
                "_expected": expected,
                "_predicted": predicted,
            }
        )

    result = pd.DataFrame(rows)
    p, r, f1, _ = precision_recall_fscore_support(
        result["_expected"], result["_predicted"], labels=[0, 1], average=None, zero_division=0
    )
    metrics = {
        "accuracy": float(accuracy_score(result["_expected"], result["_predicted"])),
        "precision_risk": float(p[1]),
        "recall_risk": float(r[1]),
        "f1_risk": float(f1[1]),
        "n": int(len(result)),
        "model": model_name,
    }
    return result.drop(columns=["_expected", "_predicted"]), metrics
