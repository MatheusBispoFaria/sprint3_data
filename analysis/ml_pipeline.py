from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from analysis.core import STOP_WORDS_PT, clean_text

RANDOM_STATE = 42
MAX_CHARS_PER_TRANSCRIPT = 10000


@dataclass
class MLBundle:
    dataframe: pd.DataFrame
    vectorizer: TfidfVectorizer
    models: dict
    test_index: pd.Index
    y_test: pd.Series
    predictions: dict
    metrics: pd.DataFrame
    train_shape: tuple[int, int]
    class_counts: dict[int, int]


def _score_norm(count: int, weight: float, n_words: int) -> float:
    density = (count / max(100, int(n_words))) * 1000
    return min(100.0, density * weight)


def sprint3_risk_score(row: pd.Series | dict) -> float:
    """Reproduz a variável-alvo usada no app original da Sprint 3."""
    getter = row.get if hasattr(row, "get") else lambda key, default=0: default
    n_words = int(getter("n_palavras", 100) or 100)
    churn = int(getter("freq_Risco de Churn", 0) or 0)
    competition = int(getter("freq_Concorrência", 0) or 0)
    return round(
        min(
            100.0,
            _score_norm(churn, 20, n_words)
            + _score_norm(competition, 10, n_words),
        ),
        1,
    )


def prepare_ml_text(text: str, max_chars: int = MAX_CHARS_PER_TRANSCRIPT) -> str:
    """Limpa o texto e reduz reuniões muito longas preservando início/meio/fim."""
    value = clean_text(text)
    if len(value) <= max_chars:
        return value
    head = int(max_chars * 0.45)
    middle = int(max_chars * 0.20)
    tail = max_chars - head - middle
    mid_start = max(0, len(value) // 2 - middle // 2)
    return f"{value[:head]} {value[mid_start:mid_start + middle]} {value[-tail:]}"


def train_models(df_analyzed: pd.DataFrame, threshold: int = 10, test_size: float = 0.20) -> MLBundle:
    """Treina os dois modelos utilizados na Sprint 3 sobre o mesmo alvo de churn."""
    df = df_analyzed.copy()
    df["idx_risco_sprint3"] = df.apply(sprint3_risk_score, axis=1)
    df["risco"] = (df["idx_risco_sprint3"] >= threshold).astype(int)

    counts = df["risco"].value_counts()
    if len(counts) < 2 or counts.min() < 3:
        raise ValueError(
            "O limiar escolhido não produz duas classes com exemplos suficientes para treino e teste."
        )

    df["texto_ml"] = df["texto"].map(prepare_ml_text)
    X_train, X_test, y_train, y_test = train_test_split(
        df["texto_ml"],
        df["risco"],
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["risco"],
    )

    # Mesma configuração conceitual do app original da Sprint 3.
    vectorizer = TfidfVectorizer(
        stop_words=list(STOP_WORDS_PT),
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.90,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    models = {
        "Regressão Logística": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced",
            n_estimators=300,
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    predictions: dict[str, dict] = {}
    metric_rows = []
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        pred = model.predict(X_test_tfidf)
        proba = model.predict_proba(X_test_tfidf)[:, 1]
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, pred, average=None, labels=[0, 1], zero_division=0
        )
        auc = roc_auc_score(y_test, proba) if y_test.nunique() == 2 else np.nan
        cm = confusion_matrix(y_test, pred, labels=[0, 1])
        predictions[name] = {"y_pred": pred, "y_proba": proba, "cm": cm}
        metric_rows.append(
            {
                "Modelo": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision (Risco)": p[1],
                "Recall (Risco)": r[1],
                "F1 (Risco)": f1[1],
                "ROC-AUC": auc,
            }
        )

    metrics = pd.DataFrame(metric_rows).set_index("Modelo")
    return MLBundle(
        dataframe=df,
        vectorizer=vectorizer,
        models=models,
        test_index=X_test.index,
        y_test=y_test,
        predictions=predictions,
        metrics=metrics,
        train_shape=X_train_tfidf.shape,
        class_counts={int(k): int(v) for k, v in counts.items()},
    )


def predict_text(bundle: MLBundle, text: str, model_name: str) -> dict:
    if model_name not in bundle.models:
        raise KeyError(model_name)
    matrix = bundle.vectorizer.transform([prepare_ml_text(text)])
    model = bundle.models[model_name]
    pred = int(model.predict(matrix)[0])
    proba = float(model.predict_proba(matrix)[0, 1])
    return {
        "classe": "Risco" if pred == 1 else "Neutro",
        "probabilidade_risco": proba,
    }
