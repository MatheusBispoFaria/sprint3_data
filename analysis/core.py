from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "reference"
JSON_PATH = DATA_DIR / "ANON_transcricao.json"
CSV_PATH = DATA_DIR / "reunioes_transcricoes_mockado.csv"
DICTIONARY_PATH = DATA_DIR / "dicionario_termos.json"
ANALYSIS_CACHE_PATH = DATA_DIR / "analysis_features.csv"

NEGACOES = {"não", "nunca", "jamais", "nem", "tampouco", "nenhum", "nada"}
STOP_WORDS_PT = {
    "o", "a", "os", "as", "de", "do", "da", "dos", "das", "em", "um", "uma",
    "com", "para", "que", "ao", "se", "mas", "eles", "esta", "está", "na", "no",
    "nas", "nos", "por", "como", "não", "é", "são", "foi", "ser", "ter", "já",
    "eu", "você", "ele", "ela", "nós", "isso", "este", "esse", "aquele", "muito",
    "mais", "menos", "até", "então", "quando", "onde", "pois", "porque", "assim",
    "bem", "só", "aqui", "lá", "vai", "ir", "haver", "tá", "né", "aí", "pra",
    "e", "ou", "sua", "seu", "minha", "meu", "também", "depois", "antes", "sobre",
    "entre", "todos", "toda", "todo", "tudo", "nada",
}

WORD_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)
LOCUTOR_RE = re.compile(r"\[LOCUTOR\s*(\d+)\]", flags=re.IGNORECASE)


def load_dictionary(path: Path = DICTIONARY_PATH) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_dictionary(dictionary: dict[str, list[str]], path: Path = DICTIONARY_PATH) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(dictionary, f, ensure_ascii=False, indent=4)


def compile_dictionary(dictionary: dict[str, list[str]]) -> dict[str, re.Pattern]:
    patterns: dict[str, re.Pattern] = {}
    for category, terms in dictionary.items():
        escaped = sorted((re.escape(str(t).lower()) for t in terms), key=len, reverse=True)
        if escaped:
            patterns[category] = re.compile(r"\b(?:" + "|".join(escaped) + r")\b", re.IGNORECASE)
    return patterns


def load_source_data(csv_path: Path = CSV_PATH, json_path: Path = JSON_PATH) -> tuple[pd.DataFrame, list[str]]:
    """Merge Sprint 1/2 CSV metadata with real JSON transcripts and consolidate one row per meeting."""
    warnings: list[str] = []
    if not csv_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {json_path}")

    df_csv = pd.read_csv(csv_path, encoding="utf-8")
    df_csv["ID_MEETING"] = df_csv["ID_MEETING"].astype(str).str.strip()
    n_before = len(df_csv)
    df_csv = df_csv.drop_duplicates(subset=["ID_MEETING"], keep="first")
    if len(df_csv) != n_before:
        warnings.append(f"CSV: {n_before - len(df_csv)} IDs duplicados removidos.")

    rows = []
    with json_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                warnings.append(f"JSON: linha {line_no} inválida e ignorada.")

    df_json = pd.DataFrame(rows)
    if not df_json.empty:
        df_json["ID_MEETING"] = df_json["ID_MEETING"].astype(str).str.strip()
        if "ANON_TRANSCRICAO" in df_json.columns:
            df_json["_hash"] = df_json["ANON_TRANSCRICAO"].astype(str).map(
                lambda t: hashlib.md5(t.encode("utf-8")).hexdigest()
            )
            n_before = len(df_json)
            df_json = df_json.drop_duplicates(subset=["ID_MEETING", "_hash"], keep="first").drop(columns=["_hash"])
            if len(df_json) != n_before:
                warnings.append(f"JSON: {n_before - len(df_json)} duplicatas exatas removidas.")

    ids_csv = set(df_csv["ID_MEETING"])
    df_json_extra = df_json[~df_json["ID_MEETING"].isin(ids_csv)].copy() if not df_json.empty else pd.DataFrame()
    if not df_json_extra.empty and "FAIXA_FATURAMENTO_CLIENTE_EC" in df_json_extra.columns:
        df_json_extra = df_json_extra.rename(columns={"FAIXA_FATURAMENTO_CLIENTE_EC": "FAIXA_FATURAMENTO_CLIENTE"})

    df = pd.concat([df_csv, df_json_extra], ignore_index=True, sort=False)

    if "ANON_TRANSCRICAO" in df.columns:
        meta_cols = [c for c in df.columns if c != "ANON_TRANSCRICAO"]
        meta = df[meta_cols].drop_duplicates(subset=["ID_MEETING"], keep="first")
        text = (
            df.groupby("ID_MEETING", sort=False)["ANON_TRANSCRICAO"]
            .apply(lambda s: "\n".join(s.dropna().astype(str).unique()))
            .reset_index()
        )
        df = meta.merge(text, on="ID_MEETING", how="inner")

    def duration_to_minutes(value):
        try:
            parts = str(value).strip().split(":")
            if len(parts) == 3:
                return int(parts[0]) * 60 + int(parts[1]) + int(parts[2]) / 60
            if len(parts) == 2:
                return int(parts[0]) + int(parts[1]) / 60
        except (TypeError, ValueError):
            pass
        return None

    if "DURACAO_MEETING" in df.columns:
        df["duracao_min"] = df["DURACAO_MEETING"].map(duration_to_minutes)
    if "DT_MEETING" in df.columns:
        df["DT_MEETING"] = pd.to_datetime(df["DT_MEETING"], errors="coerce")
        df["mes_meeting"] = df["DT_MEETING"].dt.to_period("M").astype(str)

    df = df.rename(columns={
        "ANON_TRANSCRICAO": "texto",
        "NOME_UNIDADE": "cliente",
        "ID_MEETING": "id_reuniao",
        "NOTA_NPS": "nps",
        "TP_RECURSO": "tipo_recurso",
        "STATUS_MEETING": "status",
        "FORMATO_MEETING": "formato",
        "NOME_SEGMENTO": "segmento",
        "FAIXA_FATURAMENTO_CLIENTE": "faixa_fat",
        "UF": "uf",
        "CNAE": "cnae",
    })
    for col in ["nps", "duracao_min"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.reset_index(drop=True), warnings


def _score_norm(count: int, weight: float, n_words: int) -> float:
    density = (count / max(100, n_words)) * 1000
    return min(100.0, density * weight)


def analyze_text(text: str, dictionary: dict[str, list[str]], patterns: dict[str, re.Pattern] | None = None) -> dict:
    """Canonical Sprint 1/2 rule engine used by the consolidated app and Sprint 3 labels."""
    text = text if isinstance(text, str) else ""
    lower = text.lower()
    patterns = patterns or compile_dictionary(dictionary)

    freq = {category: 0 for category in dictionary}
    hits = {category: [] for category in dictionary}
    for category, pattern in patterns.items():
        for match in pattern.finditer(lower):
            prior = lower[max(0, match.start() - 50):match.start()]
            prior_words = WORD_RE.findall(prior)[-4:]
            if NEGACOES.intersection(prior_words):
                continue
            freq[category] += 1
            hits[category].append(match.group(0).lower())

    actual_words = len(WORD_RE.findall(lower))
    n_words = max(100, actual_words)
    opportunity = round(min(
        100.0,
        _score_norm(freq.get("Upsell", 0), 18, n_words)
        + _score_norm(freq.get("Produtos TOTVS", 0), 8, n_words)
        + _score_norm(freq.get("Gatilhos Financeiros", 0), 4, n_words),
    ), 1)
    # Standardized to the rule used in Sprints 1/2 (the Sprint 3 legacy script had different weights).
    risk = round(min(
        100.0,
        _score_norm(freq.get("Risco de Churn", 0), 22, n_words)
        + _score_norm(freq.get("Concorrência", 0), 18, n_words)
        + _score_norm(freq.get("Gatilhos Financeiros", 0), 5, n_words),
    ), 1)

    patterns_found: list[tuple[str, str]] = []
    if freq.get("Pessoas e Áreas", 0) > 0 and (freq.get("Risco de Churn", 0) > 0 or freq.get("Concorrência", 0) > 0):
        patterns_found.append(("risk", "Risco Crítico c/ Stakeholder"))
    if freq.get("Produtos TOTVS", 0) > 0 and freq.get("Upsell", 0) > 0:
        patterns_found.append(("opp", "Oportunidade Cross-Sell"))
    if freq.get("Gatilhos Financeiros", 0) > 0 and freq.get("Concorrência", 0) > 0:
        patterns_found.append(("warn", "Disputa Comercial por Preço"))
    if freq.get("Upsell", 0) > 0 and freq.get("Risco de Churn", 0) == 0 and freq.get("Concorrência", 0) == 0:
        patterns_found.append(("opp", "Pipeline de Expansão Limpo"))
    if freq.get("Risco de Churn", 0) > 2:
        patterns_found.append(("risk", "Churn Iminente"))
    if not patterns_found:
        patterns_found.append(("neutral", "Padrão Operacional"))

    return {
        "freq": freq,
        "hits": hits,
        "idx_oportunidade": opportunity,
        "idx_risco": risk,
        "padroes": patterns_found,
        "locutores": len(set(LOCUTOR_RE.findall(text))),
        "n_palavras": actual_words,
    }


def analyze_dataframe(df: pd.DataFrame, dictionary: dict[str, list[str]]) -> pd.DataFrame:
    compiled = compile_dictionary(dictionary)
    rows = []
    for row in df.itertuples(index=False):
        result = analyze_text(getattr(row, "texto", ""), dictionary, compiled)
        flat = {
            "id_reuniao": getattr(row, "id_reuniao"),
            "idx_oportunidade": result["idx_oportunidade"],
            "idx_risco": result["idx_risco"],
            "padroes": result["padroes"],
            "locutores": result["locutores"],
            "n_palavras": result["n_palavras"],
        }
        for category, count in result["freq"].items():
            flat[f"freq_{category}"] = count
        for category, values in result["hits"].items():
            flat[f"hits_{category}"] = ", ".join(sorted(set(values)))
        rows.append(flat)
    features = pd.DataFrame(rows)
    return df.merge(features, on="id_reuniao", how="inner")



def load_or_analyze_dataframe(
    df: pd.DataFrame,
    dictionary: dict[str, list[str]],
    cache_path: Path = ANALYSIS_CACHE_PATH,
) -> pd.DataFrame:
    """Load precomputed rule-engine features when available; otherwise compute and persist them."""
    if cache_path.exists():
        try:
            cache = pd.read_csv(cache_path, encoding="utf-8")
            cache["id_reuniao"] = cache["id_reuniao"].astype(str)
            expected_ids = set(df["id_reuniao"].astype(str))
            cache_ids = set(cache["id_reuniao"])
            if expected_ids == cache_ids:
                if "padroes" in cache.columns:
                    cache["padroes"] = cache["padroes"].map(
                        lambda v: json.loads(v) if isinstance(v, str) and v.strip() else []
                    )
                return df.merge(cache, on="id_reuniao", how="inner")
        except Exception:
            pass

    analyzed = analyze_dataframe(df, dictionary)
    feature_cols = [
        c for c in analyzed.columns
        if c == "id_reuniao"
        or c in {"idx_oportunidade", "idx_risco", "padroes", "locutores", "n_palavras"}
        or c.startswith("freq_")
        or c.startswith("hits_")
    ]
    cache = analyzed[feature_cols].copy()
    if "padroes" in cache.columns:
        cache["padroes"] = cache["padroes"].map(lambda v: json.dumps(v, ensure_ascii=False))
    cache.to_csv(cache_path, index=False, encoding="utf-8")
    return analyzed

def clean_text(text: str) -> str:
    value = re.sub(r"\[LOCUTOR\s*\d+\]\s*:", " ", str(text), flags=re.IGNORECASE)
    value = re.sub(r"\[(PESSOA|EMPRESA|LOCAL)\]", " ", value, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", value).strip()


def suggest_tfidf_terms(texts: Iterable[str], dictionary: dict[str, list[str]], top_n: int = 15) -> pd.DataFrame:
    corpus = [str(t) for t in texts if str(t).strip()]
    if not corpus:
        return pd.DataFrame(columns=["termo", "score_tfidf"])
    known = {t.lower() for values in dictionary.values() for t in values}
    vec = TfidfVectorizer(
        stop_words=list(STOP_WORDS_PT), ngram_range=(1, 2), min_df=3, max_features=5000
    )
    matrix = vec.fit_transform(corpus)
    terms = vec.get_feature_names_out()
    scores = matrix.sum(axis=0).A1
    out = pd.DataFrame({"termo": terms, "score_tfidf": scores}).sort_values("score_tfidf", ascending=False)
    mask = (~out["termo"].isin(known)) & out["termo"].str.match(r"^[a-záéíóúâêîôûãõàèìòùç ]+$")
    return out[mask].head(top_n).reset_index(drop=True)
