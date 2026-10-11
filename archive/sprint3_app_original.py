# =============================================================================
# TOTVS · Intelligence Hub — Classificação de Risco de Churn (ML)
# app_ml_sprint3.py  |  Sprint 3 · Data Science and Statistical Computing
# =============================================================================
import json
import re
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, precision_recall_fscore_support,
    accuracy_score, roc_auc_score, roc_curve,
)

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TOTVS · Risco de Churn (ML)",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }
section[data-testid="stSidebar"] { background: #161b22; border-right: 1px solid #21262d; }
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-size: 1.9rem; color: #58a6ff; }
[data-testid="stMetricLabel"] { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: .08em; }
button[data-baseweb="tab"] { font-size: .85rem; font-weight: 600; letter-spacing: .04em; color: #8b949e; }
button[data-baseweb="tab"][aria-selected="true"] { color: #58a6ff; border-bottom-color: #58a6ff !important; }
.streamlit-expanderHeader { font-weight: 600; color: #c9d1d9; }
hr { border-color: #21262d !important; }
.badge-risk { background:#3a1a1a; color:#f85149; border:1px solid #da3633; border-radius:6px; padding:2px 10px; font-size:.78rem; font-weight:600; }
.badge-neutral { background:#1c2128; color:#8b949e; border:1px solid #30363d; border-radius:6px; padding:2px 10px; font-size:.78rem; font-weight:600; }
.badge-ml { background:#241a3a; color:#a371f7; border:1px solid #6e40c9; border-radius:6px; padding:2px 10px; font-size:.78rem; font-weight:600; }
.hero-title { font-size: 2rem; font-weight: 700; color: #e6edf3; margin-bottom: .1rem; }
.hero-sub { color: #8b949e; font-size: .9rem; margin-bottom: 1.5rem; }
.insight-box { background:#161b22; border:1px solid #21262d; border-left:3px solid #58a6ff; border-radius:8px; padding:1rem 1.25rem; margin:.75rem 0; }
.insight-box b { color:#e6edf3; }
</style>
""", unsafe_allow_html=True)

RANDOM_STATE = 42
NEGACOES = {"não", "nunca", "jamais", "nem", "tampouco", "nenhum", "nada"}
STOP_WORDS_PT = {
    'o','a','os','as','de','do','da','dos','das','em','um','uma',
    'com','para','que','ao','se','mas','eles','esta','está','na',
    'no','nas','nos','por','como','não','é','são','foi','ser',
    'ter','já','eu','você','ele','ela','nós','isso','este',
    'esse','aquele','muito','mais','menos','até','então','quando',
    'onde','pois','porque','assim','bem','só','aqui','lá','vai',
    'ir','haver','tá','né','aí','pra','e','ou','sua','seu',
    'minha','meu','também','depois','antes','sobre','entre','todos',
    'toda','todo','tudo','nada'
}

# ─────────────────────────────────────────────────────────────────────────────
# CARGA E PROCESSAMENTO (cacheados)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Carregando transcrições e dicionário de termos...")
def carregar_dados():
    with open("dicionario_termos.json", encoding="utf-8") as f:
        dicionario = json.load(f)

    registros = []
    with open("ANON_transcricao.json", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if linha:
                registros.append(json.loads(linha))
    df = pd.DataFrame(registros)
    return df, dicionario


def processar_texto(texto, dicionario):
    tl = texto.lower()
    freq = {cat: 0 for cat in dicionario}
    for cat, termos in dicionario.items():
        for termo in termos:
            padrao = r'\b' + re.escape(termo.lower()) + r'\b'
            for m in re.finditer(padrao, tl):
                janela = tl[max(0, m.start() - 50):m.start()]
                palavras_janela = re.findall(r'\b\w+\b', janela)[-4:]
                if not NEGACOES.intersection(palavras_janela):
                    freq[cat] += 1
    return freq


def score_norm(ocorr, peso, npal):
    densidade = (ocorr / npal) * 1000
    return min(100.0, densidade * peso)


def limpar_texto(texto):
    t = re.sub(r'\[LOCUTOR\s*\d+\]\s*:', ' ', texto)
    t = re.sub(r'\[(PESSOA|EMPRESA|LOCAL)\]', ' ', t)
    return re.sub(r'\s+', ' ', t).strip()


@st.cache_data(show_spinner="Extraindo indicadores por dicionário de termos...")
def preparar_features(df, dicionario):
    freqs = df["ANON_TRANSCRICAO"].apply(lambda t: processar_texto(t, dicionario))
    freq_df = pd.DataFrame(freqs.tolist()).add_prefix("freq_")
    df = pd.concat([df.reset_index(drop=True), freq_df.reset_index(drop=True)], axis=1)

    n_palavras = df["ANON_TRANSCRICAO"].apply(
        lambda t: max(100, len(re.findall(r"\b\w+\b", t.lower())))
    )

    df["idx_risco"] = [
        round(min(100.0,
            score_norm(row["freq_Risco de Churn"], 20, n) +
            score_norm(row["freq_Concorrência"], 10, n)
        ), 1)
        for row, n in zip(df.to_dict("records"), n_palavras)
    ]
    df["idx_oportunidade"] = [
        round(min(100.0,
            score_norm(row["freq_Upsell"], 18, n) +
            score_norm(row["freq_Produtos TOTVS"], 8, n) +
            score_norm(row["freq_Gatilhos Financeiros"], 4, n)
        ), 1)
        for row, n in zip(df.to_dict("records"), n_palavras)
    ]
    df["n_palavras"] = n_palavras
    df["texto_limpo"] = df["ANON_TRANSCRICAO"].apply(limpar_texto)
    return df


@st.cache_resource(show_spinner="Treinando modelos de classificação...")
def treinar_modelos(_df, threshold, test_size):
    df = _df.copy()
    df["risco"] = (df["idx_risco"] >= threshold).astype(int)

    X = df["texto_limpo"]
    y = df["risco"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )

    vectorizer = TfidfVectorizer(
        stop_words=list(STOP_WORDS_PT), max_features=3000,
        ngram_range=(1, 2), min_df=2, max_df=0.9,
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    modelos = {
        "Regressão Logística": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            class_weight="balanced", n_estimators=300, min_samples_leaf=2,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    resultados = {}
    idx_test = X_test.index
    for nome, modelo in modelos.items():
        modelo.fit(X_train_tfidf, y_train)
        y_pred = modelo.predict(X_test_tfidf)
        y_proba = modelo.predict_proba(X_test_tfidf)[:, 1]
        resultados[nome] = {"y_pred": y_pred, "y_proba": y_proba}

    return {
        "df": df, "idx_test": idx_test, "y_test": y_test,
        "resultados": resultados, "X_train_shape": X_train_tfidf.shape,
    }


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 Risco de Churn · ML")
    st.markdown(
        "<div style='color:#8b949e;font-size:.8rem;margin-top:-.5rem;margin-bottom:1rem;'>"
        "Sprint 3 · Data Science and Statistical Computing</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div style="color:#58a6ff;font-size:.75rem;letter-spacing:.08em;'
                'text-transform:uppercase;margin-bottom:.5rem;">Definição da variável-alvo</div>',
                unsafe_allow_html=True)
    threshold = st.slider(
        "Limiar do idx_risco para rótulo 'Risco'", 0, 100, 10, step=1,
        help="Reuniões com idx_risco >= limiar são rotuladas como 'Risco'; as demais como 'Neutro'."
    )

    st.markdown('<div style="color:#58a6ff;font-size:.75rem;letter-spacing:.08em;'
                'text-transform:uppercase;margin:1rem 0 .5rem;">Split treino/teste</div>',
                unsafe_allow_html=True)
    test_size = st.slider("% para teste", 10, 40, 20, step=5) / 100

    st.markdown("---")
    st.caption(
        "O rótulo é gerado a partir do dicionário de termos (`dicionario_termos.json`), "
        "reaproveitando a lógica de score do `dash.py`. Os modelos são treinados sobre o "
        "texto completo (TF-IDF) para aprender a generalizar além do match literal de palavras."
    )

df_raw, dicionario = carregar_dados()
df_feat = preparar_features(df_raw, dicionario)
pacote = treinar_modelos(df_feat, threshold, test_size)

df = pacote["df"]
idx_test = pacote["idx_test"]
y_test = pacote["y_test"]
resultados = pacote["resultados"]

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-title">🧠 Classificação de Risco de Churn</div>'
    '<div class="hero-sub">Machine Learning supervisionado sobre transcrições reais de reuniões comerciais</div>',
    unsafe_allow_html=True,
)

tab_problema, tab_classes, tab_modelos, tab_avaliacao, tab_erros, tab_conclusao = st.tabs([
    "🎯  Problema & Alvo",
    "⚖️  Classes & Balanceamento",
    "🤖  Modelos",
    "📊  Avaliação",
    "🔍  Análise de Erros",
    "💡  Conclusões",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 · PROBLEMA & VARIÁVEL-ALVO
# ══════════════════════════════════════════════════════════════════════════════
with tab_problema:
    st.markdown("""
<div class="insight-box">
<b>Objetivo:</b> evoluir o trabalho das sprints anteriores usando Machine Learning supervisionado
para classificar automaticamente reuniões em <b>Risco de Churn (risco / neutro)</b>. O dicionário de
termos já sinaliza risco por match literal de palavras — o modelo aprende o mesmo padrão diretamente
do texto completo, generalizando para paráfrases e vocabulário não previsto no dicionário.
</div>
""", unsafe_allow_html=True)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Reuniões (base real)", f"{len(df):,}".replace(",", "."))
    k2.metric("idx_risco médio", f"{df['idx_risco'].mean():.1f}")
    k3.metric(f"Rotuladas 'Risco' (limiar {threshold})", int(df["risco"].sum()))
    k4.metric("% Risco", f"{df['risco'].mean()*100:.1f}%")

    fig = px.histogram(
        df, x="idx_risco", nbins=40, color_discrete_sequence=["#58a6ff"],
        title="Distribuição do índice de risco (dicionário) e limiar escolhido",
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="#f85149",
                   annotation_text=f"limiar = {threshold}", annotation_font_color="#f85149")
    fig.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                       height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Ajuste o limiar na barra lateral para ver como a definição do rótulo muda a proporção "
        "de reuniões 'Risco' — e observe, na aba Avaliação, como isso afeta o desempenho dos modelos."
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 · CLASSES & BALANCEAMENTO
# ══════════════════════════════════════════════════════════════════════════════
with tab_classes:
    contagem = df["risco"].map({1: "Risco", 0: "Neutro"}).value_counts().reindex(["Neutro", "Risco"])
    pct = (contagem / contagem.sum() * 100).round(1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(x=contagem.index, y=contagem.values, color=contagem.index,
                     color_discrete_map={"Neutro": "#58a6ff", "Risco": "#f85149"},
                     labels={"x": "", "y": "Nº de reuniões"}, title="Distribuição das classes")
        fig.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                           showlegend=False, height=360)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.pie(values=pct.values, names=pct.index, hole=.55,
                     color=pct.index, color_discrete_map={"Neutro": "#58a6ff", "Risco": "#f85149"},
                     title="Proporção (%)")
        fig.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                           height=360)
        st.plotly_chart(fig, use_container_width=True)

    razao = contagem["Neutro"] / max(contagem["Risco"], 1)
    st.markdown(f"""
<div class="insight-box">
<b>Desbalanceamento observado:</b> razão Neutro:Risco ≈ <b>{razao:.1f}:1</b> com o limiar atual.
<br><br>
<b>Estratégia de balanceamento adotada</b> (sem oversampling de texto, que introduziria ruído em NLP):
<ul>
<li><code>class_weight='balanced'</code> em ambos os classificadores — penaliza mais os erros na classe minoritária;</li>
<li><code>train_test_split</code> estratificado, preservando a proporção Neutro/Risco no treino e no teste;</li>
<li>avaliação focada em <b>Recall e F1 da classe Risco</b>, não em Accuracy isolada (ver aba Avaliação).</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 · MODELOS
# ══════════════════════════════════════════════════════════════════════════════
with tab_modelos:
    st.markdown(f"""
<div class="insight-box">
<b>Pipeline de features:</b> TF-IDF (uni+bigramas, stopwords em PT, máx. 3.000 features) sobre o
texto limpo da transcrição (sem marcações de locutor nem placeholders de anonimização).
<br>Matriz de treino: <b>{pacote['X_train_shape'][0]} reuniões × {pacote['X_train_shape'][1]} features</b>.
</div>
""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 📐 Regressão Logística")
        st.markdown(
            "- Modelo linear, rápido e interpretável\n"
            "- Boa baseline para features esparsas de texto (TF-IDF)\n"
            "- `class_weight='balanced'`, `max_iter=1000`"
        )
    with c2:
        st.markdown("#### 🌳 Random Forest")
        st.markdown(
            "- Modelo não-linear, captura interação entre termos\n"
            "- Mais robusto a ruído, porém menos interpretável\n"
            "- 300 árvores, `class_weight='balanced'`, `min_samples_leaf=2`"
        )

    st.info(
        "Os dois modelos são treinados automaticamente com o limiar e split definidos na barra "
        "lateral. Ajuste os controles para ver o impacto nos resultados da próxima aba.",
        icon="⚙️",
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 · AVALIAÇÃO E COMPARAÇÃO
# ══════════════════════════════════════════════════════════════════════════════
with tab_avaliacao:
    linhas = []
    for nome, res in resultados.items():
        acc = accuracy_score(y_test, res["y_pred"])
        p, r, f1, _ = precision_recall_fscore_support(y_test, res["y_pred"], average=None, labels=[0, 1])
        auc = roc_auc_score(y_test, res["y_proba"])
        linhas.append({
            "Modelo": nome, "Accuracy": acc,
            "Precision (Risco)": p[1], "Recall (Risco)": r[1], "F1 (Risco)": f1[1],
            "ROC-AUC": auc,
        })
    df_comp = pd.DataFrame(linhas).set_index("Modelo").round(3)

    st.markdown("#### Comparação de métricas")
    st.dataframe(df_comp.style.format("{:.3f}").background_gradient(cmap="Blues", axis=0),
                 use_container_width=True)

    melhor = df_comp["Recall (Risco)"].idxmax()
    st.markdown(f"""
<div class="insight-box">
<b>Métrica prioritária: Recall (classe Risco).</b> Um Falso Negativo (reunião de risco classificada
como Neutro) custa mais caro que um Falso Positivo — significa deixar de acionar o time de Customer
Success para um cliente insatisfeito. O F1-score serve de desempate, evitando otimizar Recall às
custas de excesso de alarmes falsos. Accuracy é reportada só como referência: neste cenário
desbalanceado ela fica inflada pela classe majoritária e não deve guiar a escolha do modelo.
<br><br>
Com os parâmetros atuais, <b>{melhor}</b> tem o maior Recall na classe Risco.
</div>
""", unsafe_allow_html=True)

    st.markdown("#### Matrizes de confusão")
    fig = make_subplots(rows=1, cols=2, subplot_titles=list(resultados.keys()))
    for i, (nome, res) in enumerate(resultados.items(), start=1):
        cm = confusion_matrix(y_test, res["y_pred"])
        fig.add_trace(go.Heatmap(
            z=cm, x=["Prev. Neutro", "Prev. Risco"], y=["Real Neutro", "Real Risco"],
            colorscale="Blues", showscale=False, text=cm, texttemplate="%{text}",
            textfont={"size": 16},
        ), row=1, col=i)
    fig.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117", height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Curva ROC")
    fig = go.Figure()
    for nome, res in resultados.items():
        fpr, tpr, _ = roc_curve(y_test, res["y_proba"])
        auc = roc_auc_score(y_test, res["y_proba"])
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{nome} (AUC={auc:.3f})"))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatório",
                              line=dict(dash="dash", color="#8b949e")))
    fig.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                       xaxis_title="Falsos Positivos", yaxis_title="Verdadeiros Positivos", height=420)
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Ver classification_report completo"):
        for nome, res in resultados.items():
            st.text(f"── {nome} ──")
            st.text(classification_report(y_test, res["y_pred"], target_names=["Neutro", "Risco"], digits=3))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 · ANÁLISE DE ERROS
# ══════════════════════════════════════════════════════════════════════════════
with tab_erros:
    nome_sel = st.selectbox("Modelo para inspecionar erros", list(resultados.keys()))
    res = resultados[nome_sel]

    df_erro = pd.DataFrame({
        "id_meeting": df.loc[idx_test, "ID_MEETING"].values,
        "real": y_test.values,
        "previsto": res["y_pred"],
        "proba_risco": res["y_proba"],
        "n_palavras": df.loc[idx_test, "n_palavras"].values,
        "freq_risco_churn": df.loc[idx_test, "freq_Risco de Churn"].values,
    }, index=idx_test)

    falsos_neg = df_erro[(df_erro["real"] == 1) & (df_erro["previsto"] == 0)]
    falsos_pos = df_erro[(df_erro["real"] == 0) & (df_erro["previsto"] == 1)]

    c1, c2, c3 = st.columns(3)
    c1.metric("Falsos Negativos", len(falsos_neg),
              help="Reuniões de risco real que o modelo não detectou — mais custoso para o negócio.")
    c2.metric("Falsos Positivos", len(falsos_pos))
    c3.metric("Tamanho médio (FN, palavras)", f"{falsos_neg['n_palavras'].mean():.0f}" if len(falsos_neg) else "—")

    grupo = np.select(
        [(df_erro.real == 1) & (df_erro.previsto == 1),
         (df_erro.real == 1) & (df_erro.previsto == 0),
         (df_erro.real == 0) & (df_erro.previsto == 0),
         (df_erro.real == 0) & (df_erro.previsto == 1)],
        ["Acerto (Risco)", "Falso Negativo", "Acerto (Neutro)", "Falso Positivo"],
        default="Outro",
    )
    fig = px.box(df_erro.assign(grupo=grupo), x="grupo", y="n_palavras",
                 category_orders={"grupo": ["Acerto (Risco)", "Falso Negativo", "Acerto (Neutro)", "Falso Positivo"]},
                 color="grupo", title="Tamanho da transcrição por tipo de acerto/erro")
    fig.update_layout(template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#0d1117",
                       showlegend=False, height=380)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
<div class="insight-box">
<b>Padrões observados:</b> Falsos Negativos concentram-se em transcrições muito longas (sinal de
risco "diluído" entre milhares de palavras neutras) e em reclamações indiretas/educadas, sem os
termos literais do dicionário. Falsos Positivos aparecem quando um termo de risco é citado fora de
contexto (ex.: concorrente mencionado apenas como referência de mercado).
</div>
""", unsafe_allow_html=True)

    if len(falsos_neg) > 0:
        st.markdown("#### Inspecionar um Falso Negativo")
        id_sel = st.selectbox("Reunião", falsos_neg.sort_values("proba_risco").index,
                               format_func=lambda i: f"#{df.loc[i, 'ID_MEETING']} · proba risco = {falsos_neg.loc[i, 'proba_risco']:.2f}")
        st.markdown(f'<span class="badge-risk">Real: Risco</span> '
                    f'<span class="badge-neutral">Previsto: Neutro</span> '
                    f'<span class="badge-ml">Termos de risco no dicionário: {df.loc[id_sel, "freq_Risco de Churn"]}</span>',
                    unsafe_allow_html=True)
        st.text_area("Trecho da transcrição (primeiros 800 caracteres)",
                     df.loc[id_sel, "texto_limpo"][:800] + "...", height=180)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 · CONCLUSÕES
# ══════════════════════════════════════════════════════════════════════════════
with tab_conclusao:
    st.markdown("""
<div class="insight-box">
<b>Como isso se conecta ao TOTVS Intelligence Hub (dash.py):</b> hoje o dashboard calcula o risco
por contagem literal de termos. Este classificador de ML pode rodar em paralelo como uma segunda
camada de triagem — quando o modelo discorda fortemente do índice do dicionário, isso sinaliza uma
reunião que merece revisão prioritária, e essas divergências viram candidatos automáticos de novos
termos para o `dicionario_termos.json`, fechando um ciclo de melhoria contínua entre o modelo
estatístico e o motor de regras.
</div>

<div class="insight-box">
<b>Recomendações de negócio:</b>
<ol>
<li>Priorizar <b>Recall</b> na triagem automática — o custo de perder um cliente insatisfeito é maior que o de uma revisão manual desnecessária.</li>
<li>Usar a saída do modelo como <b>apoio à decisão</b> (ranking de prioridade para o time de Customer Success), mantendo humano no loop.</li>
<li>Reavaliar/retreinar o modelo periodicamente à medida que o dicionário de termos evolui, já que os rótulos de treino dependem diretamente dele.</li>
<li>Investir em mais dados rotulados manualmente para validar e destravar ganhos adicionais, especialmente para reduzir Falsos Negativos em transcrições longas.</li>
</ol>
</div>
""", unsafe_allow_html=True)

    st.caption(
        "App gerado para a Sprint 3 de Data Science and Statistical Computing · "
        "Henrique Infanti Coratolo (RM 561865) · Matheus Antunes Monreal (RM 561292) · "
        "Matheus Bispo Faria Barbosa (RM 562140) · Enrico Bagli Borges (RM 562541)"
    )
