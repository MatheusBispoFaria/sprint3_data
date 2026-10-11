from __future__ import annotations
from collections import Counter
import json
import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from services.data_service import mean_confidence_interval
from storage.database import todas_reunioes_resumo
from ui.components import metric_card, page_header, panel
from ui.viz import fixed_graph, transparent_layout


dash.register_page(__name__, path="/dashboard", name="Dashboard", order=1, icon="▦")


def layout(**kwargs):
    return html.Div(className="page", children=[
        page_header("Dashboard"),
        html.Div(className="filter-bar dashboard-filters", children=[
            html.Div([html.Label("Classificação"), dcc.Dropdown(id="dashboard-class", options=[
                {"label": "Todas", "value": "__all__"},
                {"label": "Risco", "value": "Risco"},
                {"label": "Neutro", "value": "Neutro"}], value="__all__", clearable=False)], className="filter-item"),
            html.Div([html.Label("Sentimento"), dcc.Dropdown(id="dashboard-sentiment", options=[
                {"label": "Todos", "value": "__all__"},
                {"label": "Positivo", "value": "Positivo"},
                {"label": "Neutro", "value": "Neutro"},
                {"label": "Negativo", "value": "Negativo"}], value="__all__", clearable=False)], className="filter-item"),
        ]),
        html.Div(id="dashboard-content"),
    ])


def _empty_graph(message="Sem dados"):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False)
    transparent_layout(fig, 300)
    return fig


def _ci_label(values):
    avg, lo, hi, n = mean_confidence_interval(values)
    if n < 2: return "IC95% indisponível" if n == 1 else "—"
    return f"IC95% {lo:.1f}–{hi:.1f}"


@callback(
    Output("dashboard-content", "children"),
    Input("dashboard-class", "value"),
    Input("dashboard-sentiment", "value"),
    Input("meetings-revision", "data"),
)
def render_dashboard(selected_class, selected_sentiment, _revision):
    records = todas_reunioes_resumo()
    if selected_class not in (None, "__all__"):
        records = [r for r in records if r["classe_final"] == selected_class]
    if selected_sentiment not in (None, "__all__"):
        records = [r for r in records if r["sentimento"].lower() == selected_sentiment.lower()]
    if not records:
        return html.Div("Nenhuma reunião salva para os filtros selecionados.", className="empty-state")
    df = pd.DataFrame(records)
    r_avg, _, _, _ = mean_confidence_interval(df["risco_final"])
    o_avg, _, _, _ = mean_confidence_interval(df["oportunidade"])
    clients = df["cliente"].astype(str).str.strip().replace("", pd.NA).dropna().nunique()
    risks = px.histogram(df, x="risco_final", nbins=20, labels={"risco_final": "Risco (%)"})
    opp = px.histogram(df, x="oportunidade", nbins=20, labels={"oportunidade": "Oportunidade (%)"})
    risk_sentiment = px.bar(df["sentimento"].value_counts().rename_axis("Sentimento").reset_index(name="Reuniões"),
                            x="Sentimento", y="Reuniões")
    for fig in (risks, opp, risk_sentiment):
        transparent_layout(fig, 305)
        fig.update_layout(showlegend=False)
    counts = Counter()
    for r in records:
        try:
            counts.update(json.loads(r["analise_json"]).get("produtos_totvs", []))
        except (ValueError, TypeError):
            pass
    product_fig = go.Figure()
    if counts:
        names, nums = zip(*counts.most_common(10))
        product_fig.add_bar(x=list(nums)[::-1], y=list(names)[::-1], orientation="h")
    else:
        product_fig = _empty_graph("Nenhum produto identificado")
    transparent_layout(product_fig, 325)
    product_fig.update_layout(showlegend=False)

    return html.Div([
        html.Div([
            metric_card("Reuniões salvas", str(len(records))),
            metric_card("Clientes", str(clients)),
            metric_card("Risco médio", f"{r_avg:.1f}%", _ci_label(df["risco_final"]), "red"),
            metric_card("Oportunidade média", f"{o_avg:.1f}%", _ci_label(df["oportunidade"]), "green"),
        ], className="metric-grid four"),
        html.Div([
            panel("Distribuição de risco", fixed_graph(risks, 305)),
            panel("Distribuição de oportunidade", fixed_graph(opp, 305)),
        ], className="split-grid"),
        html.Div([
            panel("Sentimento", fixed_graph(risk_sentiment, 305)),
            panel("Produtos identificados", fixed_graph(product_fig, 325)),
        ], className="split-grid"),
    ])
