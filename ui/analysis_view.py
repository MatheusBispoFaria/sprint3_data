"""Componentes de resultado reutilizados na análise e no histórico."""
from __future__ import annotations
import plotly.graph_objects as go
from dash import dcc, html
from ui.components import metric_card, panel
from ui.viz import fixed_graph, transparent_layout


def _gauge(value: float, title: str) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=float(value),
        number={"suffix": "%", "font": {"size": 27}},
        title={"text": title, "font": {"size": 12}},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#33d4ff"},
               "bgcolor": "#0b1828", "borderwidth": 0,
               "steps": [{"range": [0, 30], "color": "#10283a"},
                         {"range": [30, 60], "color": "#1a2e3f"},
                         {"range": [60, 100], "color": "#35202c"}]},
    ))
    transparent_layout(fig, 225)
    fig.update_layout(margin={"l": 18, "r": 18, "t": 38, "b": 8})
    return fig


def render_analysis(result: dict):
    terms = result.get("termos_tfidf", [])
    terms_fig = go.Figure()
    if terms:
        terms_fig.add_trace(go.Bar(
            x=[x["peso"] for x in reversed(terms)],
            y=[x["termo"] for x in reversed(terms)], orientation="h",
        ))
    transparent_layout(terms_fig, 300)
    terms_fig.update_layout(showlegend=False, margin={"l": 100, "r": 12, "t": 15, "b": 30},
                            xaxis_title="Peso TF-IDF")
    products = ", ".join(result.get("produtos_totvs", [])) or "—"
    competitors = ", ".join(result.get("concorrentes", [])) or "—"
    patterns = ", ".join(result.get("padroes", [])) or "—"
    risk_ml = result.get("risco_ml")
    risk_sub = f"Regras {result['risco_regras']:.1f}% · ML {risk_ml:.1f}%" if risk_ml is not None else "Motor de regras"
    return html.Div([
        html.Div([
            metric_card("Classificação", result["classe_final"], tone="red" if result["classe_final"] == "Risco" else "green"),
            metric_card("Risco", f"{result['risco_final']:.1f}%", risk_sub, "red"),
            metric_card("Oportunidade", f"{result['oportunidade']:.1f}%", tone="green"),
            metric_card("Sentimento", result["sentimento"], tone="purple"),
            metric_card("Confiança", f"{result['confianca_scoring']:.1f}%"),
        ], className="metric-grid five"),
        html.Div([
            panel("Risco de churn", fixed_graph(_gauge(result["risco_final"], "Risco"), 225)),
            panel("Oportunidade", fixed_graph(_gauge(result["oportunidade"], "Oportunidade"), 225)),
            panel("Termos relevantes", fixed_graph(terms_fig, 300)),
        ], className="three-grid"),
        html.Div([
            panel("Informações extraídas", html.Div([
                html.P([html.B("Produtos: "), products]),
                html.P([html.B("Concorrentes: "), competitors]),
                html.P([html.B("Padrões: "), patterns]),
                html.P([html.B("Palavras: "), str(result.get("n_palavras", 0))]),
                html.P([html.B("Locutores: "), str(result.get("locutores", 0))]),
            ], className="detail-list")),
            panel("Próxima ação", html.Div(result.get("recomendacao", "—"), className="recommendation-box")),
        ], className="split-grid"),
    ])
