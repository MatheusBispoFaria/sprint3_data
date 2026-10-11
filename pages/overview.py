from __future__ import annotations

import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html

from ui.components import metric_card, page_header, panel
from services.data_service import get_dataframe, group_mean_ci, mean_confidence_interval
from ui.viz import fixed_graph, transparent_layout


dash.register_page(__name__, path="/base-historica", name="Base de referência", order=3, icon="◷")

DF, _ = get_dataframe()
SEGMENTS = sorted([str(x) for x in DF.get("segmento", pd.Series(dtype=str)).dropna().unique()])
UFS = sorted([str(x) for x in DF.get("uf", pd.Series(dtype=str)).dropna().unique()])


def layout(**kwargs):
    return html.Div(
        className="page",
        children=[
            page_header("Base de referência"),
            html.Div(
                className="filter-bar two-filters",
                children=[
                    html.Div([html.Label("Segmento"), dcc.Dropdown(id="overview-segment", options=[{"label": "Todos", "value": "__all__"}] + [{"label": x, "value": x} for x in SEGMENTS], value="__all__", clearable=False)], className="filter-item"),
                    html.Div([html.Label("UF"), dcc.Dropdown(id="overview-uf", options=[{"label": "Todas", "value": "__all__"}] + [{"label": x, "value": x} for x in UFS], value="__all__", clearable=False)], className="filter-item"),
                ],
            ),
            html.Div(id="overview-content"),
        ],
    )


def _empty_fig(message: str):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False)
    transparent_layout(fig, 320)
    return fig


@callback(
    Output("overview-content", "children"),
    Input("overview-segment", "value"),
    Input("overview-uf", "value"),
)
def update_overview(segment, uf):
    df = DF.copy()
    if segment and segment != "__all__" and "segmento" in df.columns:
        df = df[df["segmento"].astype(str) == segment]
    if uf and uf != "__all__" and "uf" in df.columns:
        df = df[df["uf"].astype(str) == uf]

    if df.empty:
        return html.Div("Nenhuma reunião para os filtros selecionados.", className="empty-state")

    risk_mean, risk_low, risk_high, _ = mean_confidence_interval(df["idx_risco"])
    opp_mean, opp_low, opp_high, _ = mean_confidence_interval(df["idx_oportunidade"])
    nps = pd.to_numeric(df.get("nps", pd.Series(dtype=float)), errors="coerce").dropna()
    nps_mean = float(nps.mean()) if len(nps) else float("nan")

    risk_hist = px.histogram(df, x="idx_risco", nbins=28, labels={"idx_risco": "Risco"})
    transparent_layout(risk_hist, 330)
    risk_hist.update_layout(showlegend=False)

    opp_hist = px.histogram(df, x="idx_oportunidade", nbins=28, labels={"idx_oportunidade": "Oportunidade"})
    transparent_layout(opp_hist, 330)
    opp_hist.update_layout(showlegend=False)

    if "segmento" in df.columns and df["segmento"].notna().any():
        grouped = group_mean_ci(df, "segmento", "idx_risco", min_n=4).sort_values("media", ascending=False).head(12)
        if not grouped.empty:
            err_plus = grouped["ci_high"] - grouped["media"]
            err_minus = grouped["media"] - grouped["ci_low"]
            segment_fig = go.Figure(go.Bar(
                x=grouped["segmento"], y=grouped["media"],
                error_y={"type": "data", "array": err_plus, "arrayminus": err_minus, "visible": True},
                customdata=grouped["n"],
                hovertemplate="%{x}<br>Risco médio: %{y:.1f}<br>n=%{customdata}<extra></extra>",
            ))
            segment_fig.update_layout(yaxis_title="Risco médio", xaxis_title="Segmento")
            transparent_layout(segment_fig, 360)
        else:
            segment_fig = _empty_fig("Sem grupos suficientes para IC95%")
    else:
        segment_fig = _empty_fig("Segmento indisponível")

    return html.Div([
        html.Div([
            metric_card("Reuniões", f"{len(df):,}".replace(",", ".")),
            metric_card("Risco médio", f"{risk_mean:.1f}", f"IC95% {risk_low:.1f}–{risk_high:.1f}", "red"),
            metric_card("Oportunidade média", f"{opp_mean:.1f}", f"IC95% {opp_low:.1f}–{opp_high:.1f}", "green"),
            metric_card("NPS médio", "—" if pd.isna(nps_mean) else f"{nps_mean:.1f}", "Quando disponível", "purple"),
        ], className="metric-grid four"),
        html.Div([
            panel("Distribuição de risco", fixed_graph(risk_hist, 330)),
            panel("Distribuição de oportunidade", fixed_graph(opp_hist, 330)),
        ], className="split-grid"),
        panel("Risco médio por segmento · IC95%", fixed_graph(segment_fig, 360), ),
    ])
