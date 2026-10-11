from __future__ import annotations
import dash
from dash import dcc, html


def sidebar():
    pages = sorted((p for p in dash.page_registry.values() if not p.get("hidden")),
                   key=lambda p: p.get("order", 999))
    return html.Aside(
        id="sidebar", className="sidebar sidebar-hidden", children=[
            html.Div([html.Span("T", className="brand-mark"),
                      html.Div([html.B("TOTVS"), html.Span("Meeting Intelligence")], className="brand-text")],
                     className="brand-block"),
            html.Nav([
                dcc.Link([html.Span(p.get("icon", "•"), className="nav-icon"),
                          html.Span(p["name"], className="nav-label")],
                         href=p["relative_path"], className="nav-link")
                for p in pages
            ], className="nav-list"),
        ],
    )


def page_header(title: str, description: str = ""):
    return html.Div([html.H1(title), html.P(description) if description else None], className="page-header")


def metric_card(label: str, value, sub: str = "", tone: str = "cyan"):
    return html.Div([
        html.Div(label, className="metric-label"),
        html.Div(value, className="metric-value"),
        html.Div(sub, className="metric-sub") if sub else None,
    ], className=f"metric-card tone-{tone}")


def panel(title: str, children, subtitle: str | None = None, class_name: str = ""):
    return html.Section([
        html.Div([html.H3(title), html.P(subtitle) if subtitle else None], className="panel-head"),
        children,
    ], className=f"panel {class_name}".strip())
