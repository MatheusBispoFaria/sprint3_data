from __future__ import annotations
import dash
from dash import Input, Output, State, callback, dcc, html, no_update

from services.reunioes import analisar_e_salvar
from ui.analysis_view import render_analysis
from ui.components import page_header, panel


dash.register_page(__name__, path="/", name="Transcrição", order=0, icon="◉")


def layout(**kwargs):
    return html.Div(className="page", children=[
        page_header("Analisar reunião"),
        panel("Nova reunião", html.Div([
            html.Div([
                html.Div([html.Label("Cliente"), dcc.Input(id="meeting-client", type="text", placeholder="Nome do cliente", maxLength=180)], className="field"),
                html.Div([html.Label("Assunto"), dcc.Input(id="meeting-subject", type="text", placeholder="Assunto da reunião", maxLength=220)], className="field"),
                html.Div([html.Label("Data"), dcc.DatePickerSingle(id="meeting-date", display_format="DD/MM/YYYY", placeholder="Data")], className="field"),
            ], className="input-row"),
            dcc.Textarea(id="transcript-input", value="", className="transcript-area",
                         placeholder="Cole a transcrição da reunião aqui..."),
            html.Div([
                html.Button("Analisar e salvar", id="analyze-button", n_clicks=0, className="primary-button"),
                html.Span(id="save-status", className="save-status"),
            ], className="action-row"),
        ]), class_name="entry-panel"),
        html.Div(id="analysis-output", className="analysis-output"),
    ])


@callback(
    Output("analysis-output", "children"),
    Output("save-status", "children"),
    Output("meetings-revision", "data"),
    Input("analyze-button", "n_clicks"),
    State("transcript-input", "value"),
    State("meeting-client", "value"),
    State("meeting-subject", "value"),
    State("meeting-date", "date"),
    State("meetings-revision", "data"),
    prevent_initial_call=True,
)
def run_analysis(_clicks, text, client, subject, date, revision):
    if not (text or "").strip():
        return no_update, "Insira uma transcrição.", no_update
    try:
        saved = analisar_e_salvar(text, cliente=client or "", assunto=subject or "", data_reuniao=date)
        return render_analysis(saved["analise"]), f"Reunião #{saved['id']} salva", int(revision or 0) + 1
    except ValueError as exc:
        return no_update, str(exc), no_update
    except Exception:
        return no_update, "Falha ao analisar ou salvar a reunião.", no_update
