from __future__ import annotations
import dash
from dash import Input, Output, State, callback, dcc, html, no_update

from storage.database import excluir_reuniao, listar_reunioes, obter_reuniao
from ui.analysis_view import render_analysis
from ui.components import page_header, panel


dash.register_page(__name__, path="/reunioes", name="Reuniões salvas", order=2, icon="▤")


def layout(**kwargs):
    return html.Div(className="page", children=[
        page_header("Reuniões salvas"),
        html.Div(className="filter-bar saved-filters", children=[
            html.Div([html.Label("Buscar"), dcc.Input(id="saved-search", type="search",
                          placeholder="Cliente, assunto ou transcrição...", debounce=True)], className="field"),
            html.Div([html.Label("Classificação"), dcc.Dropdown(id="saved-class", options=[
                {"label": "Todas", "value": "__all__"},
                {"label": "Risco", "value": "Risco"},
                {"label": "Neutro", "value": "Neutro"}], value="__all__", clearable=False)], className="field"),
        ]),
        panel("Histórico", html.Div([
            dcc.Dropdown(id="meeting-selector", options=[], placeholder="Selecione uma reunião"),
            html.Div(className="saved-actions", children=[
                dcc.ConfirmDialogProvider(
                    id="confirm-delete", message="Excluir esta reunião definitivamente?",
                    children=html.Button("Excluir reunião", className="danger-button"),
                ),
                html.Span(id="delete-status", className="save-status"),
            ]),
        ])),
        html.Div(id="saved-detail", className="analysis-output"),
    ])


@callback(
    Output("meeting-selector", "options"),
    Input("saved-search", "value"),
    Input("saved-class", "value"),
    Input("meetings-revision", "data"),
)
def update_saved_list(search, risk, _revision):
    records = listar_reunioes(busca=search or "", classe=risk if risk != "__all__" else None,
                               limite=500)
    return [{"label": f"#{r['id']} · {r['cliente'] or 'Sem cliente'} · {r['assunto'] or 'Reunião'} · {r['criado_em'][:10]} · {r['classe_final']}",
             "value": r["id"]} for r in records]


@callback(
    Output("saved-detail", "children"),
    Input("meeting-selector", "value"),
    Input("meetings-revision", "data"),
)
def display_saved_meeting(meeting_id, _revision):
    if not meeting_id:
        return None
    meeting = obter_reuniao(meeting_id)
    if not meeting:
        return html.Div("Reunião não encontrada.", className="empty-state")
    return html.Div([
        html.Div([
            html.Span(f"#{meeting['id']}", className="saved-id"),
            html.H2(meeting["assunto"] or "Reunião"),
            html.Span(meeting["cliente"] or "Sem cliente", className="muted-copy"),
            html.Span(meeting["data_reuniao"] or meeting["criado_em"][:10], className="muted-copy"),
        ], className="saved-header"),
        render_analysis(meeting["analise"]),
        panel("Transcrição original", html.Pre(meeting["transcricao"], className="saved-transcript")),
    ])


@callback(
    Output("delete-status", "children"),
    Output("meetings-revision", "data", allow_duplicate=True),
    Output("meeting-selector", "value"),
    Input("confirm-delete", "submit_n_clicks"),
    State("meeting-selector", "value"),
    State("meetings-revision", "data"),
    prevent_initial_call=True,
)
def delete_selected(clicks, meeting_id, revision):
    if not clicks or not meeting_id:
        return "Selecione uma reunião.", no_update, no_update
    if excluir_reuniao(meeting_id):
        return "Reunião excluída.", int(revision or 0) + 1, None
    return "Reunião não encontrada.", no_update, None
