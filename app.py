from __future__ import annotations

import os
from dash import Dash, Input, Output, State, ctx, dcc, html, page_container
from flask import jsonify, request

from ui.components import sidebar
from services.data_service import analyze_transcript, get_model_summary, get_validation_summary
from services.reunioes import analisar_e_salvar
from storage.database import (
    contar_reunioes, excluir_reuniao, init_database, listar_reunioes, obter_reuniao,
)

init_database()
app = Dash(__name__, use_pages=True, pages_folder="pages", suppress_callback_exceptions=True,
           title="TOTVS | Meeting Intelligence")
server = app.server

app.layout = html.Div(className="app-shell", children=[
    dcc.Location(id="site-url", refresh=False),
    dcc.Store(id="sidebar-open", data=False),
    dcc.Store(id="meetings-revision", data=0),
    html.Div(id="sidebar-backdrop", className="sidebar-backdrop", n_clicks=0),
    sidebar(),
    html.Div(id="content-shell", className="content-shell", children=[
        html.Header(className="topbar", children=[
            html.Button("☰", id="sidebar-toggle", className="hamburger", n_clicks=0,
                        title="Menu", **{"aria-label": "Abrir menu"}),
            html.Span("TOTVS", className="topbar-brand"),
            html.Span("Meeting Intelligence", className="topbar-product"),
        ]),
        html.Main(page_container, className="page-container"),
    ]),
])


@app.callback(
    Output("sidebar", "className"),
    Output("sidebar-backdrop", "className"),
    Output("sidebar-open", "data"),
    Input("sidebar-toggle", "n_clicks"),
    Input("sidebar-backdrop", "n_clicks"),
    Input("site-url", "pathname"),
    State("sidebar-open", "data"),
)
def toggle_sidebar(_clicks, _dismiss, _path, opened):
    clicked = ctx.triggered_id
    if clicked == "sidebar-toggle":
        opened = not bool(opened)
    else:
        opened = False
    return ("sidebar" if opened else "sidebar sidebar-hidden",
            "sidebar-backdrop is-visible" if opened else "sidebar-backdrop", opened)


@server.get("/health")
def health():
    return jsonify({"status": "ok", "reunioes_salvas": contar_reunioes()})


def _receive_transcript(default_save: bool = True, flat: bool = False):
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"erro": "Envie um JSON com o campo 'texto'."}), 400
    texto = str(data.get("texto", "") or "").strip()
    if not texto:
        return jsonify({"erro": "Informe uma transcrição."}), 400
    if len(texto) > 200_000:
        return jsonify({"erro": "Limite de 200.000 caracteres excedido."}), 413
    try:
        if data.get("salvar", default_save) is False:
            analysis = analyze_transcript(texto)
            return jsonify({**analysis, "salvo": False} if flat else {"salvo": False, "analise": analysis})
        record = analisar_e_salvar(texto, cliente=data.get("cliente", ""),
                                   assunto=data.get("assunto", ""),
                                   data_reuniao=data.get("data_reuniao"), origem="api")
        return jsonify({**record["analise"], "salvo": True, "id_reuniao": record["id"]} if flat
                       else {"salvo": True, "id": record["id"], "analise": record["analise"]}), 201
    except ValueError as exc:
        return jsonify({"erro": str(exc)}), 400


@server.post("/api/analisar")
def api_analisar():
    return _receive_transcript(flat=True)


@server.route("/api/reunioes", methods=["GET", "POST"])
def api_reunioes():
    if request.method == "POST":
        return _receive_transcript()
    try:
        return jsonify({"total": contar_reunioes(), "reunioes": listar_reunioes(
            busca=request.args.get("busca", ""),
            classe=request.args.get("classe"),
            limite=int(request.args.get("limite", "100")),
            offset=int(request.args.get("offset", "0")),
        )})
    except ValueError:
        return jsonify({"erro": "Parâmetros inválidos."}), 400


@server.route("/api/reunioes/<int:meeting_id>", methods=["GET", "DELETE"])
def api_reuniao(meeting_id):
    if request.method == "DELETE":
        if excluir_reuniao(meeting_id):
            return "", 204
        return jsonify({"erro": "Reunião não encontrada."}), 404
    meeting = obter_reuniao(meeting_id)
    if meeting is None:
        return jsonify({"erro": "Reunião não encontrada."}), 404
    return jsonify(meeting)


@server.get("/api/modelo")
def api_modelo():
    return jsonify(get_model_summary())


@server.get("/api/validacao")
def api_validacao():
    return jsonify(get_validation_summary())


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8050")), debug=False)
