"""Utilitários de aparência e dimensionamento estável dos gráficos Dash/Plotly."""
from __future__ import annotations

from dash import dcc


def transparent_layout(fig, height: int | None = None):
    kwargs = {
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"color": "#eaf4ff"},
        "margin": {"l": 45, "r": 20, "t": 35, "b": 45},
        "xaxis": {"gridcolor": "#173149", "zerolinecolor": "#173149"},
        "yaxis": {"gridcolor": "#173149", "zerolinecolor": "#173149"},
    }
    if height is not None:
        kwargs["height"] = height
        # Altura explícita: não reajustar o eixo Y para a altura do painel.
        kwargs["autosize"] = False
    fig.update_layout(**kwargs)
    return fig


def fixed_graph(fig, height: int, **kwargs):
    """Grafo com altura travada para evitar realimentação do ResizeObserver.

    O Dash/Plotly pode tentar redimensionar um Graph contido em CSS Grid
    cujo tamanho vertical é calculado a partir do próprio Graph. Essa
    dependência circular pode alongar a página a cada atualização.
    """
    if height <= 0:
        raise ValueError("A altura do gráfico deve ser positiva")
    fig.update_layout(height=height, autosize=False)
    return dcc.Graph(
        figure=fig,
        responsive=False,
        config={"displayModeBar": False, "responsive": False},
        style={
            "height": f"{height}px",
            "maxHeight": f"{height}px",
            "minHeight": f"{height}px",
            "width": "100%",
        },
        className="fixed-chart",
        **kwargs,
    )
