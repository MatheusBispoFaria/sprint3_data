"""Verifica a prevenção de crescimento vertical dos gráficos Plotly."""
import unittest
import plotly.graph_objects as go

try:
    from dash import dcc
    from ui.viz import fixed_graph
except ImportError:
    dcc = None


@unittest.skipIf(dcc is None, "Dash não está instalado neste ambiente")
class FixedGraphSizingTests(unittest.TestCase):
    def test_fixed_chart_has_explicit_non_responsive_height(self):
        figure = go.Figure(go.Bar(x=[1, 2], y=[3, 4]))
        for height in (225, 300, 330):
            with self.subTest(height=height):
                graph = fixed_graph(figure, height)
                self.assertIsInstance(graph, dcc.Graph)
                self.assertFalse(graph.responsive)
                self.assertFalse(graph.config["responsive"])
                self.assertFalse(graph.figure.layout.autosize)
                self.assertEqual(graph.figure.layout.height, height)
                self.assertEqual(graph.style["height"], f"{height}px")
                self.assertEqual(graph.style["maxHeight"], f"{height}px")
                self.assertEqual(graph.style["minHeight"], f"{height}px")

    def test_rejects_invalid_height(self):
        with self.assertRaises(ValueError):
            fixed_graph(go.Figure(), 0)


if __name__ == "__main__":
    unittest.main()
