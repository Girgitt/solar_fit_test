from __future__ import annotations

import dash
from dash import html

dash.register_page(__name__, path="/fleet-future", name="Fleet / Multi-day")

layout = html.Div(
    [
        html.H3("Fleet / Multi-day"),
        html.P("Placeholder for future analysis."),
        html.Ul(
            [
                html.Li("multi-day average power profile"),
                html.Li("range selection"),
                html.Li("average band over selected periods"),
                html.Li("comparison of multiple panels"),
            ]
        ),
    ],
    style={"padding": "8px"},
)