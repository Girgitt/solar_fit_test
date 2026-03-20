from __future__ import annotations

import dash
from dash import Dash, Input, Output, dcc, html

from app_runtime import ensure_started

ensure_started()

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title="PV MPPT Simulator",
)

TAB_PAGES = [
    ("/configuration", "Configuration"),
    ("/simulation", "Simulation"),
]


def build_tabs(current_path: str):
    current = current_path or TAB_PAGES[0][0]
    return dcc.Tabs(
        id="top-tabs",
        value=current,
        children=[dcc.Tab(label=label, value=path) for path, label in TAB_PAGES],
    )


app.layout = html.Div(
    [
        dcc.Location(id="url"),
        html.H2("PV / MPPT MQTT Simulator", style={"marginBottom": "10px"}),
        html.Div(
            "Wielostronicowa aplikacja do strojenia modelu PV, symulacji żarówki halogenowej i testowania zewnętrznego algorytmu MPPT przez MQTT.",
            style={"marginBottom": "12px", "color": "#444"},
        ),
        html.Div(id="tabs-container", style={"marginBottom": "12px"}),
        dash.page_container,
    ],
    style={"padding": "14px", "maxWidth": "1480px", "margin": "0 auto"},
)


@app.callback(Output("tabs-container", "children"), Input("url", "pathname"))
def sync_tabs_with_url(pathname):
    return build_tabs(pathname)


@app.callback(Output("url", "pathname"), Input("top-tabs", "value"), prevent_initial_call=True)
def navigate_from_tab(tab_value):
    return tab_value


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8051)
