from __future__ import annotations

import dash
from dash import Dash, Input, Output, dcc, html

app = Dash(
    __name__,
    use_pages=True,
    suppress_callback_exceptions=True,
    title="PV Analytics",
)

# Order tabs explicitly
TAB_PAGES = [
    ("/pv-daily", "PV Daily"),
    ("/fleet-future", "Fleet / Multi-day"),
]


def build_tabs(current_path: str):
    current = current_path or TAB_PAGES[0][0]
    return dcc.Tabs(
        id="top-tabs",
        value=current,
        children=[
            dcc.Tab(label=label, value=path)
            for path, label in TAB_PAGES
        ],
    )


app.layout = html.Div(
    [
        dcc.Location(id="url"),
        html.Div(id="tabs-container", style={"marginBottom": "12px"}),
        dash.page_container,
    ],
    style={"padding": "12px"},
)


@app.callback(
    Output("tabs-container", "children"),
    Input("url", "pathname"),
)
def sync_tabs_with_url(pathname):
    return build_tabs(pathname)


@app.callback(
    Output("url", "pathname"),
    Input("top-tabs", "value"),
    prevent_initial_call=True,
)
def navigate_from_tab(tab_value):
    return tab_value


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)