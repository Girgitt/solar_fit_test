from __future__ import annotations

from typing import Any, Dict, List

import dash
from dash import Input, Output, callback, dcc, html
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from app_runtime import config_store, engine
from models import build_iv_curve


dash.register_page(__name__, path="/simulation", name="Simulation")


CARD_STYLE = {
    "border": "1px solid #d9d9d9",
    "borderRadius": "10px",
    "padding": "12px",
    "backgroundColor": "#fafafa",
}


def metric_card(title: str, value: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, style={"color": "#666", "fontSize": "0.9rem", "marginBottom": "4px"}),
            html.Div(value, style={"fontSize": "1.3rem", "fontWeight": 700}),
        ],
        style=CARD_STYLE,
    )


def _build_layout() -> html.Div:
    cfg = config_store.get()
    irr_cfg = cfg["irradiance"]
    pwm_cfg = cfg["pwm"]
    snap = engine.get_snapshot()

    return html.Div(
        [
            html.H3("Simulation"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Nasłonecznienie [W/m²]", style={"fontWeight": 600, "marginBottom": "8px"}),
                            dcc.Slider(
                                id="sim-irradiance-slider",
                                min=float(irr_cfg["min"]),
                                max=float(irr_cfg["max"]),
                                step=float(irr_cfg.get("slider_step", 10.0)),
                                value=float(snap.get("irradiance", irr_cfg["initial"])),
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                        ],
                        style=CARD_STYLE,
                    ),
                    html.Div(
                        [
                            html.Div("Źródło duty", style={"fontWeight": 600, "marginBottom": "8px"}),
                            dcc.RadioItems(
                                id="sim-duty-source",
                                options=[
                                    {"label": "MQTT set_duty", "value": "mqtt"},
                                    {"label": "Ręczne nadpisanie", "value": "manual"},
                                ],
                                value="manual" if snap.get("manual_override") else "mqtt",
                                inline=True,
                            ),
                            html.Div(style={"height": "10px"}),
                            html.Div("Duty ręczne", style={"fontWeight": 600, "marginBottom": "8px"}),
                            dcc.Slider(
                                id="sim-manual-duty-slider",
                                min=float(pwm_cfg["duty_min"]),
                                max=float(pwm_cfg["duty_max"]),
                                step=max((float(pwm_cfg["duty_max"]) - float(pwm_cfg["duty_min"])) / 1000.0, 0.001),
                                value=float(snap.get("manual_duty", pwm_cfg["duty_initial"])),
                                tooltip={"placement": "bottom", "always_visible": True},
                            ),
                            html.Div(style={"height": "12px"}),
                            html.Button("Reset nagrzania żarówki", id="sim-reset-lamp", n_clicks=0),
                        ],
                        style=CARD_STYLE,
                    ),
                ],
                style={"display": "grid", "gridTemplateColumns": "1.4fr 1fr", "gap": "14px", "marginBottom": "14px"},
            ),
            html.Div(id="sim-summary-cards", style={"display": "grid", "gridTemplateColumns": "repeat(5, minmax(150px, 1fr))", "gap": "12px", "marginBottom": "14px"}),
            html.Div(id="sim-secondary-cards", style={"display": "grid", "gridTemplateColumns": "repeat(4, minmax(150px, 1fr))", "gap": "12px", "marginBottom": "14px"}),
            dcc.Graph(id="sim-graph", style={"height": "680px"}),
            html.Div(id="sim-mqtt-status", style={"marginTop": "10px"}),
            html.Div(id="sim-hidden-action", style={"display": "none"}),
            dcc.Interval(id="sim-interval", interval=500, n_intervals=0),
        ],
        style={"padding": "10px 6px 24px 6px"},
    )


layout = _build_layout


@callback(
    Output("sim-hidden-action", "children"),
    Input("sim-irradiance-slider", "value"),
    Input("sim-duty-source", "value"),
    Input("sim-manual-duty-slider", "value"),
    Input("sim-reset-lamp", "n_clicks"),
    prevent_initial_call=False,
)
def apply_controls(irradiance, duty_source, manual_duty, reset_clicks):
    if irradiance is not None:
        engine.set_irradiance(float(irradiance))
    engine.set_manual_override(duty_source == "manual")
    if manual_duty is not None:
        engine.set_manual_duty(float(manual_duty))
    if reset_clicks:
        # callback fires after each click; resetting repeatedly is harmless
        engine.reset_lamp()
    return "ok"


@callback(
    Output("sim-summary-cards", "children"),
    Output("sim-secondary-cards", "children"),
    Output("sim-graph", "figure"),
    Output("sim-mqtt-status", "children"),
    Input("sim-interval", "n_intervals"),
)
def refresh_simulation(_n):
    cfg = config_store.get()
    snap = engine.get_snapshot()
    curve = build_iv_curve(cfg, float(snap["irradiance"]), samples=int(cfg["pv"]["curve_samples"]))

    summary_cards = [
        metric_card("Nasłonecznienie", f"{snap['irradiance']:.1f} W/m²"),
        metric_card("Napięcie PV", f"{snap['voltage']:.3f} V"),
        metric_card("Prąd PV", f"{snap['current']:.3f} A"),
        metric_card("Moc PV", f"{snap['power']:.3f} W"),
        metric_card("Duty efektywne", f"{snap['effective_duty']:.4f} ({snap['effective_duty_norm']*100.0:.1f}% / {snap['duty_source']})"),
    ]

    overload_ratio = float(snap.get("overload_ratio", 0.0))
    second_cards = [
        metric_card("Opór obciążenia", f"{snap['resistance_ohm']:.4f} Ω"),
        metric_card("Stan nagrzania", f"{snap['lamp_temp_state']:.4f}"),
        metric_card("Voc / Isc", f"{snap['voc']:.3f} V / {snap['isc']:.3f} A"),
        metric_card("Przeciążenie lampy", f"{overload_ratio*100.0:.1f}% P_nom"),
        metric_card("Wiek symulacji", f"{snap['state_age_ms']} ms"),
    ]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Scatter(
            x=curve["currents"],
            y=curve["voltages"],
            mode="lines",
            name="Krzywa IV (V względem I)",
            hovertemplate="I=%{x:.4f} A<br>V=%{y:.4f} V<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=curve["currents"],
            y=curve["powers"],
            mode="lines",
            name="Krzywa mocy P(I)",
            hovertemplate="I=%{x:.4f} A<br>P=%{y:.4f} W<extra></extra>",
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=[snap["current"]],
            y=[snap["voltage"]],
            mode="markers",
            marker={"size": 11, "symbol": "diamond"},
            name="Aktualny punkt pracy (V/I)",
            hovertemplate="I=%{x:.4f} A<br>V=%{y:.4f} V<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=[snap["current"]],
            y=[snap["power"]],
            mode="markers",
            marker={"size": 11, "symbol": "x"},
            name="Aktualny punkt mocy",
            hovertemplate="I=%{x:.4f} A<br>P=%{y:.4f} W<extra></extra>",
        ),
        secondary_y=True,
    )

    external = snap.get("last_external_measurement")
    if external:
        fig.add_trace(
            go.Scatter(
                x=[external["i"]],
                y=[external["v"]],
                mode="markers",
                marker={"size": 10, "symbol": "circle-open"},
                name="Ostatni pomiar z MQTT",
                hovertemplate="I=%{x:.4f} A<br>V=%{y:.4f} V<extra></extra>",
            ),
            secondary_y=False,
        )
        second_cards.append(metric_card("Zewn. pomiar MQTT", f"{external['v']:.3f} V / {external['i']:.3f} A"))
        if snap.get("external_age_local_ms") is not None:
            second_cards.append(metric_card("Wiek odbioru MQTT", f"{snap['external_age_local_ms']} ms"))

    fig.update_layout(
        title="Charakterystyka IV i moc dla aktualnego nasłonecznienia",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 50, "r": 50, "t": 70, "b": 45},
    )
    fig.update_xaxes(title_text="Prąd [A]")
    fig.update_yaxes(title_text="Napięcie [V]", secondary_y=False)
    fig.update_yaxes(title_text="Moc [W]", secondary_y=True)

    duty_age = snap.get("duty_age_ms")
    duty_age_txt = "brak odebranego set_duty" if duty_age is None else f"ostatni set_duty {duty_age} ms temu"
    mqtt_status = html.Div(
        [
            html.Div(f"MQTT status: {snap.get('mqtt_status', '-')}", style={"fontWeight": 700}),
            html.Div(f"Połączenie: {'tak' if snap.get('mqtt_connected') else 'nie'}; {duty_age_txt}"),
            html.Div(
                "Duty jest normalizowane do skonfigurowanego zakresu PWM. Model żarówki nie ma twardego limitu mocy: przeciążenie jest widoczne jako wzrost mocy i wskaźnika P/P_nom."
            ),
        ],
        style=CARD_STYLE,
    )

    return summary_cards, second_cards, fig, mqtt_status
