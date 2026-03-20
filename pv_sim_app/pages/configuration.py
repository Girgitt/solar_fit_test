from __future__ import annotations

import json
from typing import Any, Dict

import dash
from dash import Input, Output, State, callback, dcc, html

from app_runtime import config_store, engine
from models import resistance_hot_and_cold, solve_shape_params


dash.register_page(__name__, path="/configuration", name="Configuration")


CARD_STYLE = {
    "border": "1px solid #d9d9d9",
    "borderRadius": "10px",
    "padding": "14px",
    "marginBottom": "14px",
    "backgroundColor": "#fafafa",
}

ROW_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "280px minmax(180px, 260px) 1fr",
    "gap": "12px",
    "alignItems": "center",
    "marginBottom": "10px",
}


def _bool_dropdown(component_id: str, value: bool):
    return dcc.Dropdown(
        id=component_id,
        options=[
            {"label": "Tak", "value": "true"},
            {"label": "Nie", "value": "false"},
        ],
        value="true" if value else "false",
        clearable=False,
    )


def _number(component_id: str, value: Any, step: float = 0.1):
    return dcc.Input(id=component_id, type="number", value=value, step=step, debounce=True, style={"width": "100%"})


def _text(component_id: str, value: str, input_type: str = "text"):
    return dcc.Input(id=component_id, type=input_type, value=value, debounce=True, style={"width": "100%"})


def _field(label: str, component, help_text: str):
    return html.Div(
        [
            html.Div(label, style={"fontWeight": 600}),
            component,
            html.Div(help_text, style={"color": "#666", "fontSize": "0.92rem"}),
        ],
        style=ROW_STYLE,
    )


def _build_layout() -> html.Div:
    cfg = config_store.get()
    mqtt_cfg = cfg["mqtt"]
    irr_cfg = cfg["irradiance"]
    pv_cfg = cfg["pv"]
    load_cfg = cfg["load"]
    pwm_cfg = cfg["pwm"]

    return html.Div(
        [
            html.H3("Konfiguracja symulatora PV + obciążenia"),
            dcc.Markdown(
                """
Ta aplikacja realizuje dwa sprzężone modele:

1. **Moduł PV**: krzywa IV jest zdefiniowana przez punkt referencyjny `Isc/Voc/Imp/Vmp` dla nasłonecznienia odniesienia.  
   Przy zmianie nasłonecznienia prąd skaluje się prawie liniowo, a napięcie jałowe zmienia się łagodnie logarytmicznie.

2. **Obciążenie halogenowe**: PWM steruje mocą średnią, włókno ma bezwładność cieplną, a jego opór rośnie wraz z temperaturą.  
   Dzięki temu dla szybkich zmian `set_duty` opór nie zmienia się natychmiast, co lepiej przypomina zachowanie żarówki.
                """
            ),
            html.Div(id="config-save-status", style={"marginBottom": "12px", "fontWeight": 600}),
            html.Div(
                [
                    html.H4("MQTT"),
                    _field("MQTT włączone", _bool_dropdown("cfg-mqtt-enabled", bool(mqtt_cfg["enabled"])), "Umożliwia połączenie z brokerem i pracę z topicami."),
                    _field("Host brokera", _text("cfg-mqtt-host", str(mqtt_cfg["host"])), "Adres IP lub DNS brokera MQTT."),
                    _field("Port brokera", _number("cfg-mqtt-port", mqtt_cfg["port"], 1), "Zwykle 1883 lub 8883."),
                    _field("Nazwa użytkownika", _text("cfg-mqtt-username", str(mqtt_cfg["username"])), "Pozostaw puste, jeśli broker nie wymaga logowania."),
                    _field("Hasło", _text("cfg-mqtt-password", str(mqtt_cfg["password"]), input_type="password"), "Hasło do brokera MQTT."),
                    _field("Client ID", _text("cfg-mqtt-client-id", str(mqtt_cfg["client_id"])), "Identyfikator klienta MQTT."),
                    _field("Topic pomiarowy", _text("cfg-mqtt-measurement-topic", str(mqtt_cfg["measurement_topic"])), "Topic z pomiarem V/I oraz topic publikacji pomiaru symulowanego."),
                    _field("Klucz kanału w JSON", _text("cfg-mqtt-channel-key", str(mqtt_cfg["measurement_channel_key"])), "Np. `channel_1` w strukturze `{channel_1:{v:...,i:...}}`."),
                    _field("Topic set_duty", _text("cfg-mqtt-duty-topic", str(mqtt_cfg["duty_topic"])), "Na tym topicu oczekiwane jest PWM jako float albo JSON z kluczem `set_duty`."),
                    _field("Publikuj pomiar symulowany", _bool_dropdown("cfg-mqtt-publish-enabled", bool(mqtt_cfg["publish_simulated_measurements"])), "Gdy włączone, aplikacja publikuje aktualne V/I do topicu pomiarowego."),
                    _field("Subskrybuj topic pomiarowy", _bool_dropdown("cfg-mqtt-subscribe-meas", bool(mqtt_cfg["subscribe_measurement_topic"])), "Przydatne, jeśli chcesz porównać dane z zewnętrznego źródła z punktem symulowanym."),
                    _field("Interwał publikacji [s]", _number("cfg-mqtt-publish-interval", mqtt_cfg["publish_interval_sec"], 0.05), "Jak często publikować V/I z symulatora."),
                    _field("Keepalive [s]", _number("cfg-mqtt-keepalive", mqtt_cfg["keepalive_sec"], 1), "Czas keepalive połączenia MQTT."),
                    _field("QoS", _number("cfg-mqtt-qos", mqtt_cfg["qos"], 1), "QoS dla subskrypcji i publikacji."),
                    _field("Retain pomiarów", _bool_dropdown("cfg-mqtt-retain", bool(mqtt_cfg["retain_measurements"])), "Czy publikowany pomiar ma mieć flagę retain."),
                ],
                style=CARD_STYLE,
            ),
            html.Div(
                [
                    html.H4("Nasłonecznienie i sposób deformacji krzywej IV"),
                    _field("Min nasłonecznienia", _number("cfg-irr-min", irr_cfg["min"], 1), "Dolna granica suwaka symulacji, np. 100 W/m²."),
                    _field("Max nasłonecznienia", _number("cfg-irr-max", irr_cfg["max"], 1), "Górna granica suwaka symulacji, np. 1200 W/m²."),
                    _field("Wartość początkowa", _number("cfg-irr-initial", irr_cfg["initial"], 1), "Nasłonecznienie po starcie programu."),
                    _field("Nasłonecznienie odniesienia", _number("cfg-irr-reference", irr_cfg["reference"], 1), "W tym punkcie obowiązują referencyjne `Isc/Voc/Imp/Vmp`."),
                    _field("Krok suwaka", _number("cfg-irr-step", irr_cfg["slider_step"], 1), "Rozdzielczość suwaka na stronie Simulation."),
                    _field("Współczynnik log(Voc)", _number("cfg-irr-voc-log", irr_cfg["voc_log_coeff"], 0.01), "Im większy, tym mocniej Voc reaguje na zmianę nasłonecznienia."),
                ],
                style=CARD_STYLE,
            ),
            html.Div(
                [
                    html.H4("Referencyjna charakterystyka modułu PV"),
                    _field("Isc_ref [A]", _number("cfg-pv-isc", pv_cfg["isc_ref"], 0.01), "Prąd zwarciowy dla nasłonecznienia odniesienia."),
                    _field("Voc_ref [V]", _number("cfg-pv-voc", pv_cfg["voc_ref"], 0.01), "Napięcie jałowe dla nasłonecznienia odniesienia."),
                    _field("Imp_ref [A]", _number("cfg-pv-imp", pv_cfg["imp_ref"], 0.01), "Prąd w punkcie maksymalnej mocy."),
                    _field("Vmp_ref [V]", _number("cfg-pv-vmp", pv_cfg["vmp_ref"], 0.01), "Napięcie w punkcie maksymalnej mocy."),
                    _field("Liczba próbek krzywej", _number("cfg-pv-samples", pv_cfg["curve_samples"], 1), "Ile punktów rysować na wykresie IV/PV."),
                    html.Div(id="config-pv-summary", style={"marginTop": "10px", "padding": "10px", "background": "white", "borderRadius": "8px"}),
                ],
                style=CARD_STYLE,
            ),
            html.Div(
                [
                    html.H4("Model obciążenia halogenowego"),
                    _field("Napięcie znamionowe [V]", _number("cfg-load-vnom", load_cfg["lamp_rated_voltage"], 0.01), "Z tego i z mocy znamionowej liczony jest opór gorącego włókna."),
                    _field("Moc znamionowa [W]", _number("cfg-load-pnom", load_cfg["lamp_rated_power"], 0.01), "Moc znamionowa żarówki halogenowej."),
                    _field("R_hot / R_cold", _number("cfg-load-hot-cold", load_cfg["hot_to_cold_ratio"], 0.1), "Ile razy opór rozgrzanego włókna jest większy od zimnego."),
                    _field("Stała czasowa [s]", _number("cfg-load-tau", load_cfg["thermal_tau_sec"], 0.01), "Bezwładność cieplna modelu włókna."),
                    _field("Wykładnik temp. od mocy", _number("cfg-load-exp", load_cfg["thermal_power_exponent"], 0.01), "Łagodzi zależność temperatury od chwilowej mocy."),
                    html.Div(id="config-load-summary", style={"marginTop": "10px", "padding": "10px", "background": "white", "borderRadius": "8px"}),
                ],
                style=CARD_STYLE,
            ),
            html.Div(
                [
                    html.H4("PWM"),
                    _field("Duty min", _number("cfg-pwm-min", pwm_cfg["duty_min"], 0.001), "Dolna granica sterowania PWM."),
                    _field("Duty max", _number("cfg-pwm-max", pwm_cfg["duty_max"], 0.001), "Górna granica sterowania PWM; może to być np. 1.0 albo 100.0."),
                    _field("Duty początkowe", _number("cfg-pwm-initial", pwm_cfg["duty_initial"], 0.001), "Wartość używana do czasu pierwszego komunikatu MQTT."),
                ],
                style=CARD_STYLE,
            ),
            html.Button("Zapisz konfigurację", id="cfg-save-button", n_clicks=0, style={"padding": "10px 18px", "fontWeight": 700}),
        ],
        style={"padding": "10px 6px 24px 6px"},
    )


layout = _build_layout


@callback(
    Output("config-pv-summary", "children"),
    Output("config-load-summary", "children"),
    Input("cfg-pv-isc", "value"),
    Input("cfg-pv-voc", "value"),
    Input("cfg-pv-imp", "value"),
    Input("cfg-pv-vmp", "value"),
    Input("cfg-load-vnom", "value"),
    Input("cfg-load-pnom", "value"),
    Input("cfg-load-hot-cold", "value"),
)
def preview_model(isc, voc, imp, vmp, vnom, pnom, hotcold):
    try:
        a, b = solve_shape_params(float(isc), float(voc), float(imp), float(vmp))
        pv_summary = html.Div(
            [
                html.Div(f"Wyprowadzony kształt krzywej IV: a = {a:.4f}, b = {b:.4f}"),
                html.Div(f"Moc referencyjna Pmp_ref = {float(vmp) * float(imp):.3f} W"),
                html.Div("Model zachowa podany punkt MPP jako maksimum krzywej."),
            ]
        )
    except Exception as exc:
        pv_summary = html.Div(f"Błąd parametrów PV: {exc}", style={"color": "crimson"})

    try:
        r_hot, r_cold = resistance_hot_and_cold(
            {
                "lamp_rated_voltage": float(vnom),
                "lamp_rated_power": float(pnom),
                "hot_to_cold_ratio": float(hotcold),
            }
        )
        load_summary = html.Div(
            [
                html.Div(f"Opór gorącego włókna R_hot = {r_hot:.4f} Ω"),
                html.Div(f"Opór zimnego włókna R_cold = {r_cold:.4f} Ω"),
                html.Div(f"Prąd znamionowy I_nom = {float(pnom) / max(float(vnom), 1e-9):.4f} A"),
                html.Div("W symulacji opór będzie płynnie przechodził od R_cold do R_hot wraz z nagrzewaniem. Moc nie jest sztucznie ograniczana — przewoltowanie lampy może dać P > P_nom, co będzie widoczne jako przeciążenie."),
            ]
        )
    except Exception as exc:
        load_summary = html.Div(f"Błąd parametrów obciążenia: {exc}", style={"color": "crimson"})

    return pv_summary, load_summary


@callback(
    Output("config-save-status", "children"),
    Input("cfg-save-button", "n_clicks"),
    State("cfg-mqtt-enabled", "value"),
    State("cfg-mqtt-host", "value"),
    State("cfg-mqtt-port", "value"),
    State("cfg-mqtt-username", "value"),
    State("cfg-mqtt-password", "value"),
    State("cfg-mqtt-client-id", "value"),
    State("cfg-mqtt-measurement-topic", "value"),
    State("cfg-mqtt-channel-key", "value"),
    State("cfg-mqtt-duty-topic", "value"),
    State("cfg-mqtt-publish-enabled", "value"),
    State("cfg-mqtt-subscribe-meas", "value"),
    State("cfg-mqtt-publish-interval", "value"),
    State("cfg-mqtt-keepalive", "value"),
    State("cfg-mqtt-qos", "value"),
    State("cfg-mqtt-retain", "value"),
    State("cfg-irr-min", "value"),
    State("cfg-irr-max", "value"),
    State("cfg-irr-initial", "value"),
    State("cfg-irr-reference", "value"),
    State("cfg-irr-step", "value"),
    State("cfg-irr-voc-log", "value"),
    State("cfg-pv-isc", "value"),
    State("cfg-pv-voc", "value"),
    State("cfg-pv-imp", "value"),
    State("cfg-pv-vmp", "value"),
    State("cfg-pv-samples", "value"),
    State("cfg-load-vnom", "value"),
    State("cfg-load-pnom", "value"),
    State("cfg-load-hot-cold", "value"),
    State("cfg-load-tau", "value"),
    State("cfg-load-exp", "value"),
    State("cfg-pwm-min", "value"),
    State("cfg-pwm-max", "value"),
    State("cfg-pwm-initial", "value"),
    prevent_initial_call=True,
)
def save_configuration(
    n_clicks,
    mqtt_enabled,
    mqtt_host,
    mqtt_port,
    mqtt_username,
    mqtt_password,
    mqtt_client_id,
    mqtt_measurement_topic,
    mqtt_channel_key,
    mqtt_duty_topic,
    mqtt_publish_enabled,
    mqtt_subscribe_meas,
    mqtt_publish_interval,
    mqtt_keepalive,
    mqtt_qos,
    mqtt_retain,
    irr_min,
    irr_max,
    irr_initial,
    irr_reference,
    irr_step,
    irr_voc_log,
    pv_isc,
    pv_voc,
    pv_imp,
    pv_vmp,
    pv_samples,
    load_vnom,
    load_pnom,
    load_hot_cold,
    load_tau,
    load_exp,
    pwm_min,
    pwm_max,
    pwm_initial,
):
    try:
        if irr_min >= irr_max:
            return "Błąd: min nasłonecznienia musi być mniejsze od max."
        if not (irr_min <= irr_initial <= irr_max):
            return "Błąd: wartość początkowa nasłonecznienia musi mieścić się w zakresie min/max."
        if pwm_min > pwm_max:
            return "Błąd: duty_min musi być <= duty_max."
        if not (pwm_min <= pwm_initial <= pwm_max):
            return "Błąd: duty początkowe musi mieścić się w zakresie min/max."

        updated_cfg: Dict[str, Any] = {
            "mqtt": {
                "enabled": mqtt_enabled == "true",
                "host": str(mqtt_host or "127.0.0.1"),
                "port": int(mqtt_port),
                "username": str(mqtt_username or ""),
                "password": str(mqtt_password or ""),
                "client_id": str(mqtt_client_id or "pv-mppt-simulator"),
                "measurement_topic": str(mqtt_measurement_topic or "pv/sim/measurement"),
                "measurement_channel_key": str(mqtt_channel_key or "channel_1"),
                "duty_topic": str(mqtt_duty_topic or "pv/sim/set_duty"),
                "publish_simulated_measurements": mqtt_publish_enabled == "true",
                "subscribe_measurement_topic": mqtt_subscribe_meas == "true",
                "publish_interval_sec": float(mqtt_publish_interval),
                "keepalive_sec": int(mqtt_keepalive),
                "qos": int(mqtt_qos),
                "retain_measurements": mqtt_retain == "true",
            },
            "irradiance": {
                "min": float(irr_min),
                "max": float(irr_max),
                "initial": float(irr_initial),
                "reference": float(irr_reference),
                "slider_step": float(irr_step),
                "voc_log_coeff": float(irr_voc_log),
            },
            "pv": {
                "isc_ref": float(pv_isc),
                "voc_ref": float(pv_voc),
                "imp_ref": float(pv_imp),
                "vmp_ref": float(pv_vmp),
                "curve_samples": int(pv_samples),
            },
            "load": {
                "lamp_rated_voltage": float(load_vnom),
                "lamp_rated_power": float(load_pnom),
                "hot_to_cold_ratio": float(load_hot_cold),
                "thermal_tau_sec": float(load_tau),
                "thermal_power_exponent": float(load_exp),
            },
            "pwm": {
                "duty_min": float(pwm_min),
                "duty_max": float(pwm_max),
                "duty_initial": float(pwm_initial),
            },
        }
        config_store.save(updated_cfg)
        engine.set_irradiance(float(irr_initial))
        engine.set_manual_duty(float(pwm_initial))

        mqtt_mode = "publikuje" if updated_cfg["mqtt"]["publish_simulated_measurements"] else "nie publikuje"
        subscribe_mode = "subskrybuje" if updated_cfg["mqtt"]["subscribe_measurement_topic"] else "nie subskrybuje"
        return (
            f"Zapisano konfigurację. MQTT {mqtt_mode} pomiary i {subscribe_mode} topicu pomiarowego. "
            f"Plik konfiguracyjny został zaktualizowany."
        )
    except Exception as exc:
        return f"Błąd zapisu konfiguracji: {exc}"
