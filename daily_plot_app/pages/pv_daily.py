from __future__ import annotations

import os
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
from plotly.subplots import make_subplots

import dash
from dash import Input, Output, State, callback, dcc, html
from dash.exceptions import PreventUpdate

dash.register_page(__name__, path="/pv-daily", name="PV Daily")

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

CH_URL = os.getenv("PV_CH_URL", "http://localhost:8123")
CH_DATABASE = os.getenv("PV_CH_DATABASE", "prod_archives")
CH_USER = os.getenv("PV_CH_USER", "default")
CH_PASSWORD = os.getenv("PV_CH_PASSWORD", "")

DEFAULT_STEP_SEC = 300
DEFAULT_LOOKBACK_DAYS = 14
DEFAULT_IRR_TOL = 25.0
MIN_HISTORY_SAMPLES = 8
HTTP_TIMEOUT_SEC = 10

# Map panel -> irradiance tag used for comparison band
IRRADIANCE_BY_PANEL: Dict[str, str] = {
    "sensors/mppt_10": "sensors/irr_B_S/watt_dav",
    "sensors/mppt_20": "sensors/irr_B_S/watt_dav",
    "sensors/mppt_21": "sensors/irr_B_S/watt_dav",
    "sensors/mppt_22": "sensors/irr_B_S/watt_dav",
    "sensors/mppt_31": "sensors/irr_C_S/watt_dav",
    "sensors/mppt_32": "sensors/irr_C_S/watt_dav",
    "sensors/mppt_33": "sensors/irr_C_S/watt_dav",
    "sensors/mppt_34": "sensors/irr_C_S/watt_dav",
    "sensors/mppt_35": "sensors/irr_C_S/watt_dav",
    "sensors/mppt_36": "sensors/irr_C_S/watt_dav",
    "sensors/mppt_37": "sensors/irr_C_S/watt_dav",
    "sensors/mppt_38": "sensors/irr_C_S/watt_dav",
    # add the rest
}

IRRADIANCE_GROUP_RULES = [
    (range(10, 22), "sensors/irr_B_S/watt_dav"),
    (range(30, 40), "sensors/irr_C_S/watt_dav"),
]

# -----------------------------------------------------------------------------
# HTTP ClickHouse helpers
# -----------------------------------------------------------------------------

_session = requests.Session()


def q(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def ch_query_df(sql: str) -> pd.DataFrame:
    sql = sql.rstrip().rstrip(";") + " FORMAT JSON"
    params = {
        "database": CH_DATABASE,
        "user": CH_USER,
        "password": CH_PASSWORD,
        "wait_end_of_query": "1",
    }
    resp = _session.post(
        CH_URL,
        params=params,
        data=sql.encode("utf-8"),
        timeout=HTTP_TIMEOUT_SEC,
        headers={"Content-Type": "text/plain; charset=utf-8"},
    )
    resp.raise_for_status()
    payload = resp.json()

    rows = payload.get("data", [])
    meta = payload.get("meta", [])
    if not rows and meta:
        cols = [m["name"] for m in meta]
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows)

    # best-effort datetime conversion
    for c in df.columns:
        if c == "dt":
            try:
                df[c] = pd.to_datetime(df[c])
            except Exception:
                pass

    return df


# -----------------------------------------------------------------------------
# Tag / panel discovery
# -----------------------------------------------------------------------------

def panel_sort_key(panel_prefix: str):
    try:
        n = int(panel_prefix.split("_")[-1])
        return (0, n)
    except Exception:
        return (1, panel_prefix)


def discover_panels() -> List[str]:
    sql = """
    SELECT tag
    FROM tags_registry
    WHERE tag LIKE 'sensors/mppt_%/pv_power'
    ORDER BY tag
    """
    df = ch_query_df(sql)
    if df.empty:
        return []

    panels = []
    for tag in df["tag"].astype(str).tolist():
        parts = tag.split("/")
        if len(parts) >= 3:
            panels.append("/".join(parts[:2]))

    return sorted(set(panels), key=panel_sort_key)


def resolve_irradiance_tag(panel_prefix: str) -> Optional[str]:
    if panel_prefix in IRRADIANCE_BY_PANEL:
        return IRRADIANCE_BY_PANEL[panel_prefix]

    try:
        n = int(panel_prefix.split("_")[-1])
        for rng, irr_tag in IRRADIANCE_GROUP_RULES:
            if n in rng:
                return irr_tag
    except Exception:
        pass

    return None


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def fetch_aggregated_tags(
    tags: List[str],
    start_day: date,
    range_days: int,
    step_sec: int,
) -> pd.DataFrame:
    if not tags:
        return pd.DataFrame(columns=["tag", "ts", "dt", "avg_val"])

    tags_sql = ", ".join(q(t) for t in tags)
    start_day_str = start_day.isoformat()

    sql = f"""
    WITH
        {int(step_sec)} AS step,
        toDate({q(start_day_str)}) AS target_date,
        {int(range_days)} AS range_days,
        toUInt32(toUnixTimestamp(target_date)) AS ts_from,
        toUInt32(toUnixTimestamp(target_date + range_days)) AS ts_to
    SELECT
        tr.tag,
        agg.bucket_ts AS ts,
        toDateTime(agg.bucket_ts) AS dt,
        agg.avg_val
    FROM
    (
        SELECT
            dl.ids,
            intDiv(dl.ts, step) * step AS bucket_ts,
            avg(dl.val) AS avg_val
        FROM all_archives AS dl
        INNER JOIN
        (
            SELECT ids
            FROM tags_registry
            WHERE tag IN ({tags_sql})
        ) AS f
            ON dl.ids = f.ids
        WHERE
            dl.ts >= ts_from
            AND dl.ts < ts_to
            AND dl.d >= target_date
            AND dl.d < (target_date + range_days)
        GROUP BY
            dl.ids,
            bucket_ts
    ) AS agg
    LEFT JOIN tags_registry AS tr
        ON agg.ids = tr.ids
    ORDER BY tr.tag, ts
    """
    return ch_query_df(sql)


def fetch_day_panel_data(panel_prefix: str, panel_day: date, step_sec: int) -> pd.DataFrame:
    irr_tag = resolve_irradiance_tag(panel_prefix)

    tags = [
        f"{panel_prefix}/pv_power",
        f"{panel_prefix}/pv_v",
        f"{panel_prefix}/pv_i",
    ]
    if irr_tag:
        tags.append(irr_tag)

    df = fetch_aggregated_tags(tags=tags, start_day=panel_day, range_days=1, step_sec=step_sec)
    if df.empty:
        return pd.DataFrame()

    piv = (
        df.pivot_table(index=["ts", "dt"], columns="tag", values="avg_val", aggfunc="first")
        .reset_index()
        .sort_values("ts")
    )

    col_map = {
        f"{panel_prefix}/pv_power": "pv_power",
        f"{panel_prefix}/pv_v": "pv_v",
        f"{panel_prefix}/pv_i": "pv_i",
    }
    if irr_tag:
        col_map[irr_tag] = "irr"

    piv = piv.rename(columns=col_map)

    for c in ["pv_power", "pv_v", "pv_i", "irr"]:
        if c not in piv.columns:
            piv[c] = np.nan

    base_ts = int(datetime.combine(panel_day, datetime.min.time()).timestamp())
    piv["bucket_idx"] = ((piv["ts"].astype(int) - base_ts) // step_sec).astype(int)

    return piv[["ts", "dt", "bucket_idx", "pv_power", "pv_v", "pv_i", "irr"]]


def fetch_history_for_band(
    panel_prefix: str,
    panel_day: date,
    step_sec: int,
    lookback_days: int,
) -> pd.DataFrame:
    hist_start = panel_day - timedelta(days=lookback_days)
    irr_tag = resolve_irradiance_tag(panel_prefix)
    if not irr_tag:
        return pd.DataFrame()

    tags = [f"{panel_prefix}/pv_power", irr_tag]
    df = fetch_aggregated_tags(tags=tags, start_day=hist_start, range_days=lookback_days, step_sec=step_sec)
    if df.empty:
        return pd.DataFrame()

    piv = (
        df.pivot_table(index=["ts", "dt"], columns="tag", values="avg_val", aggfunc="first")
        .reset_index()
        .sort_values("ts")
    )

    piv = piv.rename(columns={f"{panel_prefix}/pv_power": "pv_power", irr_tag: "irr"})
    if "pv_power" not in piv.columns or "irr" not in piv.columns:
        return pd.DataFrame()

    piv["dt"] = pd.to_datetime(piv["dt"])
    piv["day"] = piv["dt"].dt.date
    piv["bucket_idx"] = ((piv["dt"].dt.hour * 3600 + piv["dt"].dt.minute * 60 + piv["dt"].dt.second) // step_sec).astype(int)
    piv = piv[piv["day"] < panel_day].copy()

    return piv[["day", "ts", "dt", "bucket_idx", "pv_power", "irr"]]


# -----------------------------------------------------------------------------
# Calculations
# -----------------------------------------------------------------------------

def compute_irradiance_matched_band(
    current_df: pd.DataFrame,
    hist_df: pd.DataFrame,
    irr_tolerance: float,
) -> pd.DataFrame:
    out = current_df.copy()
    out["band_q10"] = np.nan
    out["band_q50"] = np.nan
    out["band_q90"] = np.nan
    out["band_n"] = 0

    if out.empty or hist_df.empty:
        return out

    hist_valid = hist_df.dropna(subset=["pv_power", "irr"]).copy()
    if hist_valid.empty:
        return out

    grouped = {k: g.copy() for k, g in hist_valid.groupby("bucket_idx")}

    for idx, row in out.iterrows():
        b = int(row["bucket_idx"])
        irr_now = row["irr"]
        if pd.isna(irr_now) or b not in grouped:
            continue

        candidates = grouped[b]
        chosen = pd.DataFrame()

        for factor in (1.0, 2.0, 4.0):
            tol = irr_tolerance * factor
            subset = candidates[np.abs(candidates["irr"] - irr_now) <= tol]
            if len(subset) >= MIN_HISTORY_SAMPLES:
                chosen = subset
                break

        if chosen.empty:
            tmp = candidates.assign(irr_diff=np.abs(candidates["irr"] - irr_now))
            chosen = tmp.sort_values("irr_diff").head(MIN_HISTORY_SAMPLES)

        if chosen.empty:
            continue

        vals = chosen["pv_power"].dropna().to_numpy()
        if len(vals) == 0:
            continue

        out.at[idx, "band_q10"] = float(np.quantile(vals, 0.10))
        out.at[idx, "band_q50"] = float(np.quantile(vals, 0.50))
        out.at[idx, "band_q90"] = float(np.quantile(vals, 0.90))
        out.at[idx, "band_n"] = int(len(vals))

    return out


def compute_daily_metrics(df: pd.DataFrame, step_sec: int) -> Dict[str, float]:
    if df.empty:
        return {}

    x = df.copy()
    power = x["pv_power"].astype(float)
    irr = x["irr"].astype(float)
    v = x["pv_v"].astype(float)
    i = x["pv_i"].astype(float)

    valid_power = power.dropna()
    if valid_power.empty:
        return {}

    energy_wh = float(valid_power.sum() * step_sec / 3600.0)
    peak_idx = int(valid_power.idxmax())
    peak_power_w = float(valid_power.max())
    peak_time = pd.to_datetime(x.loc[peak_idx, "dt"]).strftime("%H:%M")

    valid_irr = irr.dropna()
    insolation_whm2 = float(valid_irr.sum() * step_sec / 3600.0) if not valid_irr.empty else np.nan
    energy_per_insolation = energy_wh / insolation_whm2 if insolation_whm2 and insolation_whm2 > 0 else np.nan

    pair = x[["pv_power", "irr"]].dropna()
    corr = float(pair["pv_power"].corr(pair["irr"])) if len(pair) >= 3 else np.nan

    vi = (v * i).replace(0, np.nan)
    ratio = (power / vi).replace([np.inf, -np.inf], np.nan).dropna()
    mppt_ratio_med = float(ratio.median()) if not ratio.empty else np.nan

    inside = below = above = np.nan
    mask = x[["pv_power", "band_q10", "band_q90"]].notna().all(axis=1)
    if mask.any():
        comp = x.loc[mask, ["pv_power", "band_q10", "band_q90"]]
        inside = float(((comp["pv_power"] >= comp["band_q10"]) & (comp["pv_power"] <= comp["band_q90"])).mean() * 100.0)
        below = float((comp["pv_power"] < comp["band_q10"]).mean() * 100.0)
        above = float((comp["pv_power"] > comp["band_q90"]).mean() * 100.0)

    pmax = x.loc[peak_idx]
    return {
        "energy_wh": energy_wh,
        "peak_power_w": peak_power_w,
        "peak_time": peak_time,
        "insolation_whm2": insolation_whm2,
        "energy_per_insolation": energy_per_insolation,
        "power_irr_corr": corr,
        "mppt_ratio_med": mppt_ratio_med,
        "band_inside_pct": inside,
        "band_below_pct": below,
        "band_above_pct": above,
        "pv_v_at_best": float(pmax["pv_v"]) if pd.notna(pmax["pv_v"]) else np.nan,
        "pv_i_at_best": float(pmax["pv_i"]) if pd.notna(pmax["pv_i"]) else np.nan,
    }


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------

def build_figure(df: pd.DataFrame, panel_prefix: str, panel_day: date) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=False,
        vertical_spacing=0.10,
        row_heights=[0.62, 0.38],
        specs=[[{"secondary_y": True}], [{}]],
        subplot_titles=(
            f"{panel_prefix} — {panel_day.isoformat()}",
            "IV / MPPT trace",
        ),
    )

    if df.empty:
        fig.update_layout(template="plotly_white", height=820, title="No data")
        return fig

    t = pd.to_datetime(df["dt"])

    if df["band_q10"].notna().any():
        fig.add_trace(
            go.Scatter(x=t, y=df["band_q90"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"),
            row=1, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=t, y=df["band_q10"], mode="lines", line=dict(width=0), fill="tonexty", name="Band q10-q90"),
            row=1, col=1, secondary_y=False
        )
        fig.add_trace(
            go.Scatter(x=t, y=df["band_q50"], mode="lines", name="Band median", line=dict(dash="dot")),
            row=1, col=1, secondary_y=False
        )

    fig.add_trace(
        go.Scatter(x=t, y=df["pv_power"], mode="lines+markers", name="PV power"),
        row=1, col=1, secondary_y=False
    )

    if df["irr"].notna().any():
        fig.add_trace(
            go.Scatter(x=t, y=df["irr"], mode="lines", name="Irradiance", line=dict(width=1, color="firebrick")),
            row=1, col=1, secondary_y=True
        )

    iv = df.dropna(subset=["pv_v", "pv_i"]).copy()
    # if not iv.empty:
    #     iv["time_str"] = pd.to_datetime(iv["dt"]).dt.strftime("%H:%M")
    #     fig.add_trace(
    #         go.Scatter(
    #             x=iv["pv_v"],
    #             y=iv["pv_i"],
    #             mode="markers+lines",
    #             name="MPPT points",
    #             marker=dict(size=7, color=iv["ts"], showscale=True, colorbar=dict(title="time")),
    #             text=iv["time_str"],
    #             customdata=np.stack([iv["pv_power"].fillna(np.nan), iv["time_str"]], axis=-1),
    #             hovertemplate=(
    #                 "t=%{customdata[1]}<br>"
    #                 "V=%{x:.2f} V<br>"
    #                 "I=%{y:.2f} A<br>"
    #                 "P=%{customdata[0]:.2f} W<extra></extra>"
    #             ),
    #         ),
    #         row=2, col=1
    #     )
    if not iv.empty:
        iv["time_str"] = pd.to_datetime(iv["dt"]).dt.strftime("%H:%M")

        # build readable colorbar ticks from actual timestamps
        ts_min = int(iv["ts"].min())
        ts_max = int(iv["ts"].max())

        if ts_max > ts_min:
            tick_count = min(6, len(iv))
            tickvals = np.linspace(ts_min, ts_max, tick_count)
            ticktext = [datetime.fromtimestamp(float(v)).strftime("%H:%M") for v in tickvals]
        else:
            tickvals = [ts_min]
            ticktext = [datetime.fromtimestamp(ts_min).strftime("%H:%M")]

        fig.add_trace(
            go.Scatter(
                x=iv["pv_i"],  # current on X
                y=iv["pv_v"],  # voltage on Y  => V as a function of I
                mode="markers",  # dots only
                name="MPPT points",
                marker=dict(
                    size=8,
                    color=iv["ts"],
                    showscale=True,
                    colorbar=dict(
                        title="Time",
                        tickvals=tickvals,
                        ticktext=ticktext,
                    ),
                ),
                text=iv["time_str"],
                customdata=np.stack(
                    [
                        iv["pv_power"].fillna(np.nan),
                        iv["time_str"],
                        iv["pv_i"].fillna(np.nan),
                        iv["pv_v"].fillna(np.nan),
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "t=%{customdata[1]}<br>"
                    "I=%{x:.2f} A<br>"
                    "V=%{y:.2f} V<br>"
                    "P=%{customdata[0]:.2f} W<extra></extra>"
                ),
            ),
            row=2, col=1
        )

    fig.update_yaxes(title_text="Power [W]", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Irradiance [W/m²]", row=1, col=1, secondary_y=True)
    fig.update_xaxes(title_text="Time", row=1, col=1)
    fig.update_yaxes(title_text="Voltage [V]", row=2, col=1)
    fig.update_xaxes(title_text="Current [A]", row=2, col=1)

    fig.update_layout(
        template="plotly_white",
        height=820,
        margin=dict(l=40, r=40, t=70, b=40),
        legend=dict(orientation="h", y=1.02, x=0),
    )
    return fig


# -----------------------------------------------------------------------------
# Layout
# -----------------------------------------------------------------------------

layout = html.Div(
    [
        dcc.Store(id="pv-panels-store"),
        dcc.Interval(id="pv-init", interval=200, max_intervals=1),

        html.H3("PV Daily analytics"),

        html.Div(
            [
                html.Div(
                    [
                        html.Label("Panel"),
                        dcc.Dropdown(id="pv-panel", options=[], value=None, clearable=False, persistence=True),
                    ],
                    style={"minWidth": "320px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label("Date"),
                        dcc.DatePickerSingle(
                            id="pv-date",
                            date=date.today().isoformat(),
                            display_format="YYYY-MM-DD",
                            persistence=True,
                        ),
                    ]
                ),
                html.Div([html.Label(""), html.Button("◀ Prev day", id="pv-prev", n_clicks=0)]),
                html.Div([html.Label(""), html.Button("Next day ▶", id="pv-next", n_clicks=0)]),
                html.Div(
                    [
                        html.Label("Step [s]"),
                        dcc.Input(id="pv-step", type="number", min=60, step=60, value=DEFAULT_STEP_SEC, persistence=True),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Lookback [days]"),
                        dcc.Input(id="pv-lookback", type="number", min=7, step=1, value=DEFAULT_LOOKBACK_DAYS, persistence=True),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Irr tol [W/m²]"),
                        dcc.Input(id="pv-irr-tol", type="number", min=5, step=5, value=DEFAULT_IRR_TOL, persistence=True),
                    ]
                ),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "end", "marginBottom": "12px"},
        ),

        html.Div(id="pv-status", style={"marginBottom": "10px", "fontStyle": "italic"}),
        dcc.Graph(id="pv-graph"),
        html.H4("Daily KPIs"),
        html.Div(id="pv-kpis"),
    ]
)


# -----------------------------------------------------------------------------
# Callbacks
# -----------------------------------------------------------------------------

@callback(
    Output("pv-panels-store", "data"),
    Input("pv-init", "n_intervals"),
    prevent_initial_call=False,
)
def load_panels(_n):
    try:
        return {"panels": discover_panels()}
    except Exception as e:
        return {"panels": [], "error": str(e)}


@callback(
    Output("pv-panel", "options"),
    Output("pv-panel", "value"),
    Input("pv-panels-store", "data"),
    State("pv-panel", "value"),
    prevent_initial_call=False,
)
def set_panel_options(store, current_value):
    store = store or {}
    panels = store.get("panels", [])

    opts = [{"label": p, "value": p} for p in panels]
    if not panels:
        return opts, None

    value = current_value if current_value in panels else panels[0]
    return opts, value


@callback(
    Output("pv-date", "date"),
    Input("pv-prev", "n_clicks"),
    Input("pv-next", "n_clicks"),
    State("pv-date", "date"),
    prevent_initial_call=True,
)
def shift_day(prev_clicks, next_clicks, current_date):
    if not current_date:
        raise PreventUpdate

    d = date.fromisoformat(str(current_date))
    trig = dash.ctx.triggered_id
    if trig == "pv-prev":
        return (d - timedelta(days=1)).isoformat()
    if trig == "pv-next":
        return (d + timedelta(days=1)).isoformat()
    raise PreventUpdate


def kpi_card(label: str, value: str):
    return html.Div(
        [
            html.Div(label, style={"fontSize": "0.9rem", "opacity": 0.7}),
            html.Div(value, style={"fontSize": "1.2rem", "fontWeight": 600}),
        ],
        style={
            "border": "1px solid #ddd",
            "borderRadius": "8px",
            "padding": "10px 12px",
            "minWidth": "180px",
        },
    )


@callback(
    Output("pv-graph", "figure"),
    Output("pv-kpis", "children"),
    Output("pv-status", "children"),
    Input("pv-panel", "value"),
    Input("pv-date", "date"),
    Input("pv-step", "value"),
    Input("pv-lookback", "value"),
    Input("pv-irr-tol", "value"),
)
def render_page(panel_prefix, panel_date, step_sec, lookback_days, irr_tolerance):
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_white", height=600)

    if not panel_prefix or not panel_date:
        return empty_fig, "Select panel and date.", "Waiting for selection."

    try:
        panel_day = date.fromisoformat(str(panel_date))
        step_sec = int(step_sec or DEFAULT_STEP_SEC)
        lookback_days = int(lookback_days or DEFAULT_LOOKBACK_DAYS)
        irr_tolerance = float(irr_tolerance or DEFAULT_IRR_TOL)

        day_df = fetch_day_panel_data(panel_prefix, panel_day, step_sec)
        if day_df.empty:
            fig = build_figure(day_df, panel_prefix, panel_day)
            return fig, "No KPI data.", f"No data for {panel_prefix} on {panel_day.isoformat()}."

        hist_df = fetch_history_for_band(panel_prefix, panel_day, step_sec, lookback_days)
        merged = compute_irradiance_matched_band(day_df, hist_df, irr_tolerance)

        fig = build_figure(merged, panel_prefix, panel_day)
        m = compute_daily_metrics(merged, step_sec)

        cards = []
        if m:
            cards = html.Div(
                [
                    kpi_card("Energy", f"{m['energy_wh']:.1f} Wh"),
                    kpi_card("Peak power", f"{m['peak_power_w']:.1f} W @ {m['peak_time']}"),
                    kpi_card("Insolation", "n/a" if np.isnan(m["insolation_whm2"]) else f"{m['insolation_whm2']:.1f} Wh/m²"),
                    kpi_card("Energy / insolation", "n/a" if np.isnan(m["energy_per_insolation"]) else f"{m['energy_per_insolation']:.4f}"),
                    kpi_card("Power↔Irr corr", "n/a" if np.isnan(m["power_irr_corr"]) else f"{m['power_irr_corr']:.3f}"),
                    kpi_card("P / (V·I) median", "n/a" if np.isnan(m["mppt_ratio_med"]) else f"{m['mppt_ratio_med']:.3f}"),
                    kpi_card("Inside band", "n/a" if np.isnan(m["band_inside_pct"]) else f"{m['band_inside_pct']:.1f}%"),
                    kpi_card(
                        "Below / Above band",
                        "n/a" if np.isnan(m["band_below_pct"]) or np.isnan(m["band_above_pct"])
                        else f"{m['band_below_pct']:.1f}% / {m['band_above_pct']:.1f}%"
                    ),
                    kpi_card(
                        "Best MPPT point",
                        "n/a" if np.isnan(m["pv_v_at_best"]) or np.isnan(m["pv_i_at_best"])
                        else f"{m['pv_v_at_best']:.2f} V / {m['pv_i_at_best']:.2f} A"
                    ),
                ],
                style={"display": "flex", "gap": "10px", "flexWrap": "wrap"},
            )
        else:
            cards = "No KPI data."

        status = " | ".join(
            [
                f"Panel: {panel_prefix}",
                f"Date: {panel_day.isoformat()}",
                f"Step: {step_sec}s",
                f"Lookback: {lookback_days}d",
                f"Irr tag: {resolve_irradiance_tag(panel_prefix) or 'not mapped'}",
                f"History rows: {len(hist_df)}",
            ]
        )

        return fig, cards, status

    except Exception as e:
        empty_fig.update_layout(title="Error")
        return empty_fig, "Error while calculating KPIs.", f"Error: {e}"