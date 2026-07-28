from __future__ import annotations

from datetime import date, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

import dash
from dash import Input, Output, State, callback, dcc, html

from pages.pv_daily import (
    DEFAULT_STEP_SEC,
    discover_panels,
    fetch_aggregated_tags,
    panel_sort_key,
    resolve_irradiance_tag,
)

dash.register_page(__name__, path="/fleet-future", name="Fleet / Multi-day")

DAYLIGHT_IRR_THRESHOLD = 50.0
GOOD_COVERAGE_PCT = 85.0
PARTIAL_COVERAGE_PCT = 40.0


def classify_daylight(irr_values: pd.Series, step_sec: int) -> Tuple[str, float, float]:
    irr = irr_values.dropna().astype(float)
    daylight = irr[irr >= DAYLIGHT_IRR_THRESHOLD]
    if daylight.empty:
        return "No sun", np.nan, np.nan

    insolation_whm2 = float(daylight.sum() * step_sec / 3600.0)
    peak_irr = float(daylight.max())
    ramp_score = float(daylight.diff().abs().mean() / max(peak_irr, 1.0))

    if insolation_whm2 < 300 or peak_irr < 180:
        label = "Low sun"
    elif ramp_score < 0.08:
        label = "Sunny"
    elif ramp_score < 0.16:
        label = "Mixed"
    else:
        label = "Cloudy"

    return label, insolation_whm2, ramp_score


def quality_label(coverage_pct: float, energy_wh: float) -> str:
    if np.isnan(energy_wh):
        return "Missing"
    if coverage_pct >= GOOD_COVERAGE_PCT:
        return "Good"
    if coverage_pct >= PARTIAL_COVERAGE_PCT:
        return "Partial"
    return "Sparse"


def build_overview_metrics(
    panels: List[str],
    start_day: date,
    end_day: date,
    step_sec: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not panels or start_day > end_day:
        return pd.DataFrame(), pd.DataFrame()

    range_days = (end_day - start_day).days + 1
    irr_by_panel = {panel: resolve_irradiance_tag(panel) for panel in panels}

    tags = [f"{panel}/pv_power" for panel in panels]
    tags += sorted({irr for irr in irr_by_panel.values() if irr})
    tags = sorted(set(tags))

    raw = fetch_aggregated_tags(tags=tags, start_day=start_day, range_days=range_days, step_sec=step_sec)
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    raw["dt"] = pd.to_datetime(raw["dt"])
    raw["day"] = raw["dt"].dt.date
    raw = raw[(raw["day"] >= start_day) & (raw["day"] <= end_day)].copy()
    if raw.empty:
        return pd.DataFrame(), pd.DataFrame()

    piv = (
        raw.pivot_table(index=["ts", "dt", "day"], columns="tag", values="avg_val", aggfunc="first")
        .reset_index()
        .sort_values("ts")
    )

    day_summaries = []
    for day_value, day_frame in piv.groupby("day"):
        labels = []
        insolations = []
        ramps = []
        for irr_tag in sorted({irr for irr in irr_by_panel.values() if irr}):
            if irr_tag not in day_frame.columns:
                continue
            label, insolation_whm2, ramp_score = classify_daylight(day_frame[irr_tag], step_sec)
            if not np.isnan(insolation_whm2):
                labels.append(label)
                insolations.append(insolation_whm2)
                ramps.append(ramp_score)

        if not labels:
            day_label = "No sun"
            mean_insolation = np.nan
            mean_ramp = np.nan
        else:
            if len(set(labels)) == 1:
                day_label = labels[0]
            elif "Cloudy" in labels:
                day_label = "Mixed"
            else:
                day_label = labels[0]
            mean_insolation = float(np.nanmean(insolations))
            mean_ramp = float(np.nanmean(ramps))

        day_summaries.append(
            {
                "day": day_value,
                "day_condition": day_label,
                "day_insolation_whm2": mean_insolation,
                "day_ramp_score": mean_ramp,
            }
        )

    day_summary_df = pd.DataFrame(day_summaries)

    metric_rows = []
    for panel in panels:
        power_tag = f"{panel}/pv_power"
        irr_tag = irr_by_panel.get(panel)

        for day_value, day_frame in piv.groupby("day"):
            power = day_frame[power_tag] if power_tag in day_frame.columns else pd.Series(dtype=float)
            irr = day_frame[irr_tag] if irr_tag and irr_tag in day_frame.columns else pd.Series(dtype=float)

            energy_wh = float(power.dropna().sum() * step_sec / 3600.0) if not power.empty and power.notna().any() else np.nan
            peak_power_w = float(power.max()) if not power.empty and power.notna().any() else np.nan
            insolation_whm2 = float(irr.dropna().sum() * step_sec / 3600.0) if not irr.empty and irr.notna().any() else np.nan
            perf_ratio = energy_wh / insolation_whm2 if insolation_whm2 and insolation_whm2 > 0 and not np.isnan(energy_wh) else np.nan

            daylight_mask = irr >= DAYLIGHT_IRR_THRESHOLD if not irr.empty else pd.Series(False, index=day_frame.index)
            daylight_buckets = int(daylight_mask.sum()) if len(daylight_mask) else 0
            observed_daylight = int((power.notna() & daylight_mask).sum()) if not power.empty else 0
            coverage_pct = float(observed_daylight * 100.0 / daylight_buckets) if daylight_buckets else np.nan

            metric_rows.append(
                {
                    "day": day_value,
                    "panel": panel,
                    "energy_wh": energy_wh,
                    "peak_power_w": peak_power_w,
                    "insolation_whm2": insolation_whm2,
                    "perf_ratio": perf_ratio,
                    "coverage_pct": coverage_pct,
                    "quality": quality_label(coverage_pct, energy_wh),
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    if metrics_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    metrics_df["day_perf_median"] = metrics_df.groupby("day")["perf_ratio"].transform("median")
    metrics_df["perf_rel"] = metrics_df["perf_ratio"] / metrics_df["day_perf_median"]
    metrics_df.loc[~np.isfinite(metrics_df["perf_rel"]), "perf_rel"] = np.nan

    day_quality = (
        metrics_df.assign(good=metrics_df["coverage_pct"] >= GOOD_COVERAGE_PCT)
        .groupby("day")
        .agg(
            good_panels=("good", "sum"),
            total_panels=("panel", "count"),
            missing_panels=("energy_wh", lambda s: int(s.isna().sum())),
        )
        .reset_index()
    )

    day_summary_df = day_summary_df.merge(day_quality, on="day", how="left")
    day_summary_df["good_panels"] = day_summary_df["good_panels"].fillna(0).astype(int)
    day_summary_df["total_panels"] = day_summary_df["total_panels"].fillna(0).astype(int)
    day_summary_df["missing_panels"] = day_summary_df["missing_panels"].fillna(0).astype(int)

    return metrics_df, day_summary_df


def condition_badge(label: str):
    colors = {
        "Sunny": "#2ca02c",
        "Mixed": "#ffbf00",
        "Cloudy": "#7f7f7f",
        "Low sun": "#6baed6",
        "No sun": "#d9d9d9",
    }
    return html.Span(
        label,
        style={
            "backgroundColor": colors.get(label, "#eee"),
            "padding": "2px 8px",
            "borderRadius": "12px",
            "fontSize": "0.85rem",
            "display": "inline-block",
        },
    )


def ratio_to_color(value: float, coverage_pct: float) -> str:
    if np.isnan(value):
        return "#f3f3f3"

    x = max(0.7, min(1.3, value))
    if x <= 1.0:
        t = (x - 0.7) / 0.3
        r1, g1, b1 = (215, 48, 39)
        r2, g2, b2 = (254, 224, 139)
    else:
        t = (x - 1.0) / 0.3
        r1, g1, b1 = (254, 224, 139)
        r2, g2, b2 = (26, 152, 80)

    r = int(r1 + (r2 - r1) * t)
    g = int(g1 + (g2 - g1) * t)
    b = int(b1 + (b2 - b1) * t)

    if not np.isnan(coverage_pct) and coverage_pct < GOOD_COVERAGE_PCT:
        mix = 0.35 if coverage_pct >= PARTIAL_COVERAGE_PCT else 0.55
        r = int(r + (255 - r) * mix)
        g = int(g + (255 - g) * mix)
        b = int(b + (255 - b) * mix)

    return f"rgb({r},{g},{b})"


def build_heatmap(metrics_df: pd.DataFrame, day_summary_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(template="plotly_white", height=500, title="No data")

    if metrics_df.empty:
        return fig

    panels = sorted(metrics_df["panel"].unique().tolist(), key=panel_sort_key)
    days = sorted(metrics_df["day"].unique().tolist())

    perf_mat = metrics_df.pivot(index="day", columns="panel", values="perf_rel").reindex(index=days, columns=panels)
    cov_mat = metrics_df.pivot(index="day", columns="panel", values="coverage_pct").reindex(index=days, columns=panels)
    energy_mat = metrics_df.pivot(index="day", columns="panel", values="energy_wh").reindex(index=days, columns=panels)
    peak_mat = metrics_df.pivot(index="day", columns="panel", values="peak_power_w").reindex(index=days, columns=panels)

    condition_by_day = day_summary_df.set_index("day")["day_condition"].to_dict()
    insolation_by_day = day_summary_df.set_index("day")["day_insolation_whm2"].to_dict()

    text = []
    hover = []
    for day_value in days:
        row_text = []
        row_hover = []
        for panel in panels:
            rel = perf_mat.loc[day_value, panel]
            cov = cov_mat.loc[day_value, panel]
            energy = energy_mat.loc[day_value, panel]
            peak = peak_mat.loc[day_value, panel]

            row_text.append("—" if np.isnan(rel) else f"{rel:.2f}×<br>{cov:.0f}%")
            row_hover.append(
                "<br>".join(
                    [
                        f"Day: {day_value.isoformat()}",
                        f"Panel: {panel}",
                        f"Condition: {condition_by_day.get(day_value, 'n/a')}",
                        f"Rel perf: {'n/a' if np.isnan(rel) else f'{rel:.3f}×'}",
                        f"Coverage: {'n/a' if np.isnan(cov) else f'{cov:.1f}%'}",
                        f"Energy: {'n/a' if np.isnan(energy) else f'{energy:.1f} Wh'}",
                        f"Peak power: {'n/a' if np.isnan(peak) else f'{peak:.1f} W'}",
                        f"Irradiance: {'n/a' if np.isnan(insolation_by_day.get(day_value, np.nan)) else f'{insolation_by_day[day_value]:.1f} Wh/m²'}",
                    ]
                )
            )
        text.append(row_text)
        hover.append(row_hover)

    fig = go.Figure(
        data=[
            go.Heatmap(
                z=perf_mat.to_numpy(dtype=float),
                x=panels,
                y=[d.isoformat() for d in days],
                zmid=1.0,
                zmin=0.7,
                zmax=1.3,
                colorscale=[
                    [0.0, "#d73027"],
                    [0.5, "#fee08b"],
                    [1.0, "#1a9850"],
                ],
                colorbar=dict(title="Perf vs day median"),
                text=text,
                texttemplate="%{text}",
                textfont=dict(size=11),
                customdata=hover,
                hovertemplate="%{customdata}<extra></extra>",
                xgap=2,
                ygap=2,
            )
        ]
    )
    fig.update_layout(
        template="plotly_white",
        height=max(420, 90 + 30 * len(days)),
        margin=dict(l=40, r=40, t=60, b=40),
        title="Daily panel overview — color = normalized performance, text = ratio / daylight coverage",
    )
    fig.update_xaxes(title_text="Panel")
    fig.update_yaxes(title_text="Day", autorange="reversed")
    return fig


def build_overview_table(metrics_df: pd.DataFrame, day_summary_df: pd.DataFrame):
    if metrics_df.empty:
        return "No overview data."

    panels = sorted(metrics_df["panel"].unique().tolist(), key=panel_sort_key)
    summary_by_day = day_summary_df.set_index("day").to_dict("index")

    header = html.Thead(
        html.Tr(
            [
                html.Th("Day"),
                html.Th("Sky"),
                html.Th("Irr [Wh/m²]"),
                html.Th("Good panels"),
                html.Th("Missing"),
                *[html.Th(panel.split("/")[-1]) for panel in panels],
            ]
        )
    )

    body_rows = []
    for day_value in sorted(metrics_df["day"].unique().tolist()):
        day_metrics = metrics_df[metrics_df["day"] == day_value].set_index("panel")
        summary = summary_by_day.get(day_value, {})

        cells = [
            html.Td(day_value.isoformat(), style={"whiteSpace": "nowrap", "fontWeight": 600}),
            html.Td(condition_badge(summary.get("day_condition", "n/a"))),
            html.Td(
                "n/a"
                if np.isnan(summary.get("day_insolation_whm2", np.nan))
                else f"{summary['day_insolation_whm2']:.0f}"
            ),
            html.Td(f"{summary.get('good_panels', 0)}/{summary.get('total_panels', 0)}"),
            html.Td(str(summary.get("missing_panels", 0))),
        ]

        for panel in panels:
            if panel not in day_metrics.index:
                cells.append(html.Td("—", style={"backgroundColor": "#f3f3f3"}))
                continue

            row = day_metrics.loc[panel]
            rel = row["perf_rel"]
            cov = row["coverage_pct"]
            energy = row["energy_wh"]
            peak_power = row["peak_power_w"]
            bg = ratio_to_color(rel, cov)

            if np.isnan(rel):
                text = "—"
            else:
                text = f"{rel:.2f}× | {cov:.0f}%"

            title = " | ".join(
                [
                    f"{panel}",
                    f"Quality: {row['quality']}",
                    f"Energy: {'n/a' if np.isnan(energy) else f'{energy:.1f} Wh'}",
                    f"Peak: {'n/a' if np.isnan(peak_power) else f'{peak_power:.1f} W'}",
                ]
            )
            cells.append(
                html.Td(
                    text,
                    title=title,
                    style={
                        "backgroundColor": bg,
                        "whiteSpace": "nowrap",
                        "fontFamily": "monospace",
                        "textAlign": "center",
                    },
                )
            )

        body_rows.append(html.Tr(cells))

    return html.Div(
        [
            html.Div(
                "Cell text = relative performance vs same-day fleet median | daylight coverage. "
                "Colors: red lower, yellow near median, green higher; pale colors suggest partial/sparse coverage.",
                style={"marginBottom": "8px", "fontStyle": "italic"},
            ),
            html.Div(
                html.Table([header, html.Tbody(body_rows)], style={"borderCollapse": "collapse", "width": "100%"}),
                style={"overflowX": "auto", "border": "1px solid #ddd", "borderRadius": "8px", "padding": "6px"},
            ),
        ]
    )


layout = html.Div(
    [
        dcc.Store(id="fleet-panels-store"),
        dcc.Interval(id="fleet-init", interval=200, max_intervals=1),
        html.H3("Fleet / Multi-day overview"),
        html.P(
            "Use this view to scan many days at once. "
            "The heatmap/table compares each panel to the same-day fleet median after normalizing by irradiance, "
            "and also shows daylight data coverage."
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Label("Date range"),
                        dcc.DatePickerRange(
                            id="fleet-range",
                            start_date=(date.today() - timedelta(days=13)).isoformat(),
                            end_date=date.today().isoformat(),
                            display_format="YYYY-MM-DD",
                            persistence=True,
                        ),
                    ]
                ),
                html.Div(
                    [
                        html.Label("Panels"),
                        dcc.Dropdown(id="fleet-panels", options=[], value=[], multi=True, persistence=True),
                    ],
                    style={"minWidth": "440px", "flex": "1"},
                ),
                html.Div(
                    [
                        html.Label("Step [s]"),
                        dcc.Input(id="fleet-step", type="number", min=60, step=60, value=DEFAULT_STEP_SEC, persistence=True),
                    ]
                ),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "end", "marginBottom": "12px"},
        ),
        html.Div(id="fleet-status", style={"marginBottom": "10px", "fontStyle": "italic"}),
        dcc.Graph(id="fleet-heatmap"),
        html.H4("Daily table"),
        html.Div(id="fleet-table"),
    ],
    style={"padding": "8px"},
)


@callback(
    Output("fleet-panels-store", "data"),
    Input("fleet-init", "n_intervals"),
    prevent_initial_call=False,
)
def load_panels(_n):
    try:
        return {"panels": discover_panels()}
    except Exception as e:
        return {"panels": [], "error": str(e)}


@callback(
    Output("fleet-panels", "options"),
    Output("fleet-panels", "value"),
    Input("fleet-panels-store", "data"),
    State("fleet-panels", "value"),
    prevent_initial_call=False,
)
def set_panel_options(store, current_value):
    store = store or {}
    panels = store.get("panels", [])
    opts = [{"label": p, "value": p} for p in panels]
    if not panels:
        return opts, []

    selected = [p for p in (current_value or []) if p in panels]
    return opts, (selected or panels)


@callback(
    Output("fleet-heatmap", "figure"),
    Output("fleet-table", "children"),
    Output("fleet-status", "children"),
    Input("fleet-range", "start_date"),
    Input("fleet-range", "end_date"),
    Input("fleet-panels", "value"),
    Input("fleet-step", "value"),
)
def render_overview(start_date, end_date, panels, step_sec):
    empty_fig = go.Figure()
    empty_fig.update_layout(template="plotly_white", height=420)

    if not start_date or not end_date or not panels:
        return empty_fig, "Select range and panels.", "Waiting for selection."

    try:
        start_day = date.fromisoformat(str(start_date))
        end_day = date.fromisoformat(str(end_date))
        if start_day > end_day:
            return empty_fig, "Invalid date range.", "Start date must be <= end date."

        step_sec = int(step_sec or DEFAULT_STEP_SEC)
        metrics_df, day_summary_df = build_overview_metrics(
            panels=sorted(set(panels), key=panel_sort_key),
            start_day=start_day,
            end_day=end_day,
            step_sec=step_sec,
        )

        if metrics_df.empty:
            return empty_fig, "No overview data.", f"No data for selected range {start_day.isoformat()} .. {end_day.isoformat()}."

        fig = build_heatmap(metrics_df, day_summary_df)
        table = build_overview_table(metrics_df, day_summary_df)

        status = " | ".join(
            [
                f"Range: {start_day.isoformat()} .. {end_day.isoformat()}",
                f"Panels: {len(sorted(set(panels)))}",
                f"Days: {metrics_df['day'].nunique()}",
                f"Step: {step_sec}s",
            ]
        )
        return fig, table, status

    except Exception as e:
        empty_fig.update_layout(title="Error")
        return empty_fig, "Error while building overview.", f"Error: {e}"
