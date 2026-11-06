#!/usr/bin/env python3
"""Export I/V sweep characteristics (default panel: sensors/mppt_39) for a date range to Excel.

Selection logic (kept compatible with daily_plot_app / all_archives SQL patterns):
  for each local calendar day in the requested range, pick the N archive rows with the
  highest `val` on sensors/<panel>/pv_power, then fetch the `text` column of
  sensors/<panel>/iv_points for exactly those timestamps and expand the stored JSON:
    {"complete": bool, "reason": str, "cycle_id": int,
     "points": [{"seq": n, "ts": <epoch ms>, "duty_pct": "...", "duty": "...",
                 "v": "...", "i": "...", "power": "...", "pv_v": "...", "pv_i": "...",
                 "pv_power": "..."}, ...]}
Only N sweeps per day are exported, so the workbook stays small regardless of how much
raw sweep data the archives hold. `pv_v`/`pv_i`/`pv_power` duplicate v/i/power in every
sampled sweep and are not exported.

Workbook sheets:
  info         run parameters, per-day selection stats, warnings
  power_top    the selected power rows (ts, local time, registered power, sweep metadata)
  iv_summary   one row per sweep: v_max / i_max / p_max / v,i at p_max / fill factor
  iv_points    expanded sweep points (long format)
  plots_by_day per-day overlay of the selected I-V curves   (unless --no-plots)
  plots_each   per-day grid of individual sweeps, I(V) + P(V)

Deps: requests, numpy, matplotlib, openpyxl.
Env defaults match daily_plot_app: PV_CH_URL, PV_CH_DATABASE, PV_CH_USER,
PV_CH_PASSWORD, PV_TIMEZONE.

Usage:
  PV_CH_URL=http://pv-db-mirror.local:8123 PV_CH_DATABASE=ilmet_prod_archives \
    PV_CH_USER=default PV_CH_PASSWORD=default \
    venv_linux_py311/bin/python daily_plot_app/iv_export.py \
      --date-from 2026-07-26 --date-to 2026-07-28 --top 5 \
      --out plots/mppt_39_iv.xlsx

  --date-from/--date-to accept either YYYY-MM-DD (whole local days, end date inclusive)
  or an ISO datetime (e.g. 2026-07-26T10:00); a datetime end is exclusive.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (Agg backend must be set first)
import numpy as np
import requests
from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

# -----------------------------------------------------------------------------
# ClickHouse config / helpers (same env vars as daily_plot_app)
# -----------------------------------------------------------------------------

CFG: Dict[str, str] = {
    "url": os.getenv("PV_CH_URL", "http://localhost:8123"),
    "database": os.getenv("PV_CH_DATABASE", "prod_archives"),
    "user": os.getenv("PV_CH_USER", "default"),
    "password": os.getenv("PV_CH_PASSWORD", ""),
    "timezone": os.getenv("PV_TIMEZONE", "Europe/Warsaw"),
}

HTTP_TIMEOUT_SEC = 60

_session = requests.Session()


def q(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def ch_query(sql: str) -> List[dict]:
    sql = sql.rstrip().rstrip(";") + " FORMAT JSON"
    resp = _session.post(
        CFG["url"],
        params={
            "database": CFG["database"],
            "user": CFG["user"],
            "password": CFG["password"],
            "wait_end_of_query": "1",
        },
        data=sql.encode("utf-8"),
        timeout=HTTP_TIMEOUT_SEC,
        headers={"Content-Type": "text/plain; charset=utf-8"},
    )
    resp.raise_for_status()
    return resp.json().get("data", [])


# -----------------------------------------------------------------------------
# Time helpers
# -----------------------------------------------------------------------------


def parse_bound(text: str, tz: ZoneInfo, is_end: bool) -> datetime:
    """Parse YYYY-MM-DD (whole-day bound) or ISO datetime into a tz-aware datetime."""
    text = text.strip()
    try:
        d = date.fromisoformat(text)
        base = datetime.combine(d, dtime.min, tzinfo=tz)
        return base + timedelta(days=1) if is_end else base
    except ValueError:
        pass

    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=tz)
    return dt.astimezone(tz)


def iter_days(start: datetime, end: datetime) -> Iterable[Tuple[date, datetime, datetime]]:
    """Yield (local day, clipped day start, clipped day end) for [start, end)."""
    cur = start.date()
    last = (end - timedelta(microseconds=1)).date()
    while cur <= last:
        day_start = datetime.combine(cur, dtime.min, tzinfo=start.tzinfo)
        day_end = day_start + timedelta(days=1)
        lo = max(day_start, start)
        hi = min(day_end, end)
        if lo < hi:
            yield cur, lo, hi
        cur += timedelta(days=1)


def fmt_epoch(ts: int, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(int(ts), tz).strftime("%Y-%m-%d %H:%M:%S")


def fmt_ms(ts_ms: int, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def time_of_day(ts_ms: int, tz: ZoneInfo) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz).strftime("%H:%M:%S")


# -----------------------------------------------------------------------------
# Tag lookup / data fetching
# -----------------------------------------------------------------------------


def get_tag_ids(iv_tag: str, power_tag: str) -> Tuple[int, int]:
    sql = f"""
    SELECT ids, tag
    FROM tags_registry
    WHERE tag IN ({q(iv_tag)}, {q(power_tag)})
    """
    found = {str(r["tag"]): int(r["ids"]) for r in ch_query(sql)}
    for tag in (iv_tag, power_tag):
        if tag not in found:
            sys.exit(f"Tag not found in tags_registry: {tag}")
    return found[iv_tag], found[power_tag]


def day_window_sql(day: date, ts_from: int, ts_to: int, tag_id: int) -> str:
    """WHERE clause for one tag over a clipped local day.

    `d` bounds span local-date +/- 1 day so they are correct whether archive tables are
    named by UTC day (mirror/prod convention) or by local day.
    """
    d_lo = (day - timedelta(days=1)).isoformat()
    d_hi = (day + timedelta(days=2)).isoformat()
    return (
        f"WHERE ids = {tag_id}\n"
        f"  AND ts >= {ts_from}\n"
        f"  AND ts < {ts_to}\n"
        f"  AND d >= toDate({q(d_lo)})\n"
        f"  AND d < toDate({q(d_hi)})"
    )


def fetch_day_top(power_id: int, day: date, ts_from: int, ts_to: int, top_n: int) -> Tuple[int, Optional[float], List[dict]]:
    where = day_window_sql(day, ts_from, ts_to, power_id)

    stats = ch_query(f"SELECT count() AS n, max(val) AS mx FROM all_archives {where}")
    n_candidates = int(stats[0]["n"]) if stats else 0
    max_val = stats[0]["mx"] if stats else None

    rows: List[dict] = []
    if n_candidates:
        rows = ch_query(f"SELECT ts, val FROM all_archives {where} ORDER BY val DESC LIMIT {int(top_n)}")
    return n_candidates, max_val, rows


def fetch_iv_texts(ts_values: List[int], iv_id: int, day_first: date, day_last: date, chunk: int = 400) -> Dict[int, str]:
    out: Dict[int, str] = {}
    d_lo = (day_first - timedelta(days=1)).isoformat()
    d_hi = (day_last + timedelta(days=2)).isoformat()

    for i in range(0, len(ts_values), chunk):
        part = ts_values[i:i + chunk]
        ts_csv = ", ".join(str(int(t)) for t in part)
        sql = f"""
        SELECT ts, text
        FROM all_archives
        WHERE ids = {iv_id}
          AND ts IN ({ts_csv})
          AND d >= toDate({q(d_lo)})
          AND d < toDate({q(d_hi)})
        """
        for row in ch_query(sql):
            out[int(row["ts"])] = row.get("text") or ""
    return out


# -----------------------------------------------------------------------------
# JSON parsing / summaries
# -----------------------------------------------------------------------------


def parse_sweep(text: str) -> Tuple[dict, List[dict]]:
    doc = json.loads(text)
    points: List[dict] = []

    for p in doc.get("points") or []:
        try:
            duty_pct_raw = p.get("duty_pct", p.get("duty"))
            duty_raw = p.get("duty", duty_pct_raw)
            points.append(
                {
                    "seq": int(p.get("seq", len(points) + 1)),
                    "ts_ms": int(p["ts"]),
                    "duty_pct": float(duty_pct_raw),
                    "duty": float(duty_raw),
                    "v": float(p.get("v", p.get("pv_v"))),
                    "i": float(p.get("i", p.get("pv_i"))),
                    "power": float(p.get("power", p.get("pv_power"))),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue

    meta = {
        "cycle_id": doc.get("cycle_id"),
        "reason": doc.get("reason"),
        "complete": doc.get("complete"),
    }
    return meta, points


def summarize_sweep(rec: dict, tz: ZoneInfo) -> dict:
    pts = rec["points"]
    v = np.array([p["v"] for p in pts], dtype=float)
    i = np.array([p["i"] for p in pts], dtype=float)
    pw = np.array([p["power"] for p in pts], dtype=float)
    duty = np.array([p["duty_pct"] for p in pts], dtype=float)
    ts_ms = [p["ts_ms"] for p in pts]

    j = int(np.argmax(pw))
    v_max = float(np.max(v))
    i_max = float(np.max(i))
    p_max = float(pw[j])
    fill_factor = p_max / (v_max * i_max) if v_max > 0 and i_max > 0 else float("nan")

    return {
        "v": v,
        "i": i,
        "power": pw,
        "v_max": v_max,
        "i_max": i_max,
        "p_max": p_max,
        "v_at_pmax": float(v[j]),
        "i_at_pmax": float(i[j]),
        "duty_at_pmax": float(duty[j]),
        "fill_factor": fill_factor,
        "n_points": len(pts),
        "sweep_start_local": fmt_ms(min(ts_ms), tz),
        "sweep_end_local": fmt_ms(max(ts_ms), tz),
    }


# -----------------------------------------------------------------------------
# Plots
# -----------------------------------------------------------------------------


def build_day_overlay(day: date, sweeps: List[dict], panel: str) -> plt.Figure:
    n = len(sweeps)
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=110)
    cmap = plt.get_cmap("turbo")
    colors = [cmap(x) for x in np.linspace(0.05, 0.95, n)] if n > 1 else [cmap(0.5)]

    for sw, c in zip(sweeps, colors):
        label = f"{sw['label_time']}  |  {sw['val']:.2f} W reg  |  #{sw['cycle_id']}"
        ax.plot(sw["v"], sw["i"], "-o", color=c, ms=3, lw=1.3, label=label)
        ax.plot(sw["v_at_pmax"], sw["i_at_pmax"], "*", color="red", ms=13, zorder=5)

    ax.set_xlabel("Voltage [V]")
    ax.set_ylabel("Current [A]")
    ax.set_title(f"{panel} I-V characteristics — {day.isoformat()} — top {n} sweeps by registered pv_power (stars = own p_max)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="best", framealpha=0.85)
    fig.tight_layout()
    return fig


def build_day_grid(day: date, sweeps: List[dict], panel: str) -> plt.Figure:
    n = len(sweeps)
    cols = min(3, n)
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.3 * rows), dpi=100, squeeze=False)

    for k, sw in enumerate(sweeps):
        ax = axes[k // cols][k % cols]
        ax.plot(sw["v"], sw["i"], "-o", color="tab:blue", ms=3, lw=1.2, label="I [A]")
        ax.plot(sw["v_at_pmax"], sw["i_at_pmax"], "*", color="red", ms=12)
        ax.set_xlabel("V [V]", fontsize=8)
        ax.set_ylabel("I [A]", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_title(
            f"{sw['label_time']} | {sw['val']:.2f} W reg | max {sw['p_max']:.2f} W | {sw['reason']} | #{sw['cycle_id']}",
            fontsize=8,
        )

        ax2 = ax.twinx()
        ax2.plot(sw["v"], sw["power"], "--", color="tab:orange", lw=1.0)
        ax2.set_ylabel("P [W]", fontsize=8, color="tab:orange")
        ax2.tick_params(labelsize=7, colors="tab:orange")
        ax2.grid(False)

    for j in range(n, rows * cols):
        axes.flat[j].axis("off")

    fig.suptitle(f"{panel} — {day.isoformat()} — individual sweeps", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


# -----------------------------------------------------------------------------
# Excel writing
# -----------------------------------------------------------------------------


def clean_value(v):
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and math.isnan(v):
        return None
    return v


def write_table(
    ws,
    start_row: int,
    headers: List[str],
    rows: List[list],
    widths: Optional[List[float]] = None,
    formats: Optional[Dict[int, str]] = None,
    freeze: bool = True,
) -> int:
    for c, h in enumerate(headers, 1):
        cell = ws.cell(start_row, c, h)
        cell.font = Font(bold=True)
        cell.fill = PatternFill("solid", fgColor="D9E1F2")

    for ri, row in enumerate(rows, start_row + 1):
        for c, v in enumerate(row, 1):
            ws.cell(ri, c, clean_value(v))

    if widths:
        for c, w in enumerate(widths, 1):
            ws.column_dimensions[get_column_letter(c)].width = w

    if formats:
        for c, fmt in formats.items():
            for ri in range(start_row + 1, start_row + 1 + len(rows)):
                ws.cell(ri, c).number_format = fmt

    if freeze:
        ws.freeze_panes = ws.cell(start_row + 1, 1)

    return start_row + 1 + len(rows)


def insert_image_strip(ws, items: List[Tuple[str, Path]], target_width_px: int) -> None:
    ws.sheet_view.showGridLines = False
    row = 1
    for title, png in items:
        cell = ws.cell(row, 1, title)
        cell.font = Font(bold=True)

        img = XLImage(str(png))
        scale = target_width_px / float(img.width)
        img.width = int(img.width * scale)
        img.height = int(img.height * scale)
        ws.add_image(img, f"A{row + 1}")

        # default row height is 15 pt = 20 px; advance below the scaled image
        row += 1 + int(math.ceil(img.height / 20.0)) + 2


def save_workbook(out_path: Path, ex: dict) -> None:
    wb = Workbook()

    # ---- info -----------------------------------------------------------
    ws = wb.active
    ws.title = "info"
    r = 1
    c = ws.cell(r, 1, "PV I/V sweep export")
    c.font = Font(bold=True, size=14)
    r += 2

    info_pairs = [
        ("generated_at", ex["generated_at"]),
        ("clickhouse_url", ex["cfg"]["url"]),
        ("database", ex["cfg"]["database"]),
        ("panel", ex["panel"]),
        ("iv tag / id", f"{ex['iv_tag']} / {ex['iv_id']}"),
        ("power tag / id", f"{ex['power_tag']} / {ex['power_id']}"),
        ("timezone", ex["cfg"]["timezone"]),
        ("date_from", ex["date_from_str"]),
        ("date_to", ex["date_to_str"]),
        ("top N per day", ex["top_n"]),
        ("days scanned", ex["days_total"]),
        ("days with sweeps", ex["days_with_sweeps"]),
        ("sweeps exported", len(ex["summary_rows"])),
        ("power rows missing iv text", ex["missing_iv"]),
        ("sweep parse errors / empty", ex["parse_errors"]),
    ]
    for k, v in info_pairs:
        ws.cell(r, 1, k).font = Font(bold=True)
        ws.cell(r, 2, v)
        r += 1

    r += 1
    ws.cell(r, 1, "Per-day selection stats").font = Font(bold=True)
    r += 1
    r = write_table(
        ws,
        r,
        ["day", "power rows in window", "max power [W]", "selected (top N)", "iv rows found", "missing iv", "sweeps exported"],
        ex["day_table_rows"],
        widths=[14, 22, 14, 16, 14, 12, 16],
        formats={3: "0.000"},
        freeze=False,
    )

    r += 1
    ws.cell(r, 1, "Warnings").font = Font(bold=True)
    r += 1
    if ex["warnings"]:
        for w in ex["warnings"]:
            ws.cell(r, 1, w)
            r += 1
    else:
        ws.cell(r, 1, "(none)")
        r += 1

    ws.column_dimensions["A"].width = 56
    ws.column_dimensions["B"].width = 34

    # ---- power_top ------------------------------------------------------
    ws = wb.create_sheet("power_top")
    write_table(
        ws,
        1,
        ["day", "archive_ts", "archive_time_local", "power_w", "iv_found", "cycle_id", "reason", "complete", "n_points"],
        ex["top_rows"],
        widths=[12, 12, 22, 12, 10, 12, 30, 10, 10],
        formats={4: "0.000"},
    )

    # ---- iv_summary -----------------------------------------------------
    ws = wb.create_sheet("iv_summary")
    write_table(
        ws,
        1,
        [
            "day", "cycle_id", "archive_ts", "archive_time_local", "reason", "complete",
            "power_at_selection_w", "n_points", "sweep_start_local", "sweep_end_local",
            "v_min", "v_max", "i_min", "i_max", "p_max", "v_at_pmax", "i_at_pmax",
            "duty_pct_at_pmax", "fill_factor",
        ],
        ex["summary_rows"],
        widths=[12, 12, 12, 22, 30, 10, 20, 10, 24, 24, 10, 10, 10, 10, 10, 12, 12, 18, 12],
        formats={7: "0.000", 11: "0.000", 12: "0.000", 13: "0.000", 14: "0.000", 15: "0.000",
                 16: "0.000", 17: "0.000", 18: "0.000", 19: "0.000"},
    )

    # ---- iv_points ------------------------------------------------------
    ws = wb.create_sheet("iv_points")
    write_table(
        ws,
        1,
        ["day", "cycle_id", "archive_ts", "archive_time_local", "seq", "point_ts_ms", "point_time_local",
         "duty_pct", "duty", "v", "i", "power"],
        ex["point_rows"],
        widths=[12, 12, 12, 22, 6, 14, 26, 10, 10, 10, 10, 10],
        formats={8: "0.000", 9: "0.000", 10: "0.000", 11: "0.000", 12: "0.000"},
    )

    # ---- plots ----------------------------------------------------------
    if ex["plot_images"]:
        ws = wb.create_sheet("plots_by_day")
        items = [
            (f"{day.isoformat()}  ({ex['sweeps_per_day'].get(day, 0)} sweeps, top by registered pv_power)", overlay)
            for day, (overlay, _grid) in sorted(ex["plot_images"].items())
        ]
        insert_image_strip(ws, items, target_width_px=1050)

        ws = wb.create_sheet("plots_each")
        items = [
            (f"{day.isoformat()}  — individual sweeps  ({ex['sweeps_per_day'].get(day, 0)})", grid)
            for day, (_overlay, grid) in sorted(ex["plot_images"].items())
        ]
        insert_image_strip(ws, items, target_width_px=1050)

    wb.save(out_path)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Export N highest-power I/V characteristics per day for a panel to Excel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--date-from", required=True, help="Start: YYYY-MM-DD or ISO datetime")
    p.add_argument("--date-to", default=None, help="End: YYYY-MM-DD (inclusive day) or ISO datetime (exclusive); defaults to --date-from")
    p.add_argument("--top", type=int, default=5, help="N highest-power sweeps per day")
    p.add_argument("--panel", default="sensors/mppt_39", help="Panel tag prefix")
    p.add_argument("--out", default=None, help="Output .xlsx path")
    p.add_argument("--no-plots", action="store_true", help="Skip the plot sheets (data-only workbook)")
    p.add_argument("--url", default=CFG["url"])
    p.add_argument("--database", default=CFG["database"])
    p.add_argument("--user", default=CFG["user"])
    p.add_argument("--password", default=CFG["password"])
    p.add_argument("--timezone", default=CFG["timezone"])
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = build_argparser().parse_args(argv)

    CFG["url"] = args.url
    CFG["database"] = args.database
    CFG["user"] = args.user
    CFG["password"] = args.password
    CFG["timezone"] = args.timezone

    tz = ZoneInfo(args.timezone)
    start = parse_bound(args.date_from, tz, is_end=False)
    end = parse_bound(args.date_to or args.date_from, tz, is_end=True)
    if end <= start:
        sys.exit("Invalid range: end must be after start (pass --date-to for a datetime start)")
    if (end - start).days > 400:
        sys.exit("Range too long ( > 400 days ); narrow --date-from / --date-to")

    top_n = max(1, int(args.top))
    panel = args.panel.rstrip("/")
    iv_tag = f"{panel}/iv_points"
    power_tag = f"{panel}/pv_power"

    iv_id, power_id = get_tag_ids(iv_tag, power_tag)
    print(f"Range {start.isoformat()} .. {end.isoformat()} | panel {panel} | iv id {iv_id} | power id {power_id} | top {top_n}/day")

    # ---- select top-N power rows per day ---------------------------------
    records: List[dict] = []
    day_table_rows: List[list] = []
    warnings: List[str] = []
    days_total = 0

    for day, d_start, d_end in iter_days(start, end):
        days_total += 1
        ts_from, ts_to = int(d_start.timestamp()), int(d_end.timestamp())
        n_candidates, max_val, top_rows = fetch_day_top(power_id, day, ts_from, ts_to, top_n)

        seen = set()
        selected = []
        for tr in top_rows:
            ts = int(tr["ts"])
            if ts in seen:  # duplicate archive rows for the same ts/time are rare; keep top one
                continue
            seen.add(ts)
            selected.append({"ts": ts, "val": float(tr["val"])})

        for s in selected:
            records.append({"day": day, "ts": s["ts"], "val": s["val"]})

        day_table_rows.append([day.isoformat(), n_candidates, max_val, len(selected), 0, 0, 0])
        print(f"  {day.isoformat()}: power rows {n_candidates}, selected {len(selected)}")

    if not records:
        print("No power rows found in the requested range.")

    # ---- fetch iv texts for the selected timestamps ----------------------
    day_first = start.date()
    day_last = (end - timedelta(microseconds=1)).date()
    texts: Dict[int, str] = {}
    if records:
        all_ts = sorted({r["ts"] for r in records})
        texts = fetch_iv_texts(all_ts, iv_id, day_first, day_last)
        print(f"Fetched iv texts: {len(texts)} / {len(all_ts)} selected timestamps")

    # ---- parse sweeps -----------------------------------------------------
    top_rows_out: List[list] = []
    summary_rows: List[list] = []
    point_rows: List[list] = []
    sweeps_by_day: Dict[date, List[dict]] = {}
    missing_iv = 0
    parse_errors = 0

    for rec in records:
        text = texts.get(rec["ts"])
        rec["meta"] = None
        rec["points"] = []

        if not text:
            missing_iv += 1
            warnings.append(f"{rec['day'].isoformat()}: no iv_points text at ts={rec['ts']} (power {rec['val']:.3f} W)")
        else:
            try:
                meta, points = parse_sweep(text)
            except json.JSONDecodeError as e:
                parse_errors += 1
                warnings.append(f"{rec['day'].isoformat()}: JSON parse error at ts={rec['ts']}: {e}")
            else:
                rec["meta"] = meta
                rec["points"] = points
                if not points:
                    parse_errors += 1
                    warnings.append(f"{rec['day'].isoformat()}: empty points array at ts={rec['ts']}")

        meta = rec["meta"] or {}
        top_rows_out.append([
            rec["day"].isoformat(),
            rec["ts"],
            fmt_epoch(rec["ts"], tz),
            rec["val"],
            bool(rec["points"]),
            meta.get("cycle_id"),
            meta.get("reason"),
            meta.get("complete"),
            len(rec["points"]),
        ])

        if not rec["points"]:
            continue

        s = summarize_sweep(rec, tz)
        summary_rows.append([
            rec["day"].isoformat(),
            meta.get("cycle_id"),
            rec["ts"],
            fmt_epoch(rec["ts"], tz),
            meta.get("reason"),
            meta.get("complete"),
            rec["val"],
            s["n_points"],
            s["sweep_start_local"],
            s["sweep_end_local"],
            float(np.min(s["v"])),
            s["v_max"],
            float(np.min(s["i"])),
            s["i_max"],
            s["p_max"],
            s["v_at_pmax"],
            s["i_at_pmax"],
            s["duty_at_pmax"],
            s["fill_factor"],
        ])

        for p in rec["points"]:
            point_rows.append([
                rec["day"].isoformat(),
                meta.get("cycle_id"),
                rec["ts"],
                fmt_epoch(rec["ts"], tz),
                p["seq"],
                p["ts_ms"],
                fmt_ms(p["ts_ms"], tz),
                p["duty_pct"],
                p["duty"],
                p["v"],
                p["i"],
                p["power"],
            ])

        sweep = {
            "v": s["v"],
            "i": s["i"],
            "power": s["power"],
            "v_at_pmax": s["v_at_pmax"],
            "i_at_pmax": s["i_at_pmax"],
            "p_max": s["p_max"],
            "val": rec["val"],
            "cycle_id": meta.get("cycle_id"),
            "reason": meta.get("reason"),
            "label_time": time_of_day(rec["points"][0]["ts_ms"], tz),
        }
        sweeps_by_day.setdefault(rec["day"], []).append(sweep)

    # fill iv/missing counters into the per-day table
    for row in day_table_rows:
        day = date.fromisoformat(row[0])
        day_recs = [r for r in records if r["day"] == day]
        found = sum(1 for r in day_recs if r["points"])
        row[4] = found
        row[5] = sum(1 for r in day_recs if not r["points"])
        row[6] = len(sweeps_by_day.get(day, []))

    # ---- plots ------------------------------------------------------------
    plot_images: Dict[date, Tuple[Path, Path]] = {}
    sweeps_per_day: Dict[date, int] = {}
    tmpdir: Optional[Path] = None

    if sweeps_by_day and not args.no_plots:
        tmpdir = Path(tempfile.mkdtemp(prefix="iv_export_"))
        for day in sorted(sweeps_by_day):
            sweeps = sweeps_by_day[day]
            sweeps_per_day[day] = len(sweeps)

            fig = build_day_overlay(day, sweeps, panel)
            overlay_png = tmpdir / f"overlay_{day.isoformat()}.png"
            fig.savefig(overlay_png, bbox_inches="tight", facecolor="white")
            plt.close(fig)

            fig = build_day_grid(day, sweeps, panel)
            grid_png = tmpdir / f"sweeps_{day.isoformat()}.png"
            fig.savefig(grid_png, bbox_inches="tight", facecolor="white")
            plt.close(fig)

            plot_images[day] = (overlay_png, grid_png)
        print(f"Generated {len(plot_images)} day plot sets")

    # ---- output ------------------------------------------------------------
    if args.out:
        out_path = Path(args.out).expanduser()
    else:
        slug = panel.replace("/", "_")
        out_path = Path(f"iv_{slug}_{args.date_from}_{args.date_to or args.date_from}.xlsx")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    export = {
        "generated_at": datetime.now(tz).isoformat(timespec="seconds"),
        "cfg": dict(CFG),
        "panel": panel,
        "iv_tag": iv_tag,
        "power_tag": power_tag,
        "iv_id": iv_id,
        "power_id": power_id,
        "date_from_str": args.date_from,
        "date_to_str": args.date_to or args.date_from,
        "top_n": top_n,
        "days_total": days_total,
        "days_with_sweeps": len(sweeps_by_day),
        "missing_iv": missing_iv,
        "parse_errors": parse_errors,
        "day_table_rows": day_table_rows,
        "warnings": warnings,
        "top_rows": top_rows_out,
        "summary_rows": summary_rows,
        "point_rows": point_rows,
        "plot_images": plot_images,
        "sweeps_per_day": sweeps_per_day,
    }

    save_workbook(out_path, export)
    if tmpdir:
        shutil.rmtree(tmpdir, ignore_errors=True)

    size_mb = out_path.stat().st_size / 1e6
    print(
        f"\nWorkbook: {out_path.resolve()} ({size_mb:.1f} MB)\n"
        f"days scanned {days_total}, days with sweeps {len(sweeps_by_day)}, sweeps {len(summary_rows)}, "
        f"points {len(point_rows)}, missing iv {missing_iv}, parse errors {parse_errors}"
    )
    if warnings:
        print(f"Warnings: {len(warnings)} (see 'info' sheet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
