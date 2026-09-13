"""Reporting interfaces kept separate from training and optional plotting."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import performance_metrics


def extended_metrics(trades, mtm_df):
    metrics = performance_metrics(trades, mtm_df)
    if trades is None or len(trades) == 0:
        return {**metrics, "avg_bars_in_trade": np.nan, "med_bars_in_trade": np.nan, "avg_holding_days": np.nan}
    if "bars_in_trade" in trades:
        duration = pd.to_numeric(trades["bars_in_trade"], errors="coerce").dropna().to_numpy()
        return {
            **metrics,
            "avg_bars_in_trade": float(np.mean(duration)) if len(duration) else np.nan,
            "med_bars_in_trade": float(np.median(duration)) if len(duration) else np.nan,
            "avg_holding_days": np.nan,
        }
    duration = pd.to_datetime(trades["exit_t"]) - pd.to_datetime(trades["entry_t"])
    days = duration.dt.total_seconds().dropna().to_numpy() / 86_400.0
    return {
        **metrics,
        "avg_bars_in_trade": np.nan,
        "med_bars_in_trade": np.nan,
        "avg_holding_days": float(np.mean(days)) if len(days) else np.nan,
    }


def generate_trade_report(df_slice, trades, mtm_df, equity_r=None, out_prefix="logs_rl/rl_report"):
    """Write inspectable JSON/CSV/HTML report artifacts without requiring plotting libraries."""
    prefix = Path(out_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics = extended_metrics(trades, mtm_df)
    json_path, csv_path, equity_path, html_path = (
        prefix.with_name(prefix.name + "_metrics.json"),
        prefix.with_name(prefix.name + "_trades.csv"),
        prefix.with_name(prefix.name + "_equity.csv"),
        prefix.with_suffix(".html"),
    )
    json_path.write_text(
        json.dumps(metrics, indent=2, default=lambda value: None if np.isnan(value) else value), encoding="utf-8"
    )
    trades.to_csv(csv_path, index=False)
    (mtm_df if mtm_df is not None else pd.DataFrame()).to_csv(equity_path, index=True)
    html_path.write_text(
        f"<html><body><h2>RL Strategy Report</h2><p>Generated: {datetime.now(timezone.utc).isoformat()}</p><pre>{json.dumps(metrics, indent=2, default=str)}</pre><p>Trades: {csv_path.name}</p></body></html>",
        encoding="utf-8",
    )
    return {
        "html": str(html_path),
        "csv": str(csv_path),
        "equity_csv": str(equity_path),
        "json": str(json_path),
        "metrics": metrics,
    }


def aggregate_walkforward_results(results):
    """Notebook cell 17 aggregation, retained as a local reusable helper."""
    columns = [
        "total_R",
        "PF",
        "win_rate",
        "trades",
        "sharpe",
        "sortino",
        "maxDD_pct",
        "MAR_R",
        "return_pct",
        "turnover",
    ]
    aggregate = {name: {} for name in ("mean", "median", "std", "min", "max")}
    for column in columns:
        if column in results:
            values = results[column].astype(float).values
            if len(values):
                aggregate["mean"][column] = float(np.nanmean(values))
                aggregate["median"][column] = float(np.nanmedian(values))
                aggregate["std"][column] = float(np.nanstd(values, ddof=1)) if len(values) > 1 else np.nan
                aggregate["min"][column] = float(np.nanmin(values))
                aggregate["max"][column] = float(np.nanmax(values))
    return aggregate


def generate_walkforward_report(results, report_dir, algo):
    """Write original-style local CSV, aggregate JSON, and HTML summary."""
    directory = Path(report_dir)
    directory.mkdir(parents=True, exist_ok=True)
    csv_path, json_path, html_path = (
        directory / "wf_results.csv",
        directory / "wf_aggregates.json",
        directory / "wf_report.html",
    )
    aggregate = aggregate_walkforward_results(results)
    results.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(aggregate, indent=2, default=str), encoding="utf-8")
    html_path.write_text(
        f"<h2>Walk-Forward Report — {algo}</h2><h3>Per-Fold Summary</h3>{results.to_html(index=False)}<h3>Aggregates</h3><pre>{json.dumps(aggregate, indent=2, default=str)}</pre>",
        encoding="utf-8",
    )
    return {"csv": str(csv_path), "json": str(json_path), "html": str(html_path), "aggregates": aggregate}


def zip_directory(root_dir, output_zip):
    """Explicit local ZIP helper; callers decide whether to create an archive."""
    root, destination = Path(root_dir), Path(output_zip)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in root.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(root))
    return str(destination)
