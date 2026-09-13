"""Market-data acquisition and normalization.

The yfinance dependency is optional so dependency-light package imports work.
"""

from __future__ import annotations

import pandas as pd


def download_eurusd(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Download and normalize the OHLCV frame used by the original notebook."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install forex-rl-simulator[research] to download market data.") from exc
    df = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False, group_by="column")
    if isinstance(df.columns, pd.MultiIndex):
        if df.columns.nlevels >= 2 and len(df.columns.get_level_values(1).unique()) == 1:
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = ["_".join(str(x) for x in col if x is not None).lower() for col in df.columns]
    df = df.rename(columns=str.lower).loc[:, lambda frame: ~frame.columns.duplicated()]
    required, cols = ["open", "high", "low", "close", "volume"], {}
    for base in required:
        matches = [column for column in df.columns if isinstance(column, str) and column.startswith(base)]
        if matches:
            cols[base] = base if base in df.columns else matches[0]
    missing = [base for base in required if base not in cols]
    if missing:
        raise ValueError(f"Could not locate required columns: {missing}. Got {list(df.columns)}")
    out = df[[cols[base] for base in required]].copy()
    out.columns = required
    for column in required:
        out[column] = pd.to_numeric(out[column].squeeze(), errors="coerce")
    return out.dropna().sort_index()
