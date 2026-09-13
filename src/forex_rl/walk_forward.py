"""Chronological walk-forward boundary construction."""

import pandas as pd


def build_walkforward_folds(df: pd.DataFrame, min_train_months=24, val_months=6, test_months=6,
                            step_months=6, use_rolling=True, rolling_train_months=36):
    start, end = df.index.min(), df.index.max()
    folds, test_start, fold = [], start + pd.DateOffset(months=min_train_months + val_months), 1
    while True:
        train_end = test_start - pd.DateOffset(months=val_months)
        test_end = test_start + pd.DateOffset(months=test_months)
        if test_end > end:
            break
        train_start = max(train_end - pd.DateOffset(months=max(min_train_months, rolling_train_months)), start) if use_rolling else start
        folds.append({"fold": fold, "train_start": train_start, "train_end": train_end,
                      "val_start": train_end, "val_end": test_start,
                      "test_start": test_start, "test_end": test_end})
        fold += 1
        test_start += pd.DateOffset(months=step_months)
    return folds
