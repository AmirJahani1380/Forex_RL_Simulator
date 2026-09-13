"""Train-only feature scaling."""

import pandas as pd


class ZScoreScaler:
    def __init__(self):
        self.means = None
        self.stds = None

    def fit(self, df_feat: pd.DataFrame):
        self.means = df_feat.mean()
        self.stds = df_feat.std().replace(0, 1e-8)

    def transform(self, df_feat: pd.DataFrame) -> pd.DataFrame:
        if self.means is None or self.stds is None:
            raise RuntimeError("ZScoreScaler must be fitted before transform.")
        return (df_feat - self.means) / self.stds

    def fit_transform(self, df_feat: pd.DataFrame) -> pd.DataFrame:
        self.fit(df_feat)
        return self.transform(df_feat)


def make_scaled_frames(train_df: pd.DataFrame, eval_df: pd.DataFrame, feature_cols):
    scaler = ZScoreScaler()
    scaler.fit(train_df[feature_cols])
    return scaler.transform(train_df[feature_cols]), scaler.transform(eval_df[feature_cols]), scaler
