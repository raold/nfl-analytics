"""
Feature catalog helpers reflecting dissertation Chapter 3.

Defines families and a simple builder to materialize feature sets from a base
DataFrame. Real implementations should source from marts and enforce as-of joins.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class FeatureFamily:
    name: str
    columns: list[str]


FAMILIES: dict[str, FeatureFamily] = {
    "situational": FeatureFamily(
        name="situational",
        columns=["down", "ydstogo", "qtr", "score_diff", "seconds_remaining"],
    ),
    "team_form": FeatureFamily(
        name="team_form",
        columns=[
            "epa_off_rolling",
            "epa_def_rolling",
            "success_rate_off",
            "success_rate_def",
            "pace_drives",
        ],
    ),
    "market": FeatureFamily(
        name="market",
        columns=["spread_close", "total_close", "hold", "line_velocity", "cross_book_delta"],
    ),
    "roster": FeatureFamily(
        name="roster",
        columns=["qb_availability", "wr_depth", "ol_health", "def_inj_ags", "rest_days"],
    ),
}


def build_features(df: pd.DataFrame, families: list[str]) -> pd.DataFrame:
    """Select a column subset corresponding to requested families.

    Assumes df already contains these columns; callers should join marts first.
    """
    cols: list[str] = []
    for fam in families:
        if fam not in FAMILIES:
            continue
        cols.extend(FAMILIES[fam].columns)
    # Deduplicate while preserving order
    seen = set()
    ordered_cols: list[str] = []
    for c in cols:
        if c in seen:
            continue
        if c in df.columns:
            ordered_cols.append(c)
            seen.add(c)
    return df[ordered_cols].copy()


__all__ = ["FeatureFamily", "FAMILIES", "build_features"]
