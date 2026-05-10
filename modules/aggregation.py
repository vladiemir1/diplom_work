from __future__ import annotations

import pandas as pd


SENTIMENTS = ("positive", "negative", "neutral")


def aggregate_results(results_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "aspect",
        "mention_count",
        "positive_count",
        "negative_count",
        "neutral_count",
        "confidence_avg",
    ]
    if results_df.empty:
        return pd.DataFrame(columns=columns)

    grouped = results_df.groupby("aspect", dropna=False)
    rows = []
    for aspect, group in grouped:
        sentiments = group["sentiment"].fillna("neutral")
        rows.append(
            {
                "aspect": aspect,
                "mention_count": int(len(group)),
                "positive_count": int((sentiments == "positive").sum()),
                "negative_count": int((sentiments == "negative").sum()),
                "neutral_count": int((sentiments == "neutral").sum()),
                "confidence_avg": round(float(group["confidence"].fillna(0).mean()), 3),
            }
        )

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["mention_count", "aspect"], ascending=[False, True]
    )

