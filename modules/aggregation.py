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
        "positive_share",
        "negative_share",
        "neutral_share",
        "negative_rate",
        "confidence_avg",
    ]
    if results_df.empty:
        return pd.DataFrame(columns=columns)

    # Exclude legacy "Общее впечатление" aspect if present
    filtered = results_df[results_df["aspect"] != "Общее впечатление"].copy()
    if filtered.empty:
        return pd.DataFrame(columns=columns)

    grouped = filtered.groupby("aspect", dropna=False)
    rows = []
    for aspect, group in grouped:
        sentiments = group["sentiment"].fillna("neutral")
        mention_count = int(len(group))
        positive_count = int((sentiments == "positive").sum())
        negative_count = int((sentiments == "negative").sum())
        neutral_count = int((sentiments == "neutral").sum())
        rows.append(
            {
                "aspect": aspect,
                "mention_count": mention_count,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "positive_share": round(positive_count / mention_count * 100, 1),
                "negative_share": round(negative_count / mention_count * 100, 1),
                "neutral_share": round(neutral_count / mention_count * 100, 1),
                "negative_rate": round(negative_count / mention_count * 100, 1),
                "confidence_avg": round(float(group["confidence"].fillna(0).mean()), 3),
            }
        )

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["mention_count", "aspect"], ascending=[False, True]
    )
