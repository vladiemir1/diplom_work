from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


SENTIMENT_COLORS = {
    "Положительная": "#2f9e44",
    "Отрицательная": "#e03131",
    "Нейтральная": "#868e96",
}

SHARE_COLUMNS = {
    "positive_share": "Положительная",
    "negative_share": "Отрицательная",
    "neutral_share": "Нейтральная",
}


def _apply_compact_layout(fig: go.Figure, height: int = 330) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=24, r=24, t=56, b=34),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#30384a", size=13),
        title_font=dict(size=18, color="#252b3a"),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="#e6ebf2", zeroline=False)
    fig.update_yaxes(gridcolor="#e6ebf2", zeroline=False)
    return fig


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, xaxis_visible=False, yaxis_visible=False)
    fig.add_annotation(text="Нет данных для отображения", showarrow=False)
    return _apply_compact_layout(fig)


def build_sentiment_share_chart(agg_df: pd.DataFrame) -> go.Figure:
    if agg_df.empty:
        return _empty_figure("Тональность по аспектам")

    long_df = agg_df.melt(
        id_vars="aspect",
        value_vars=list(SHARE_COLUMNS),
        var_name="sentiment",
        value_name="share",
    )
    long_df["sentiment_label"] = long_df["sentiment"].map(SHARE_COLUMNS)
    fig = px.bar(
        long_df,
        x="aspect",
        y="share",
        color="sentiment_label",
        barmode="stack",
        text=long_df["share"].map(lambda value: f"{value:.1f}%"),
        title="Процент тональности по аспектам",
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_traces(textposition="inside", insidetextanchor="middle")
    fig.update_layout(yaxis_ticksuffix="%", yaxis_range=[0, 100])
    fig.update_xaxes(title="")
    fig.update_yaxes(title="Доля")
    return _apply_compact_layout(fig, height=360)


def build_negative_rate_chart(agg_df: pd.DataFrame) -> go.Figure:
    if agg_df.empty:
        return _empty_figure("Аспекты по доле негатива")

    sorted_df = agg_df.sort_values("negative_rate", ascending=True)
    fig = px.bar(
        sorted_df,
        x="negative_rate",
        y="aspect",
        orientation="h",
        text=sorted_df["negative_rate"].map(lambda value: f"{value:.1f}%"),
        title="Аспекты с наибольшей долей негатива",
        color="negative_rate",
        color_continuous_scale=["#ffe3e3", "#e03131"],
        range_x=[0, max(100, float(sorted_df["negative_rate"].max()))],
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(title="Доля негатива", ticksuffix="%")
    fig.update_yaxes(title="")
    return _apply_compact_layout(fig, height=320)


def build_overall_sentiment_donut(results_df: pd.DataFrame) -> go.Figure:
    if results_df.empty or "sentiment_label" not in results_df.columns:
        return _empty_figure("Общая тональность")

    counts = (
        results_df["sentiment_label"]
        .fillna("Нейтральная")
        .value_counts()
        .rename_axis("sentiment_label")
        .reset_index(name="count")
    )
    fig = px.pie(
        counts,
        names="sentiment_label",
        values="count",
        hole=0.58,
        title="Общая тональность упоминаний",
        color="sentiment_label",
        color_discrete_map=SENTIMENT_COLORS,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _apply_compact_layout(fig, height=320)

