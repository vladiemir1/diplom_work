from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


SENTIMENT_LABELS = {
    "positive_count": "Положительная",
    "negative_count": "Отрицательная",
    "neutral_count": "Нейтральная",
}


def _empty_figure(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(title=title, xaxis_visible=False, yaxis_visible=False)
    fig.add_annotation(text="Нет данных для отображения", showarrow=False)
    return fig


def build_aspect_mentions_chart(agg_df: pd.DataFrame) -> go.Figure:
    if agg_df.empty:
        return _empty_figure("Упоминания аспектов")
    fig = px.bar(
        agg_df,
        x="aspect",
        y="mention_count",
        color="aspect",
        text="mention_count",
        title="Количество упоминаний аспектов",
    )
    fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Упоминаний")
    return fig


def build_sentiment_distribution_chart(agg_df: pd.DataFrame) -> go.Figure:
    if agg_df.empty:
        return _empty_figure("Распределение тональности")
    long_df = agg_df.melt(
        id_vars="aspect",
        value_vars=list(SENTIMENT_LABELS),
        var_name="sentiment",
        value_name="count",
    )
    long_df["sentiment_label"] = long_df["sentiment"].map(SENTIMENT_LABELS)
    fig = px.bar(
        long_df,
        x="aspect",
        y="count",
        color="sentiment_label",
        barmode="stack",
        title="Распределение тональности по аспектам",
        color_discrete_map={
            "Положительная": "#2f9e44",
            "Отрицательная": "#e03131",
            "Нейтральная": "#868e96",
        },
    )
    fig.update_layout(xaxis_title="", yaxis_title="Упоминаний", legend_title="")
    return fig


def build_confidence_chart(agg_df: pd.DataFrame) -> go.Figure:
    if agg_df.empty:
        return _empty_figure("Средняя уверенность")
    fig = px.bar(
        agg_df,
        x="aspect",
        y="confidence_avg",
        color="confidence_avg",
        text="confidence_avg",
        title="Средняя уверенность анализа по аспектам",
        range_y=[0, 1],
        color_continuous_scale="Teal",
    )
    fig.update_layout(xaxis_title="", yaxis_title="Уверенность", coloraxis_showscale=False)
    return fig


def build_date_sentiment_chart(results_df: pd.DataFrame) -> go.Figure:
    if results_df.empty or "date" not in results_df.columns:
        return _empty_figure("Динамика тональности")
    date_df = results_df.dropna(subset=["date"]).copy()
    if date_df.empty:
        return _empty_figure("Динамика тональности")
    date_df["date"] = pd.to_datetime(date_df["date"], errors="coerce").dt.date
    date_df = date_df.dropna(subset=["date"])
    if date_df.empty:
        return _empty_figure("Динамика тональности")
    grouped = date_df.groupby(["date", "sentiment_label"]).size().reset_index(name="count")
    fig = px.line(
        grouped,
        x="date",
        y="count",
        color="sentiment_label",
        markers=True,
        title="Динамика тональности по датам",
    )
    fig.update_layout(xaxis_title="", yaxis_title="Упоминаний", legend_title="")
    return fig

