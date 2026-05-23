from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from modules.aggregation import aggregate_results
from modules.data_loader import (
    DataLoadError,
    load_reviews,
    prepare_reviews_dataframe,
    selectable_text_columns,
    validate_reviews_df,
)
from modules.export import to_csv_bytes, to_wide_format, to_xlsx_bytes
from modules.openai_analyzer import AnalyzerConfig, AnalyzerError, analyze_reviews
from modules.preprocessing import preprocess_reviews
from modules.visualization import (
    build_negative_rate_chart,
    build_overall_sentiment_donut,
    build_sentiment_share_chart,
)


APP_TITLE = "Система анализа тональности отзывов маркетплейса"


def init_state() -> None:
    defaults = {
        "raw_df": None,
        "processed_df": None,
        "results_df": None,
        "agg_df": None,
        "wide_df": None,
        "source_name": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_analysis() -> None:
    for key in ("processed_df", "results_df", "agg_df", "wide_df"):
        st.session_state[key] = None


def set_raw_data(df: pd.DataFrame, source_name: str) -> None:
    st.session_state["raw_df"] = df
    st.session_state["source_name"] = source_name
    clear_analysis()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --app-bg: #f4f5f7;
            --panel-bg: #ffffff;
            --panel-muted: #eef2f7;
            --border: #dce3ee;
            --text: #172033;
            --muted: #6f7788;
            --accent: #135abc;
            --accent-soft: #e8f0ff;
            --success: #2f9e44;
            --danger: #d94848;
        }
        .stApp {
            background:
                radial-gradient(circle at 85% 8%, rgba(19, 90, 188, .10), transparent 24%),
                linear-gradient(180deg, #f7f8fb 0%, var(--app-bg) 36%, #eef2f6 100%);
        }
        .main .block-container {
            padding-top: 0 !important;
            padding-bottom: 2rem !important;
            padding-left: 2rem !important;
            padding-right: 2rem !important;
            margin-top: -1.5rem;
            max-width: 1200px;
        }
        /* Hide anchor links in headers */
        h1 a, h2 a, h3 a, h4 a, h5 a, h6 a {
            display: none !important;
        }
        .stMarkdown a.header-anchor {
            display: none !important;
        }
        #MainMenu, footer, header {
            visibility: hidden;
        }
        div[data-testid="stHeader"], div[data-testid="stDecoration"] {
            display: none;
        }
        .hero {
            background:
                linear-gradient(135deg, rgba(255,255,255,.98), rgba(244,247,252,.96));
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 24px 32px !important;
            box-shadow: 0 14px 32px rgba(35, 50, 75, .07);
            margin-bottom: 5px !important;
            position: relative;
            overflow: hidden;
        }
        .hero::after {
            content: "";
            position: absolute;
            top: 0;
            right: 0;
            width: 32%;
            height: 100%;
            background:
                linear-gradient(90deg, transparent, rgba(19, 90, 188, .05)),
                repeating-linear-gradient(135deg, rgba(19, 90, 188, .10) 0 1px, transparent 1px 18px);
            pointer-events: none;
        }
        .hero-content {
            position: relative;
            z-index: 1;
            max-width: 780px;
        }
        .eyebrow {
            color: var(--accent);
            font-size: 11px;
            font-weight: 700;
            letter-spacing: .08em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .hero h1 {
            margin: 0 0 4px 0;
            font-size: 28px;
            line-height: 1.15;
            letter-spacing: 0;
            color: var(--text);
        }
        .hero p {
            margin: 0;
            color: #4d5870;
            font-size: 13px;
            max-width: 620px;
        }
        .hero-steps {
            display: flex;
            flex-wrap: wrap;
            gap: 7px;
            margin-top: 10px;
        }
        .step-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 5px 10px;
            border: 1px solid #d6deeb;
            background: rgba(255,255,255,.72);
            color: #33415c;
            border-radius: 999px;
            font-size: 12px;
            white-space: nowrap;
        }
        .step-index {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 18px;
            height: 18px;
            border-radius: 50%;
            background: var(--accent);
            color: white;
            font-size: 11px;
            font-weight: 700;
        }
        .section-head {
            margin: 4px 0 2px 0;
        }
        .section-kicker {
            color: var(--accent);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: .06em;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .section-title {
            color: var(--text);
            font-size: 22px;
            font-weight: 750;
            line-height: 1.2;
            margin: 0;
        }
        .section-note {
            color: var(--muted);
            font-size: 14px;
            margin-top: 4px;
        }
        .table-spacer {
            margin-top: 10px;
        }
        @media (max-width: 900px) {
            .hero::after {
                opacity: .35;
            }
        }
        .empty-state {
            margin: 10px 0 8px 0;
            padding: 18px 20px;
            background: rgba(255,255,255,.72);
            border: 1px dashed #c8d3e3;
            border-radius: 8px;
            color: #536078;
        }
        .empty-state-title {
            color: var(--text);
            font-weight: 700;
            font-size: 16px;
            margin-bottom: 4px;
        }
        .metric-card {
            background: var(--panel-bg);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px 18px 17px 18px;
            min-height: 96px;
            box-shadow: 0 10px 24px rgba(30, 45, 70, .055);
            position: relative;
            overflow: hidden;
        }
        .metric-card::before {
            content: "";
            position: absolute;
            left: 0;
            top: 0;
            width: 4px;
            height: 100%;
            background: var(--accent);
        }
        .metric-label {
            color: var(--muted);
            font-size: 13px;
            margin-bottom: 8px;
        }
        .metric-value {
            color: var(--text);
            font-size: 28px;
            font-weight: 700;
        }
        .sentiment-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin: 14px 0 4px 0;
        }
        .sentiment-mini {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 12px 14px;
            box-shadow: 0 8px 18px rgba(30, 45, 70, .04);
        }
        .sentiment-mini-label {
            color: var(--muted);
            font-size: 12px;
        }
        .sentiment-mini-value {
            color: var(--text);
            font-size: 21px;
            font-weight: 750;
            margin-top: 3px;
        }
        .aspect-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 14px;
            margin: 12px 0 22px 0;
        }
        .aspect-card {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 15px 16px;
            box-shadow: 0 10px 24px rgba(30, 45, 70, .05);
        }
        .aspect-card-head {
            display: flex;
            align-items: flex-start;
            justify-content: space-between;
            gap: 12px;
            margin-bottom: 12px;
        }
        .aspect-name {
            color: var(--text);
            font-size: 16px;
            font-weight: 750;
        }
        .aspect-count {
            color: var(--accent);
            background: var(--accent-soft);
            border-radius: 999px;
            padding: 4px 9px;
            font-size: 12px;
            font-weight: 750;
            white-space: nowrap;
        }
        .sentiment-bar {
            display: flex;
            height: 10px;
            overflow: hidden;
            border-radius: 999px;
            background: #eef2f7;
            margin-bottom: 12px;
        }
        .bar-positive { background: #2f9e44; }
        .bar-negative { background: #e03131; }
        .bar-neutral { background: #868e96; }
        .aspect-stats {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 8px;
        }
        .aspect-stat-label {
            color: var(--muted);
            font-size: 11px;
            margin-bottom: 2px;
        }
        .aspect-stat-value {
            color: var(--text);
            font-size: 14px;
            font-weight: 750;
        }
        .panel {
            background: #ffffff;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 14px 14px 8px 14px;
            box-shadow: 0 10px 24px rgba(30, 45, 70, .045);
            margin-bottom: 14px;
        }
        @media (max-width: 900px) {
            .aspect-grid, .sentiment-strip {
                grid-template-columns: 1fr;
            }
        }
        .column-list {
            display: block;
            padding: 11px 13px;
            background: #f8fafc;
            border: 1px solid var(--border);
            border-radius: 8px;
            color: #536078;
            font-size: 13px;
            line-height: 1.45;
            margin: 10px 0 14px 0;
        }
        .stDataFrame {
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 10px 24px rgba(30, 45, 70, .045);
        }
        div[data-testid="stTabs"] button {
            font-size: 15px;
            color: #4d5870;
        }
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: var(--accent);
            font-weight: 700;
        }
        div[data-testid="stFileUploader"] section {
            background: rgba(255,255,255,.78);
            border: 1px dashed #b7c5d8;
            border-radius: 8px;
        }
        div[data-testid="stFileUploader"] section:hover {
            border-color: var(--accent);
            background: #ffffff;
        }
        /* Русификация uploader (попытка через CSS) */
        [data-testid="stFileUploadDropzone"] > div > span {
            display: none !important;
        }
        [data-testid="stFileUploadDropzone"] > div::before {
            content: "Перетащите файл сюда";
            display: block;
            font-size: 14px;
            color: #172033;
            margin-bottom: 5px;
            font-weight: 500;
        }
        [data-testid="stFileUploadDropzone"] > div > small {
            display: none !important;
        }
        [data-testid="stFileUploadDropzone"] > div::after {
            content: "Ограничение: 200 МБ на файл • CSV, XLSX";
            display: block;
            font-size: 12px;
            color: #6f7788;
            margin-top: 5px;
        }
        .stButton > button, .stDownloadButton > button {
            border-radius: 8px;
            font-weight: 700;
        }
        .stButton > button[kind="primary"] {
            background: var(--accent);
            border-color: var(--accent);
        }
        .stAlert {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: object) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def sentiment_mini(label: str, count: int, share: float) -> str:
    return (
        '<div class="sentiment-mini">'
        f'<div class="sentiment-mini-label">{label}</div>'
        f'<div class="sentiment-mini-value">{count} / {share:.1f}%</div>'
        "</div>"
    )


def render_aspect_cards(agg_df: pd.DataFrame) -> None:
    if agg_df.empty:
        st.info("Нет аспектных данных для отображения.")
        return

    # Only show the 4 approved aspects
    display_df = agg_df[agg_df["aspect"] != "Общее впечатление"]
    if display_df.empty:
        st.info("Нет аспектных данных для отображения.")
        return

    cards = []
    for _, row in display_df.iterrows():
        positive = float(row.get("positive_share", 0))
        negative = float(row.get("negative_share", 0))
        neutral = float(row.get("neutral_share", 0))
        cards.append(
            '<div class="aspect-card">'
            '<div class="aspect-card-head">'
            f'<div class="aspect-name">{row["aspect"]}</div>'
            f'<div class="aspect-count">{int(row["mention_count"])} упомин.</div>'
            "</div>"
            '<div class="sentiment-bar">'
            f'<div class="bar-positive" style="width:{positive}%"></div>'
            f'<div class="bar-negative" style="width:{negative}%"></div>'
            f'<div class="bar-neutral" style="width:{neutral}%"></div>'
            "</div>"
            '<div class="aspect-stats">'
            "<div><div class=\"aspect-stat-label\">Позитив</div>"
            f'<div class="aspect-stat-value">{positive:.1f}%</div></div>'
            "<div><div class=\"aspect-stat-label\">Негатив</div>"
            f'<div class="aspect-stat-value">{negative:.1f}%</div></div>'
            "<div><div class=\"aspect-stat-label\">Нейтрально</div>"
            f'<div class="aspect-stat-value">{neutral:.1f}%</div></div>'
            "</div>"
            "</div>"
        )

    st.markdown('<div class="aspect-grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def section_header(kicker: str, title: str, note: str | None = None) -> None:
    note_html = f'<div class="section-note">{note}</div>' if note else ""
    st.markdown(
        f"""
        <div class="section-head">
            <div class="section-kicker">{kicker}</div>
            <h2 class="section-title">{title}</h2>
            {note_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_acquisition() -> None:
    section_header(
        "Шаг 1",
        "Получение данных",
        "Загрузите CSV или XLSX-файл с отзывами.",
    )
    uploaded_file = st.file_uploader(
        "Загрузите CSV или XLSX-файл с отзывами",
        type=["csv", "xlsx"],
        help="Поддерживаются экспортированные таблицы с отзывами маркетплейса.",
    )
    if uploaded_file is not None:
        try:
            df = load_reviews(uploaded_file)
            set_raw_data(df, uploaded_file.name)
            st.success(f"Файл загружен: {uploaded_file.name}")
        except DataLoadError as exc:
            st.error(str(exc))


def render_preview() -> str | None:
    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-title">Данные ещё не загружены</div>
                <div>После загрузки файла здесь появятся предпросмотр, выбор текстового столбца и запуск анализа.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return None

    section_header(
        "Шаг 2",
        "Предпросмотр",
        f"Источник: {st.session_state.get('source_name') or 'данные пользователя'}",
    )

    text_columns = selectable_text_columns(raw_df)
    default_index = text_columns.index("review_text") if "review_text" in text_columns else 0

    cols = st.columns(3)
    with cols[0]:
        metric_card("Строк в таблице", len(raw_df))
    with cols[1]:
        metric_card("Столбцов", len(raw_df.columns))
    with cols[2]:
        metric_card("Есть review_text", "Да" if "review_text" in raw_df.columns else "Нет")

    st.markdown('<div class="table-spacer"></div>', unsafe_allow_html=True)
    st.dataframe(
        raw_df.head(20),
        use_container_width=True,
        height=420,
    )
    st.markdown(
        f'<div class="column-list"><strong>Найденные столбцы:</strong> {", ".join(map(str, raw_df.columns))}</div>',
        unsafe_allow_html=True,
    )
    text_column = st.selectbox(
        "Выберите столбец с текстом отзыва",
        options=text_columns,
        index=default_index,
    )

    is_valid, errors = validate_reviews_df(raw_df, text_column=text_column)
    for error in errors:
        st.error(error)
    return text_column if is_valid else None


def run_analysis(text_column: str) -> None:
    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        st.warning("Сначала загрузите данные.")
        return

    try:
        prepared_df = prepare_reviews_dataframe(raw_df, text_column=text_column)
        processed_df = preprocess_reviews(prepared_df)
    except Exception:
        st.error("Произошла ошибка при подготовке данных. Проверьте структуру файла.")
        return

    valid_df = processed_df[processed_df["is_valid"]].copy()
    if valid_df.empty:
        st.error("Нет валидных отзывов для анализа.")
        st.session_state["processed_df"] = processed_df
        return

    model = (
        os.getenv("GIGACHAT_MODEL")
        or os.getenv("LLM_MODEL")
        or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    provider = (os.getenv("LLM_PROVIDER") or "openai").strip()
    base_url = (
        os.getenv("GIGACHAT_BASE_URL")
        or os.getenv("LLM_BASE_URL")
        or ""
    ).strip()
    api_key = None
    batch_size = 10
    progress = st.progress(0)
    status = st.empty()

    def on_progress(done: int, total: int) -> None:
        progress.progress(done / total)
        status.caption(f"Проанализировано {done} из {total} отзывов")

    try:
        with st.status("Выполняется анализ тональности по аспектам...", expanded=False) as analysis_status:
            results_df = analyze_reviews(
                valid_df,
                config=AnalyzerConfig(
                    model=model,
                    batch_size=batch_size,
                    base_url=base_url or None,
                    api_key=api_key,
                    provider=provider,
                ),
                progress_callback=on_progress,
            )
            agg_df = aggregate_results(results_df)
            wide_df = to_wide_format(results_df, valid_df)
            analysis_status.update(label="Анализ завершён", state="complete")
    except AnalyzerError as exc:
        st.error(str(exc))
        return
    except Exception:  # pragma: no cover
        st.error("Анализ прерван из-за внутренней ошибки обработки. Повторите попытку.")
        return

    st.session_state["processed_df"] = processed_df
    st.session_state["results_df"] = results_df
    st.session_state["agg_df"] = agg_df
    st.session_state["wide_df"] = wide_df
    progress.progress(1.0)
    st.success("Результаты готовы.")


def render_analysis_button(text_column: str | None) -> None:
    if st.session_state.get("raw_df") is None:
        return

    section_header(
        "Шаг 3",
        "Запуск анализа",
        "После запуска система обработает валидные отзывы и построит таблицы с дашбордом.",
    )

    disabled = text_column is None or st.session_state.get("raw_df") is None
    if st.button("Запустить анализ", type="primary", disabled=disabled, use_container_width=True):
        run_analysis(text_column or "review_text")


def render_results() -> None:
    results_df = st.session_state.get("results_df")
    agg_df = st.session_state.get("agg_df")
    processed_df = st.session_state.get("processed_df")
    raw_df = st.session_state.get("raw_df")
    wide_df = st.session_state.get("wide_df")

    if results_df is None or agg_df is None:
        return

    # Build wide_df if not yet present (backwards compatibility)
    if wide_df is None:
        valid_df = processed_df[processed_df["is_valid"]].copy() if processed_df is not None else None
        wide_df = to_wide_format(results_df, valid_df)
        st.session_state["wide_df"] = wide_df

    section_header(
        "Итоги",
        "Результаты анализа",
        "Сводные показатели, карточки аспектов, графики и экспорт.",
    )

    # Filter out "Общее впечатление" for metrics
    aspect_results = results_df[results_df["aspect"] != "Общее впечатление"] if "aspect" in results_df.columns else results_df

    invalid_count = 0
    if processed_df is not None and "is_valid" in processed_df.columns:
        invalid_count = int((~processed_df["is_valid"]).sum())
    valid_count = len(processed_df) - invalid_count if processed_df is not None else 0
    total_mentions = len(aspect_results)
    negative_share = round(float((aspect_results["sentiment"] == "negative").mean() * 100), 1) if total_mentions else 0.0

    cols = st.columns(4)
    with cols[0]:
        metric_card("Всего отзывов", len(raw_df) if raw_df is not None else 0)
    with cols[1]:
        metric_card("Валидных отзывов", valid_count)
    with cols[2]:
        metric_card("Аспектных упоминаний", total_mentions)
    with cols[3]:
        metric_card("Доля негатива", f"{negative_share}%")

    sentiment_counts = aspect_results["sentiment_label"].fillna("Нейтральная").value_counts() if total_mentions else pd.Series(dtype=int)
    st.markdown(
        '<div class="sentiment-strip">'
        + sentiment_mini("Положительная", int(sentiment_counts.get("Положительная", 0)), float((aspect_results["sentiment"] == "positive").mean() * 100) if total_mentions else 0)
        + sentiment_mini("Отрицательная", int(sentiment_counts.get("Отрицательная", 0)), float((aspect_results["sentiment"] == "negative").mean() * 100) if total_mentions else 0)
        + sentiment_mini("Нейтральная", int(sentiment_counts.get("Нейтральная", 0)), float((aspect_results["sentiment"] == "neutral").mean() * 100) if total_mentions else 0)
        + "</div>",
        unsafe_allow_html=True,
    )

    section_header("Аспекты", "Карточки аспектов", "Доли тональности считаются внутри каждого аспекта.")
    render_aspect_cards(agg_df)

    section_header("Аналитика", "Аналитический дашборд", "Графики показывают распределение тональности по аспектам и помогают определить проблемные зоны.")
    chart_left, chart_right = st.columns([1.55, 1])
    with chart_left:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(build_sentiment_share_chart(agg_df), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    with chart_right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.plotly_chart(build_overall_sentiment_donut(results_df), use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.plotly_chart(build_negative_rate_chart(agg_df), use_container_width=True, config={"displayModeBar": False})
    st.markdown("</div>", unsafe_allow_html=True)

    # Summary table (aggregates without confidence)
    section_header("Сводка", "Таблица аспектов", "Проценты округлены до одного знака.")
    summary_df = agg_df[agg_df["aspect"] != "Общее впечатление"].rename(
        columns={
            "aspect": "Аспект",
            "mention_count": "Упоминаний",
            "positive_count": "Позитив",
            "negative_count": "Негатив",
            "neutral_count": "Нейтрально",
            "positive_share": "Позитив, %",
            "negative_share": "Негатив, %",
            "neutral_share": "Нейтрально, %",
        }
    )[
        [
            "Аспект",
            "Упоминаний",
            "Позитив, %",
            "Негатив, %",
            "Нейтрально, %",
        ]
    ]
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # Wide-format results table
    section_header("Детализация", "Таблица результатов анализа", "Одна строка соответствует одному отзыву. В столбцах показаны результаты анализа по аспектам.")
    display_wide = wide_df.rename(
        columns={
            "review_id": "ID отзыва",
            "product_name": "Товар",
            "rating": "Оценка",
            "date": "Дата",
            "review_text": "Текст отзыва",
            "quality": "Качество товара",
            "description_match": "Соответствие описанию",
            "packaging": "Упаковка",
            "delivery": "Доставка",
            "overall_sentiment": "Общая тональность",
        }
    )
    display_columns = [
        column
        for column in ["ID отзыва", "Товар", "Оценка", "Дата", "Текст отзыва", "Качество товара", "Соответствие описанию", "Упаковка", "Доставка", "Общая тональность"]
        if column in display_wide.columns
    ]
    st.dataframe(display_wide[display_columns], use_container_width=True, hide_index=True, height=420)

    # Export
    section_header("Экспорт", "Выгрузка результатов", "CSV и XLSX содержат таблицу результатов в широком формате.")
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Скачать результаты CSV",
            data=to_csv_bytes(wide_df),
            file_name="review_analysis_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Скачать полный XLSX",
            data=to_xlsx_bytes(raw_df, processed_df, wide_df, agg_df),
            file_name="review_analysis_report.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )


def main() -> None:
    load_dotenv()
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()
    inject_css()

    st.markdown(
        f"""
        <section class="hero">
            <div class="hero-content">
                <div class="eyebrow">АНАЛИЗ ОТЗЫВОВ</div>
                <h1>{APP_TITLE}</h1>
                <p>Рабочее пространство для загрузки отзывов, анализа тональности по аспектам, просмотра аналитики и выгрузки результатов.</p>
                <div class="hero-steps">
                    <span class="step-pill"><span class="step-index">1</span>Загрузка CSV/XLSX</span>
                    <span class="step-pill"><span class="step-index">2</span>Аспектный анализ</span>
                    <span class="step-pill"><span class="step-index">3</span>Дашборд</span>
                    <span class="step-pill"><span class="step-index">4</span>Экспорт</span>
                </div>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    render_data_acquisition()
    text_column = render_preview()
    render_analysis_button(text_column)
    render_results()


if __name__ == "__main__":
    main()
