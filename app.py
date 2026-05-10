from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from modules.aggregation import aggregate_results
from modules.data_loader import (
    DataLoadError,
    load_reviews,
    prepare_reviews_dataframe,
    selectable_text_columns,
    validate_reviews_df,
)
from modules.export import to_csv_bytes, to_xlsx_bytes
from modules.openai_analyzer import AnalyzerConfig, AnalyzerError, analyze_reviews
from modules.preprocessing import preprocess_reviews
from modules.visualization import (
    build_aspect_mentions_chart,
    build_confidence_chart,
    build_date_sentiment_chart,
    build_sentiment_distribution_chart,
)
from modules.wb_parser import WBParserError, extract_nm_id, fetch_and_save_wb_reviews


APP_TITLE = "Система анализа тональности отзывов маркетплейса"


def init_state() -> None:
    defaults = {
        "raw_df": None,
        "processed_df": None,
        "results_df": None,
        "agg_df": None,
        "source_name": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def clear_analysis() -> None:
    for key in ("processed_df", "results_df", "agg_df"):
        st.session_state[key] = None


def set_raw_data(df: pd.DataFrame, source_name: str) -> None:
    st.session_state["raw_df"] = df
    st.session_state["source_name"] = source_name
    clear_analysis()


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at 15% 10%, rgba(44, 123, 229, .10), transparent 28%),
                linear-gradient(180deg, #f7f9fc 0%, #eef3f8 100%);
        }
        .main .block-container {
            padding-top: 2rem;
            max-width: 1180px;
        }
        .hero {
            background: #ffffff;
            border: 1px solid #d8e1eb;
            border-radius: 8px;
            padding: 28px 30px;
            box-shadow: 0 14px 36px rgba(31, 45, 61, .08);
            margin-bottom: 22px;
        }
        .hero h1 {
            margin: 0 0 8px 0;
            font-size: 34px;
            line-height: 1.15;
            letter-spacing: 0;
            color: #172033;
        }
        .hero p {
            margin: 0;
            color: #4b587c;
            font-size: 16px;
        }
        .metric-card {
            background: #ffffff;
            border: 1px solid #d8e1eb;
            border-radius: 8px;
            padding: 16px 18px;
            min-height: 96px;
        }
        .metric-label {
            color: #5c6780;
            font-size: 13px;
            margin-bottom: 8px;
        }
        .metric-value {
            color: #172033;
            font-size: 28px;
            font-weight: 700;
        }
        div[data-testid="stTabs"] button {
            font-size: 15px;
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


def render_data_acquisition() -> None:
    st.subheader("Получение данных")
    source_tab, wb_tab = st.tabs(["Загрузка файла", "Парсер Wildberries"])

    with source_tab:
        uploaded_file = st.file_uploader(
            "Загрузите CSV или XLSX-файл с отзывами",
            type=["csv", "xlsx"],
            help="Текстовый столбец можно выбрать после загрузки.",
        )
        if uploaded_file is not None:
            try:
                df = load_reviews(uploaded_file)
                set_raw_data(df, uploaded_file.name)
                st.success(f"Файл загружен: {uploaded_file.name}")
            except DataLoadError as exc:
                st.error(str(exc))

    with wb_tab:
        st.caption("Публичный сбор отзывов без API-ключа продавца. Доступность зависит от ответа Wildberries.")
        wb_url = st.text_input(
            "Ссылка на товар Wildberries",
            placeholder="https://www.wildberries.ru/catalog/5870243/detail.aspx",
        )
        limit = st.number_input(
            "Максимум отзывов",
            min_value=10,
            max_value=1000,
            value=100,
            step=10,
        )
        if st.button("Получить отзывы WB", use_container_width=True):
            try:
                nm_id = extract_nm_id(wb_url)
                with st.status(f"Получаю отзывы по артикулу {nm_id}...", expanded=True) as status:
                    df, saved_path = fetch_and_save_wb_reviews(wb_url, limit=int(limit))
                    status.update(label=f"Готово: сохранено {len(df)} отзывов", state="complete")
                set_raw_data(df, str(saved_path))
                st.success(f"Отзывы сохранены в {saved_path}")
            except WBParserError as exc:
                st.error(str(exc))


def render_preview() -> str | None:
    raw_df = st.session_state.get("raw_df")
    if raw_df is None:
        st.info("Загрузите файл или получите отзывы Wildberries, чтобы продолжить.")
        return None

    st.subheader("Предпросмотр")
    st.caption(f"Источник: {st.session_state.get('source_name') or 'данные пользователя'}")
    cols = st.columns(3)
    with cols[0]:
        metric_card("Строк в таблице", len(raw_df))
    with cols[1]:
        metric_card("Столбцов", len(raw_df.columns))
    with cols[2]:
        metric_card("Есть review_text", "Да" if "review_text" in raw_df.columns else "Нет")

    st.dataframe(raw_df.head(20), use_container_width=True)
    st.write("Найденные столбцы:", ", ".join(map(str, raw_df.columns)))

    text_columns = selectable_text_columns(raw_df)
    default_index = text_columns.index("review_text") if "review_text" in text_columns else 0
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
    except Exception as exc:
        st.error(f"Не удалось подготовить данные: {exc}")
        return

    valid_df = processed_df[processed_df["is_valid"]].copy()
    if valid_df.empty:
        st.error("Нет валидных отзывов для анализа.")
        st.session_state["processed_df"] = processed_df
        return

    model = (
        st.session_state.get("llm_model")
        or os.getenv("LLM_MODEL")
        or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    )
    base_url = (st.session_state.get("llm_base_url") or os.getenv("LLM_BASE_URL") or "").strip()
    api_key = (st.session_state.get("llm_api_key") or "").strip() or None
    batch_size = int(st.session_state.get("batch_size", 10))
    progress = st.progress(0)
    status = st.empty()

    def on_progress(done: int, total: int) -> None:
        progress.progress(done / total)
        status.caption(f"Проанализировано {done} из {total} отзывов")

    try:
        with st.status("Выполняется аспектный анализ...", expanded=False) as analysis_status:
            results_df = analyze_reviews(
                valid_df,
                config=AnalyzerConfig(
                    model=model,
                    batch_size=batch_size,
                    base_url=base_url or None,
                    api_key=api_key,
                ),
                progress_callback=on_progress,
            )
            agg_df = aggregate_results(results_df)
            analysis_status.update(label="Анализ завершён", state="complete")
    except AnalyzerError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # pragma: no cover - defensive UI guard
        st.error(f"Ошибка анализа: {exc}")
        return

    st.session_state["processed_df"] = processed_df
    st.session_state["results_df"] = results_df
    st.session_state["agg_df"] = agg_df
    progress.progress(1.0)
    st.success("Результаты готовы.")


def render_analysis_controls(text_column: str | None) -> None:
    if st.session_state.get("raw_df") is None:
        return

    st.subheader("Запуск анализа")
    st.caption("После запуска приложение обработает валидные отзывы и построит таблицы с дашбордом.")

    with st.expander("Расширенные настройки NLP-модуля", expanded=False):
        st.caption(
            "Можно использовать OpenAI или OpenAI-compatible API. "
            "Если поля оставить пустыми, настройки будут взяты из `.env`."
        )
        provider_cols = st.columns([1, 1])
        with provider_cols[0]:
            st.session_state["llm_model"] = st.text_input(
                "Модель",
                value=st.session_state.get(
                    "llm_model",
                    os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or "gpt-4o-mini",
                ),
                help="Например: gpt-4o-mini, openai/gpt-4o-mini, deepseek-chat.",
            )
        with provider_cols[1]:
            st.session_state["llm_base_url"] = st.text_input(
                "Base URL API",
                value=st.session_state.get("llm_base_url", os.getenv("LLM_BASE_URL", "")),
                placeholder="Оставьте пустым для OpenAI",
                help="Для OpenAI-compatible провайдеров: https://openrouter.ai/api/v1 и т.п.",
            )
        st.session_state["llm_api_key"] = st.text_input(
            "API-ключ",
            value=st.session_state.get("llm_api_key", ""),
            type="password",
            placeholder="Можно оставить пустым, если ключ задан в .env",
            help="Если заполнено, используется только в текущей сессии Streamlit и не экспортируется.",
        )
        st.session_state["batch_size"] = st.number_input(
            "Размер батча",
            min_value=1,
            max_value=30,
            value=int(st.session_state.get("batch_size", 10)),
            step=1,
        )

    disabled = text_column is None or st.session_state.get("raw_df") is None
    if st.button("Запустить анализ", type="primary", disabled=disabled, use_container_width=True):
        run_analysis(text_column or "review_text")


def render_results() -> None:
    results_df = st.session_state.get("results_df")
    agg_df = st.session_state.get("agg_df")
    processed_df = st.session_state.get("processed_df")
    raw_df = st.session_state.get("raw_df")

    if results_df is None or agg_df is None:
        return

    overview_tab, table_tab, dashboard_tab, export_tab = st.tabs(
        ["Обзор", "Результаты по отзывам", "Дашборд", "Экспорт"]
    )

    with overview_tab:
        invalid_count = 0
        if processed_df is not None and "is_valid" in processed_df.columns:
            invalid_count = int((~processed_df["is_valid"]).sum())
        negative_share = 0.0
        if not results_df.empty:
            negative_share = round(float((results_df["sentiment"] == "negative").mean() * 100), 1)

        cols = st.columns(4)
        with cols[0]:
            metric_card("Всего отзывов", len(raw_df) if raw_df is not None else 0)
        with cols[1]:
            metric_card("Валидных отзывов", len(processed_df) - invalid_count if processed_df is not None else 0)
        with cols[2]:
            metric_card("Аспектных упоминаний", len(results_df))
        with cols[3]:
            metric_card("Доля негатива", f"{negative_share}%")

        st.dataframe(agg_df, use_container_width=True)

    with table_tab:
        filtered = results_df.copy()
        col1, col2 = st.columns(2)
        with col1:
            aspects = ["Все"] + sorted(filtered["aspect"].dropna().unique().tolist())
            selected_aspect = st.selectbox("Аспект", aspects)
        with col2:
            sentiments = ["Все"] + sorted(filtered["sentiment_label"].dropna().unique().tolist())
            selected_sentiment = st.selectbox("Тональность", sentiments)

        if selected_aspect != "Все":
            filtered = filtered[filtered["aspect"] == selected_aspect]
        if selected_sentiment != "Все":
            filtered = filtered[filtered["sentiment_label"] == selected_sentiment]

        st.dataframe(filtered, use_container_width=True, hide_index=True)

    with dashboard_tab:
        st.plotly_chart(build_aspect_mentions_chart(agg_df), use_container_width=True)
        st.plotly_chart(build_sentiment_distribution_chart(agg_df), use_container_width=True)
        st.plotly_chart(build_confidence_chart(agg_df), use_container_width=True)
        st.plotly_chart(build_date_sentiment_chart(results_df), use_container_width=True)

    with export_tab:
        st.caption("XLSX содержит листы с исходными, обработанными, итоговыми и агрегированными данными.")
        col1, col2 = st.columns(2)
        with col1:
            st.download_button(
                "Скачать результаты CSV",
                data=to_csv_bytes(results_df),
                file_name="review_analysis_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with col2:
            st.download_button(
                "Скачать полный XLSX",
                data=to_xlsx_bytes(raw_df, processed_df, results_df, agg_df),
                file_name="review_analysis_report.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    init_state()
    inject_css()

    st.markdown(
        f"""
        <section class="hero">
            <h1>{APP_TITLE}</h1>
            <p>Загрузка отзывов, аспектный анализ, аналитика по проблемным зонам и экспорт результатов.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

    render_data_acquisition()
    text_column = render_preview()
    render_analysis_controls(text_column)
    render_results()


if __name__ == "__main__":
    main()
