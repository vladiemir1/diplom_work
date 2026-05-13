# Контекст проекта для следующего AI-агента

Дата составления: 10 мая 2026.

Проект: `diplom_mvp`.

## Коротко о проекте

Это MVP для ВКР: локальное Streamlit-приложение на Python для анализа тональности пользовательских отзывов маркетплейса. Пользователь загружает CSV/XLSX с отзывами, выбирает текстовый столбец, запускает аспектный анализ через внешний LLM/NLP-модуль, смотрит таблицы и графики, затем экспортирует результаты в CSV или XLSX.

Тема ВКР: разработка системы анализа тональности пользовательских отзывов на маркетплейсе с использованием нейронных сетей.

Главная идея: продавец или аналитик маркетплейса не читает отзывы вручную, а быстро видит проблемные зоны по аспектам:

- качество товара;
- соответствие описанию;
- упаковка;
- доставка;
- общее впечатление.

В MVP нейросетевой модуль реализован через внешний LLM API. Архитектура оставляет возможность позже заменить его на локальную модель, например RuBERT или другой классификатор.

## Финальное ТЗ

Финальное техническое задание лежит в файле `final_tz_marketplace_review_analyzer.md`.

Требования из ТЗ:

- локальное веб-приложение на Streamlit;
- язык разработки Python;
- загрузка CSV/XLSX-файлов с отзывами;
- выбор текстового столбца пользователем;
- предварительная обработка текста;
- аспектный анализ отзывов через LLM API;
- один отзыв может дать несколько аспектных строк результата;
- поддержка OpenAI, OpenAI-compatible API и GigaChat;
- учет рейтинга как дополнительного сигнала при анализе;
- отображение результатов в таблицах и графиках;
- экспорт результатов в CSV и XLSX;
- хранение состояния в `st.session_state`;
- разделение UI-логики и аналитического модуля;
- базовые unit-тесты.

Дополнительное требование: сбор публичных отзывов Wildberries по ссылке или артикулу без WB API-ключа продавца. Сейчас эта часть реализована отдельным CLI-скриптом, а не встроена в основной Streamlit-интерфейс.

## Что уже сделано

Реализовано:

- основной Streamlit-интерфейс в `app.py`;
- загрузка CSV/XLSX;
- автоматическое чтение CSV в кодировках `utf-8`, `utf-8-sig`, `cp1251`;
- выбор текстового столбца, если `review_text` отсутствует;
- нормализация входной таблицы к контракту приложения;
- автоматическое создание `review_id`, если его нет;
- нормализация даты через `pd.to_datetime`;
- предобработка текста: удаление управляющих символов, схлопывание пробелов, поиск пустых отзывов;
- аспектный анализ через OpenAI-compatible клиент;
- поддержка GigaChat через OAuth-обмен Authorization Key на access token;
- строгая JSON Schema для ответа модели и fallback на JSON mode;
- нормализация некорректных/неполных ответов модели;
- агрегация результатов по аспектам;
- расчет долей positive/negative/neutral и средней уверенности;
- графики Plotly;
- экспорт CSV с BOM `utf-8-sig`;
- экспорт XLSX с листами исходных, обработанных, итоговых и агрегированных данных;
- CLI-парсер публичных отзывов Wildberries;
- unit-тесты для загрузки данных, предобработки, агрегации, экспорта, LLM-модуля, GigaChat и WB-парсера;
- пример входных данных в `data/sample_reviews.csv`;
- пример результата анализа в `review_analysis_results.csv`;
- краткий ВКР-контекст в `docs/vkr_context_summary.md`;
- README с инструкциями запуска и настройки.

## Структура проекта

```text
diplom_mvp/
├── app.py
├── requirements.txt
├── README.md
├── PROJECT_CONTEXT.md
├── final_tz_marketplace_review_analyzer.md
├── tz_marketplace_review_analyzer.md
├── review_analysis_results.csv
├── ВКР_3_ГЛАВА.docx
├── data/
│   └── sample_reviews.csv
├── docs/
│   └── vkr_context_summary.md
├── modules/
│   ├── __init__.py
│   ├── aggregation.py
│   ├── data_loader.py
│   ├── export.py
│   ├── openai_analyzer.py
│   ├── preprocessing.py
│   ├── visualization.py
│   └── wb_parser.py
├── scripts/
│   └── parse_wb_reviews.py
└── tests/
    ├── __init__.py
    ├── test_basic.py
    ├── test_openai_analyzer.py
    ├── test_parse_wb_reviews_script.py
    └── test_wb_parser.py
```

## Основной пользовательский сценарий

1. Пользователь запускает приложение:

   ```bash
   streamlit run app.py
   ```

2. В интерфейсе загружает CSV или XLSX с отзывами.

3. Приложение показывает предпросмотр первых строк, количество строк/столбцов и список найденных колонок.

4. Пользователь выбирает столбец с текстом отзыва.

5. В расширенных настройках пользователь при необходимости выбирает провайдера NLP-модуля:

   - `OpenAI`;
   - `OpenAI-compatible`;
   - `GigaChat`.

6. Пользователь запускает анализ.

7. Приложение:

   - приводит таблицу к внутреннему формату;
   - очищает текст;
   - исключает пустые отзывы из анализа;
   - отправляет валидные отзывы в LLM батчами;
   - получает один или несколько аспектов на отзыв;
   - агрегирует результаты;
   - сохраняет данные в `st.session_state`.

8. Интерфейс показывает:

   - метрики по числу отзывов и аспектных упоминаний;
   - долю негатива;
   - карточки аспектов;
   - stacked bar chart по тональности аспектов;
   - donut chart общей тональности;
   - bar chart аспектов по доле негатива;
   - таблицу агрегатов;
   - детальную таблицу результатов с фильтрами.

9. Пользователь скачивает:

   - `review_analysis_results.csv`;
   - `review_analysis_report.xlsx`.

## Входные данные

Поддерживаются CSV и XLSX.

Ожидаемые поля:

| Поле | Обязательность | Описание |
|---|---|---|
| `review_id` | необязательное | идентификатор отзыва |
| `product_name` | необязательное | название товара |
| `rating` | необязательное | числовая оценка |
| `review_text` | необязательное при ручном выборе | текст отзыва |
| `date` | необязательное | дата отзыва |

Если `review_text` отсутствует, пользователь выбирает другой текстовый столбец. Если `review_id` отсутствует, приложение создает последовательные ID от `1` до `N`.

Пример входного файла: `data/sample_reviews.csv`.

## Выходные данные

Предобработанная таблица содержит:

```text
review_id, product_name, rating, date, review_text, processed_text, is_valid, error_message
```

Итоговая таблица анализа содержит:

```text
review_id, product_name, rating, date, review_text, processed_text, aspect, sentiment, sentiment_label, confidence, explanation
```

Агрегированная таблица содержит:

```text
aspect, mention_count, positive_count, negative_count, neutral_count,
positive_share, negative_share, neutral_share, negative_rate, confidence_avg
```

Пример итогового CSV уже есть в корне: `review_analysis_results.csv`.

## Аспекты и тональность

Аспекты:

- `Качество товара`;
- `Соответствие описанию`;
- `Упаковка`;
- `Доставка`;
- `Общее впечатление`.

Технические значения тональности:

- `positive`;
- `negative`;
- `neutral`.

Русские подписи в интерфейсе:

- `Положительная`;
- `Отрицательная`;
- `Нейтральная`.

`Общее впечатление` используется как fallback, если модель не выделила конкретный аспект или вернула аспект вне разрешенного списка.

## Модульная архитектура

### `app.py`

Главный Streamlit-интерфейс.

Ключевые функции:

- `init_state()` - инициализирует `st.session_state`;
- `clear_analysis()` - очищает результаты при загрузке новых данных;
- `set_raw_data()` - кладет исходную таблицу и имя источника в состояние;
- `inject_css()` - добавляет CSS интерфейса;
- `render_data_acquisition()` - блок загрузки файла;
- `render_preview()` - предпросмотр таблицы и выбор текстового столбца;
- `run_analysis()` - основной запуск подготовки, анализа и агрегации;
- `render_analysis_controls()` - расширенные настройки NLP и кнопка запуска;
- `render_results()` - метрики, графики, таблицы, фильтры, экспорт;
- `main()` - сборка страницы.

Важный факт: в текущем `app.py` нет встроенной формы для парсинга Wildberries. WB-сбор реализован отдельно через CLI.

### `modules/data_loader.py`

Отвечает за загрузку и нормализацию входных файлов.

Ключевые функции:

- `load_reviews(source)` - читает CSV/XLSX из Streamlit UploadedFile или локального пути;
- `validate_reviews_df(df, text_column)` - проверяет, что таблица не пустая и выбранный текстовый столбец существует;
- `selectable_text_columns(df)` - возвращает подходящие текстовые колонки;
- `ensure_review_id(df)` - добавляет `review_id`, если его нет;
- `normalize_date_column(df)` - приводит `date` к datetime;
- `prepare_reviews_dataframe(df, text_column)` - приводит таблицу к контракту приложения.

### `modules/preprocessing.py`

Предобработка текста.

Ключевые функции:

- `clean_review_text(value)` - заменяет управляющие символы пробелами, схлопывает whitespace, обрезает края;
- `preprocess_reviews(df, text_column="review_text")` - добавляет `processed_text`, `is_valid`, `error_message`.

Пустой отзыв получает:

```text
is_valid = False
error_message = "Пустой текст отзыва"
```

### `modules/openai_analyzer.py`

Главный аналитический LLM-модуль.

Ключевые сущности:

- `AnalyzerConfig` - модель, batch size, temperature, base URL, API key, provider;
- `AnalyzerError` - пользовательская ошибка анализа;
- `ASPECTS` - разрешенные аспекты;
- `SENTIMENTS` - разрешенные значения тональности;
- `SENTIMENT_LABELS` - русские подписи тональности.

Ключевые функции:

- `load_api_key()` - получает ключ из явного параметра, `LLM_API_KEY` или `OPENAI_API_KEY`;
- `is_gigachat_config()` - определяет, что используется GigaChat;
- `get_gigachat_access_token()` - получает access token GigaChat через OAuth;
- `build_response_schema()` - JSON Schema ожидаемого ответа;
- `_build_client()` - создает OpenAI-compatible клиент;
- `_extract_response_text()` - извлекает текст из разных форматов ответа клиента;
- `_parse_json_response()` - парсит чистый JSON, JSON из markdown fence или JSON внутри лишнего текста;
- `_call_openai_batch()` - отправляет батч отзывов в модель;
- `_normalize_model_items()` - нормализует результат модели;
- `_reviews_to_records()` - готовит отзывы для payload;
- `analyze_reviews()` - публичная функция аспектного анализа.

Поведение `analyze_reviews()`:

- если входная таблица пустая, возвращает пустую таблицу с ожидаемыми колонками;
- проверяет наличие столбца анализа, по умолчанию `processed_text`;
- строит клиента, если он не передан;
- отправляет отзывы батчами;
- вызывает callback прогресса;
- для каждого результата добавляет исходные поля отзыва;
- если модель ничего не вернула по отзыву, добавляет нейтральное `Общее впечатление`.

Prompt требует от модели:

- вернуть только JSON;
- найти один или несколько аспектов на отзыв;
- использовать только разрешенные аспекты;
- учитывать `rating` как дополнительный сигнал;
- вернуть `confidence` от `0` до `1`;
- писать краткое русское объяснение.

### `modules/aggregation.py`

Агрегация аспектных результатов.

`aggregate_results(results_df)` группирует данные по `aspect` и считает:

- число упоминаний;
- количество positive/negative/neutral;
- доли positive/negative/neutral в процентах;
- `negative_rate`;
- среднюю уверенность `confidence_avg`.

Пустая таблица возвращается с полным набором ожидаемых колонок.

### `modules/visualization.py`

Plotly-графики.

Ключевые функции:

- `build_sentiment_share_chart(agg_df)` - stacked bar chart долей тональности по аспектам;
- `build_negative_rate_chart(agg_df)` - horizontal bar chart аспектов по доле негатива;
- `build_overall_sentiment_donut(results_df)` - donut chart общей тональности.

Если данных нет, строится пустой график с текстом `Нет данных для отображения`.

### `modules/export.py`

Экспорт.

Ключевые функции:

- `to_csv_bytes(df)` - экспортирует итоговую таблицу в CSV bytes с `utf-8-sig`;
- `to_xlsx_bytes(raw_df, processed_df, results_df, agg_df)` - формирует Excel bytes с листами:
  - `Исходные данные`;
  - `Обработанные данные`;
  - `Результаты анализа`;
  - `Агрегаты`.

### `modules/wb_parser.py`

Парсер публичных отзывов Wildberries.

Ключевые функции:

- `extract_nm_id(value)` - извлекает артикул WB из ссылки, параметра или строки с цифрами;
- `_request_feedbacks(nm_id)` - пробует несколько публичных endpoint Wildberries;
- `_extract_review_text(feedback)` - объединяет `text`, `pros`, `cons`;
- `feedbacks_to_dataframe(data, nm_id, limit)` - переводит JSON WB в DataFrame;
- `fetch_wb_reviews(product_url, limit)` - получает DataFrame отзывов;
- `save_wb_reviews_csv(df, nm_id, output_dir)` - сохраняет `data/wb_reviews_<nmId>.csv`;
- `fetch_and_save_wb_reviews(product_url, limit, output_dir)` - получает и сохраняет отзывы.

Ограничение: парсер зависит от публичных endpoint Wildberries. Если WB изменит формат ответа или ограничит доступ, будет ошибка `WBParserError`.

### `scripts/parse_wb_reviews.py`

CLI-обертка для WB-парсера.

Примеры:

```bash
python scripts/parse_wb_reviews.py "https://www.wildberries.ru/catalog/5870243/detail.aspx"
python scripts/parse_wb_reviews.py 5870243 --limit 50
python scripts/parse_wb_reviews.py 5870243 --limit 50 --output-dir data
```

По умолчанию сохраняет до 400 текстовых отзывов.

## Настройки окружения

Зависимости:

```text
streamlit
pandas
numpy
plotly
scikit-learn
openpyxl
requests
python-dotenv
openai
pytest
python-docx
```

Установка:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

OpenAI или OpenAI-compatible:

```env
LLM_API_KEY=ваш_ключ
LLM_MODEL=gpt-4o-mini
LLM_BASE_URL=
```

Старые имена переменных OpenAI тоже поддерживаются:

```env
OPENAI_API_KEY=ваш_ключ
OPENAI_MODEL=gpt-4o-mini
```

GigaChat:

```env
LLM_PROVIDER=gigachat
GIGACHAT_CREDENTIALS=ваш_authorization_key
GIGACHAT_MODEL=GigaChat
GIGACHAT_BASE_URL=https://gigachat.devices.sberbank.ru/api/v1
```

Для локальной разработки при проблеме сертификата:

```env
GIGACHAT_VERIFY_SSL=false
```

Важно: реальные ключи из локального `.env` не нужно вставлять в документацию, README, тесты или экспортируемые таблицы.

## Тесты

Запуск:

```bash
pytest
```

Что покрыто:

- загрузка CSV;
- загрузка XLSX;
- создание `review_id`;
- выбор текстовых колонок;
- предобработка и пустые отзывы;
- агрегация результатов;
- экспорт CSV/XLSX;
- мокированный OpenAI-compatible анализ;
- fallback на `Общее впечатление`;
- отсутствие API-ключа;
- определение GigaChat-конфига;
- OAuth-токен GigaChat через мок `requests.post`;
- парсинг JSON из markdown/лишнего текста;
- извлечение WB nmId;
- преобразование WB feedbacks в DataFrame;
- мокированный fetch and save WB;
- CLI `--help` для скрипта парсинга.

## Важные текущие ограничения

- Парсер Wildberries реализован как CLI-скрипт, но не встроен в основной Streamlit UI.
- Анализ зависит от доступности внешнего LLM API.
- Без API-ключа анализ не запустится, но тесты LLM-модуля используют мок и не отправляют реальные запросы.
- Качество аспектной классификации зависит от выбранной модели.
- Нет локальной ML-модели в репозитории. RuBERT упомянут как направление развития.
- Нет отдельного слоя постоянного хранения результатов, кроме скачивания файлов и текущего `st.session_state`.
- `.env` есть локально, но его значения не должны попадать в git или документацию.
- ВКР-документ `ВКР_3_ГЛАВА.docx` используется как контекст. По заметке в `docs/vkr_context_summary.md`, в нем есть незавершенные места: реферат, список источников и комментарии к диаграммам.

## Что можно делать дальше

Приоритетные возможные доработки:

1. Встроить WB-парсер в Streamlit-интерфейс отдельным блоком рядом с загрузкой файла.
2. Добавить сохранение истории анализов или загрузку ранее полученного результата.
3. Добавить локальную модель анализа тональности вместо внешнего LLM.
4. Расширить тесты Streamlit-логики, насколько это удобно для текущего стека.
5. Добавить e2e smoke-тест запуска приложения.
6. Улучшить README, если финальная версия должна строго соответствовать ВКР.
7. Синхронизировать формулировку ТЗ и фактическую реализацию WB-парсера: сейчас ТЗ говорит о публичном сборе, а реализация находится в CLI.

## Быстрый ориентир для нового агента

Если нужно понять проект за несколько минут, читать в таком порядке:

1. `PROJECT_CONTEXT.md` - этот файл.
2. `final_tz_marketplace_review_analyzer.md` - финальное ТЗ.
3. `README.md` - запуск и пользовательская документация.
4. `app.py` - пользовательский поток Streamlit.
5. `modules/openai_analyzer.py` - аналитический LLM-модуль.
6. `modules/data_loader.py`, `modules/preprocessing.py`, `modules/aggregation.py` - основной data pipeline.
7. `modules/wb_parser.py` и `scripts/parse_wb_reviews.py` - Wildberries.
8. `tests/` - подтверждение ожидаемого поведения.

## Команды для проверки

```bash
pytest
streamlit run app.py
python scripts/parse_wb_reviews.py --help
```

