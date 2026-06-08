from datetime import datetime
from pathlib import Path
import subprocess
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import (
    STOCKS_PATH,
    NEWS_RAW_PATH,
    NEWS_PATH,
    NEWS_24H_RESULTS_PATH,
    NEWS_24H_SIGNALS_PATH,
    NEWS_24H_PREDICTIONS_PATH,
    NEWS_24H_COMPARISON_RESULTS_PATH,
    NEWS_24H_COMPARISON_SIGNALS_PATH,
    NEWS_24H_COMPARISON_PREDICTIONS_PATH,
    NEWS_MEMORY_DIRECTION_RESULTS_PATH,
    NEWS_MEMORY_DIRECTION_PREDICTIONS_PATH,
    NEWS_MEMORY_DIRECTION_SIGNALS_PATH,
    MAGNITUDE_RESULTS_PATH,
    MAGNITUDE_PREDICTIONS_PATH,
    MAGNITUDE_SOURCE_ANALYSIS_PATH,
    RETURN_24H_RESULTS_PATH,
    RETURN_24H_PREDICTIONS_PATH,
    CALENDAR_RETURN_24H_RESULTS_PATH,
    CALENDAR_RETURN_24H_PREDICTIONS_PATH,
    ABNORMAL_RETURN_24H_RESULTS_PATH,
    ABNORMAL_RETURN_24H_PREDICTIONS_PATH,
    FUTURE_RETURN_24H_PREDICTIONS_PATH,
    FUTURE_ABNORMAL_24H_PREDICTIONS_PATH,
    MARKET_NEWS_ATTRIBUTION_PATH,
    MARKET_NEWS_DRIVER_SUMMARY_PATH,
    MARKET_NEWS_TICKER_SUMMARY_PATH,
    DEFAULT_TICKERS,
)
from src.data_loader import (
    load_stocks,
    load_news,
    load_results,
    load_news_24h_predictions,
    load_magnitude_predictions,
)
from src.market_data import update_stocks_file
from src.news_pipeline import update_news_features_pipeline
from src.visualization import (
    plot_price_chart_with_news_24h_signals,
    get_news_24h_signals_table,
)


MAIN_DATA_FILES = {
    STOCKS_PATH: "часовые котировки MOEX",
    NEWS_PATH: "новости с признаками",
}

OLD_DIRECTION_RESULT_FILES = [
    (
        "archive/old_main_model_20260517/final_model_outputs/diploma_experiment_results.csv",
        "Старая 4ч-модель",
    ),
    (
        "archive/retrain_event_threshold_60_20260514/final_model_outputs_before_retrain/diploma_experiment_results.csv",
        "Старая 1ч-модель",
    ),
    (
        "archive/cascade_direction_20260514/final_model_outputs_before_cascade/diploma_experiment_results.csv",
        "Архивная cascade-модель",
    ),
]


def get_file_mtime(path):
    file_path = Path(path)
    if not file_path.exists():
        return None
    return file_path.stat().st_mtime


def get_missing_main_data_files():
    missing_files = []
    for file_path, description in MAIN_DATA_FILES.items():
        if not Path(file_path).exists():
            missing_files.append((file_path, description))
    return missing_files


def update_project_data(
    update_market,
    update_news,
    market_date_from=None,
    news_date_from=None,
    use_heavy_news_features=False,
):
    if update_market:
        stocks_updated = update_stocks_file(
            tickers=DEFAULT_TICKERS,
            output_path=STOCKS_PATH,
            timeframe="1h",
            date_from=market_date_from,
        )
        st.success(f"Котировки обновлены: {len(stocks_updated)} строк")

    if update_news:
        news_features_updated = update_news_features_pipeline(
            raw_news_path=NEWS_RAW_PATH,
            features_path=NEWS_PATH,
            date_from=news_date_from,
            date_till=None,
            use_rubert=use_heavy_news_features,
            use_sentiment=use_heavy_news_features,
            rubert_batch_size=64,
            sentiment_batch_size=64,
            sentiment_max_length=256,
        )
        st.success(
            f"Новости и признаки обновлены: {len(news_features_updated)} строк в parquet"
        )


def get_python_executable():
    venv_python = Path(".venv/bin/python")
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def retrain_news_24h_model():
    command = [
        get_python_executable(),
        "scripts/train_main_news_24h_model.py",
    ]
    return subprocess.run(
        command,
        cwd=Path(".").resolve(),
        capture_output=True,
        text=True,
        timeout=60 * 30,
    )


def get_date_range_label(df, date_column, empty_label="данных нет"):
    if df is None or df.empty or date_column not in df.columns:
        return empty_label

    dates = pd.to_datetime(df[date_column], errors="coerce").dropna()
    if dates.empty:
        return empty_label

    date_min = dates.min().strftime("%Y-%m-%d")
    date_max = dates.max().strftime("%Y-%m-%d")
    return f"с {date_min} по {date_max}"


@st.cache_data
def load_old_direction_results(file_mtimes):
    frames = []
    for file_path, source_name in OLD_DIRECTION_RESULT_FILES:
        path = Path(file_path)
        if not path.exists():
            continue

        frame = pd.read_csv(path)
        if "task" in frame.columns:
            frame = frame[frame["task"] == "direction_prediction"].copy()
        if frame.empty:
            continue

        frame["source"] = source_name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def format_news_24h_model_name(value):
    names = {
        "logreg_balanced": "LogReg balanced",
        "hist_gradient_boosting": "HistGradientBoosting",
        "extra_trees": "ExtraTrees",
        "random_forest": "RandomForest",
        "gradient_boosting": "GradientBoosting",
        "catboost": "CatBoost",
        "mlp": "MLP",
        "ridge": "Ridge",
    }
    return names.get(value, value)


def format_news_24h_feature_set(value):
    names = {
        "news_only": "Только новости",
        "price_only_on_news": "Цена/рынок в момент новости",
        "news_plus_price": "Новости + цена/рынок",
    }
    return names.get(value, value)


def format_direction_feature_set(value):
    names = {
        "news_only": "Только новости",
        "news_memory": "Новости + память тикера",
        "price_only_on_news": "Цена/рынок",
        "news_plus_price": "Новости + цена/рынок",
        "news_price_memory": "Новости + цена/рынок + память",
    }
    return names.get(value, value)


def format_magnitude_feature_set(value):
    names = {
        "price_only_on_news": "Цена/рынок в момент новости",
        "news_only": "Только новости",
        "news_source": "Новости + источник",
        "news_memory": "Новости + память тикера",
        "news_plus_price": "Новости + цена/рынок",
        "news_plus_market": "Новости + рыночный фон",
        "news_plus_imoex": "Новости + IMOEX/RTSI",
        "imoex_only_on_news": "Только IMOEX/RTSI",
        "news_source_plus_price": "Новости + источник + цена/рынок",
        "news_memory_plus_price": "Новости + память + цена/рынок",
    }
    return names.get(value, value)


def format_magnitude_task(value):
    names = {
        "magnitude_classification": "Классы силы движения",
        "magnitude_regression": "Регрессия величины движения",
    }
    return names.get(value, value)


def make_model_label(df):
    label = df["Модель"].astype(str)
    if "Набор признаков" in df.columns:
        label = df["Набор признаков"].astype(str) + " / " + label
    return label


def plot_event_model_comparison(event_table):
    if event_table.empty:
        return go.Figure()

    plot_df = event_table.copy()
    plot_df["Модель и признаки"] = make_model_label(plot_df)

    fig = px.scatter(
        plot_df,
        x="pr_auc",
        y="precision_at_5pct",
        size="roc_auc",
        color="Модель",
        symbol="Набор признаков",
        hover_data={
            "Модель и признаки": True,
            "roc_auc": ":.3f",
            "pr_auc": ":.3f",
            "precision_at_5pct": ":.1%",
            "precision_at_10pct": ":.1%",
            "f1": ":.3f",
            "Модель": False,
            "Набор признаков": False,
        },
        labels={
            "pr_auc": "PR-AUC",
            "precision_at_5pct": "Precision@5%",
            "roc_auc": "ROC-AUC",
        },
        title="Event detection: качество сильных новостных сигналов",
    )
    fig.update_layout(
        template="plotly_dark",
        height=520,
        margin=dict(l=20, r=20, t=70, b=20),
        legend_title_text="Модель / признаки",
    )
    fig.update_traces(marker=dict(opacity=0.85, line=dict(width=1, color="#f5f5f5")))
    fig.update_yaxes(tickformat=".0%", gridcolor="rgba(255,255,255,0.12)")
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.12)")
    return fig


def plot_direction_model_comparison(direction_table):
    if direction_table.empty:
        return go.Figure()

    plot_df = direction_table.copy()
    plot_df["Модель и признаки"] = make_model_label(plot_df)
    plot_df["precision_at_5pct"] = plot_df["precision_at_5pct"].fillna(0)

    fig = px.scatter(
        plot_df,
        x="f1",
        y="roc_auc",
        size="precision_at_5pct",
        color="Постановка",
        symbol="Модель",
        hover_data={
            "Модель и признаки": True,
            "Горизонт": True,
            "Порог события": True,
            "pr_auc": ":.3f",
            "precision_at_5pct": ":.1%",
            "precision_at_10pct": ":.1%",
            "balanced_accuracy": ":.3f",
            "Постановка": False,
            "Модель": False,
        },
        labels={
            "f1": "F1",
            "roc_auc": "ROC-AUC",
            "precision_at_5pct": "Precision@5%",
        },
        title="Direction: старые и новые постановки",
    )
    fig.add_hline(
        y=0.5,
        line_dash="dash",
        line_color="rgba(255,255,255,0.45)",
        annotation_text="случайный уровень ROC-AUC",
    )
    fig.update_layout(
        template="plotly_dark",
        height=560,
        margin=dict(l=20, r=20, t=70, b=20),
        legend_title_text="Постановка / модель",
    )
    fig.update_traces(marker=dict(opacity=0.82, line=dict(width=1, color="#f5f5f5")))
    fig.update_xaxes(range=[0.35, 0.72], gridcolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(range=[0.48, 0.66], gridcolor="rgba(255,255,255,0.12)")
    return fig


def get_actual_direction(signals):
    if "first_hit_direction" in signals.columns:
        return signals["first_hit_direction"].fillna(signals["target_direction_24h"]) == 1
    return signals["target_direction_24h"] == 1


def add_direction_filter_columns(signals, direction_confidence_min):
    if signals.empty:
        return signals.copy(), signals.copy()

    confident_signals = signals[
        signals["direction_confidence"] >= direction_confidence_min
    ].copy()
    neutral_signals = signals[
        signals["direction_confidence"] < direction_confidence_min
    ].copy()
    return confident_signals, neutral_signals


def apply_direction_model(signals, direction_predictions, feature_set, model_name):
    if signals.empty or direction_predictions is None or direction_predictions.empty:
        return signals

    direction = direction_predictions[
        (direction_predictions["feature_set"] == feature_set)
        & (direction_predictions["model"] == model_name)
    ].copy()
    if direction.empty:
        return signals

    direction = direction[
        [
            "security",
            "start_time",
            "first_hit_direction",
            "first_hit_up_proba",
            "direction_confidence",
        ]
    ].drop_duplicates(subset=["security", "start_time"])
    direction = direction.rename(columns={"direction_confidence": "new_direction_confidence"})

    out = signals.merge(
        direction,
        on=["security", "start_time"],
        how="left",
    )
    out["direction_up_proba"] = out["first_hit_up_proba"].fillna(out["direction_up_proba"])
    out["direction_confidence"] = out["new_direction_confidence"].fillna(
        (out["direction_up_proba"] - 0.5).abs()
    )
    return out.drop(
        columns=["first_hit_up_proba", "new_direction_confidence"],
        errors="ignore",
    )


def render_metric_explainer(movement_threshold, signal_threshold):
    with st.expander("Как читать эти метрики"):
        st.markdown(
            f"""
            **Событие** здесь означает, что после новости по тикеру в течение следующих
            24 часов цена сходила вверх или вниз минимум на **{movement_threshold:.1%}**.

            **ROC-AUC** показывает, насколько хорошо модель в целом отделяет новости
            с будущим сильным движением от новостей без такого движения. Значение `0.5`
            похоже на случайное угадывание, чем выше — тем лучше.

            **PR-AUC** сильнее фокусируется именно на классе событий. Эта метрика полезна,
            когда нас интересуют не все часы подряд, а именно редкие сильные движения.

            **Precision@5%** — качество самых уверенных 5% сигналов модели. Например,
            `87.4%` означает, что среди верхних по уверенности сигналов примерно 87 из 100
            действительно сопровождались движением не меньше выбранного порога.

            **Сигналы** — это количество моментов, где вероятность события выше выбранного
            порога сигнала. Сейчас порог сигнала равен **{signal_threshold:.2f}**.

            **Факт событий** — доля реальных событий среди всех сигналов, которые прошли
            выбранный порог. Это практическая проверка качества стрелок на графике.
            """
        )


def render_news_24h_page(
    stocks,
    results,
    signal_slices,
    predictions,
    direction_results=None,
    direction_predictions=None,
):
    st.subheader("Новостные сигналы на горизонте 24 часа")

    if predictions is None or results is None or predictions.empty:
        st.info(
            "Файлы финальной 24ч-модели пока не найдены. "
            "Запусти `python scripts/train_main_news_24h_model.py`."
        )
        return

    thresholds = sorted(predictions["threshold"].dropna().unique())
    feature_sets = sorted(predictions["feature_set"].dropna().unique())
    model_names = sorted(predictions["model"].dropna().unique())

    default_feature_index = (
        feature_sets.index("price_only_on_news")
        if "price_only_on_news" in feature_sets
        else 0
    )
    default_model_index = (
        model_names.index("catboost")
        if "catboost" in model_names
        else model_names.index("hist_gradient_boosting")
        if "hist_gradient_boosting" in model_names
        else 0
    )

    feature_set = st.session_state.get(
        "event24_feature_set",
        feature_sets[default_feature_index],
    )
    model_name = st.session_state.get(
        "event24_model",
        model_names[default_model_index],
    )
    if feature_set not in feature_sets:
        feature_set = feature_sets[default_feature_index]
    if model_name not in model_names:
        model_name = model_names[default_model_index]

    controls = st.columns([1, 1])
    with controls[0]:
        movement_threshold = st.selectbox(
            "Порог движения за 24ч",
            thresholds,
            format_func=lambda value: f"{value:.1%}",
            key="event24_movement_threshold",
        )
    with controls[1]:
        signal_threshold = st.slider(
            "Порог сигнала",
            min_value=0.50,
            max_value=0.90,
            value=0.70,
            step=0.05,
            key="event24_signal_threshold",
        )
    

    direction_feature_sets = (
        sorted(direction_predictions["feature_set"].dropna().unique())
        if direction_predictions is not None and not direction_predictions.empty
        else []
    )
    direction_models = (
        sorted(direction_predictions["model"].dropna().unique())
        if direction_predictions is not None and not direction_predictions.empty
        else []
    )
    direction_feature_set = None
    direction_model_name = None
    direction_confidence_min = 0.10
    if direction_feature_sets and direction_models:
        direction_feature_set = (
            "news_memory" if "news_memory" in direction_feature_sets else direction_feature_sets[0]
        )
        direction_model_name = (
            "logreg_balanced" if "logreg_balanced" in direction_models else direction_models[0]
        )

    selected = predictions[
        (predictions["threshold"] == movement_threshold)
        & (predictions["feature_set"] == feature_set)
        & (predictions["model"] == model_name)
    ].copy()
    selected["actual_event"] = selected["future_abs_return_24h"] >= movement_threshold

    selected_signals = selected[selected["event_proba"] >= signal_threshold].copy()
    selected_signals = selected_signals.sort_values("start_time", ascending=False)
    if direction_feature_set is not None and direction_model_name is not None:
        selected_signals = apply_direction_model(
            selected_signals,
            direction_predictions,
            direction_feature_set,
            direction_model_name,
        )

    result_row = results[
        (results["task"] == "event_24h")
        & (results["threshold"] == movement_threshold)
        & (results["feature_set"] == feature_set)
        & (results["model"] == model_name)
    ]

    if selected_signals.empty:
        st.warning(
            "Для выбранных параметров нет сигналов. "
            "Попробуй снизить порог сигнала или выбрать другую модель."
        )
        return

    available_tickers = selected_signals["security"].value_counts()
    chart_controls = st.columns([1.2, 1])
    with chart_controls[0]:
        selected_ticker = st.selectbox(
            "Тикер для графика",
            available_tickers.index.tolist(),
            format_func=lambda value: f"{value} ({available_tickers[value]} сигналов)",
            key="event24_ticker",
        )
    with chart_controls[1]:
        days_back = st.slider(
            "Показывать дней",
            min_value=30,
            max_value=520,
            value=360,
            step=30,
            key="event24_days_back",
        )

    st.plotly_chart(
        plot_price_chart_with_news_24h_signals(
            stocks=stocks,
            signals=selected_signals,
            ticker=selected_ticker,
            days_back=days_back,
            direction_confidence_min=direction_confidence_min,
        ),
        width="stretch",
        key="news_24h_market_signals_chart",
    )

    st.markdown("#### Параметры модели и качество")
    model_controls = st.columns([1.2, 1])
    with model_controls[0]:
        st.selectbox(
            "Набор признаков",
            feature_sets,
            index=feature_sets.index(feature_set),
            format_func=format_news_24h_feature_set,
            key="event24_feature_set",
        )
    with model_controls[1]:
        st.selectbox(
            "Модель",
            model_names,
            index=model_names.index(model_name),
            format_func=format_news_24h_model_name,
            key="event24_model",
        )

    metric_cols = st.columns(5)
    if not result_row.empty:
        best_row = result_row.iloc[0]
        metric_cols[0].metric("ROC-AUC", f"{best_row['roc_auc']:.3f}")
        metric_cols[1].metric("PR-AUC", f"{best_row['pr_auc']:.3f}")
        metric_cols[2].metric("Precision@5%", f"{best_row['precision_at_5pct']:.1%}")
    else:
        metric_cols[0].metric("ROC-AUC", "н/д")
        metric_cols[1].metric("PR-AUC", "н/д")
        metric_cols[2].metric("Precision@5%", "н/д")

    metric_cols[3].metric("Сигналы", f"{len(selected_signals)}")
    metric_cols[4].metric("Факт событий", f"{selected_signals['actual_event'].mean():.1%}")

    render_metric_explainer(movement_threshold, signal_threshold)

    st.caption("Полная таблица сигналов выбранной модели")
    st.dataframe(
        get_news_24h_signals_table(selected_signals, direction_confidence_min),
        width="stretch",
        hide_index=True,
    )

    if signal_slices is not None and not signal_slices.empty:
        st.caption("Сводка по порогам сигнала для основной модели")
        summary = signal_slices[signal_slices["threshold"] == movement_threshold].copy()
        summary["Набор признаков"] = summary["feature_set"].map(format_news_24h_feature_set)
        summary["Модель"] = summary["model"].map(format_news_24h_model_name)
        st.dataframe(
            summary[
                [
                    "Набор признаков",
                    "Модель",
                    "event_cutoff",
                    "signals",
                    "true_event_rate",
                    "hit_rate",
                    "avg_event_proba",
                    "avg_direction_confidence",
                    "avg_best_side_return",
                ]
            ].rename(columns={
                "event_cutoff": "Порог сигнала",
                "signals": "Сигналов",
                "true_event_rate": "Доля реальных событий",
                "hit_rate": "Угадано направление",
                "avg_event_proba": "Средняя вероятность события",
                "avg_direction_confidence": "Средняя уверенность направления",
                "avg_best_side_return": "Среднее движение по стороне",
            }),
            width="stretch",
            hide_index=True,
        )


def get_forecast_label(event_proba, direction_up_proba, signal_threshold):
    if event_proba < signal_threshold:
        return "Нейтрально"
    if direction_up_proba >= 0.5:
        return "Рост"
    return "Падение"


def get_forecast_color(label):
    if label == "Рост":
        return "#2ecc71"
    if label == "Падение":
        return "#ff4f45"
    return "#95a5a6"


def plot_news_24h_prediction_card(ticker, signal_row, signal_threshold):
    event_proba = float(signal_row["event_proba"])
    event_color = "#aeb7bd"

    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="number+gauge",
            value=event_proba * 100,
            title={
                "text": (
                    f"<b>{ticker}</b><br>"
                    f"Вероятность сильного движения за 24ч"
                )
            },
            number={
                "suffix": "%",
                "font": {"size": 52, "color": event_color},
            },
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": event_color},
            },
        )
    )
    fig.update_layout(
        height=330,
        template="plotly_dark",
        margin=dict(l=30, r=30, t=90, b=30),
    )
    return fig


def render_ticker_forecast_page(stocks, predictions, direction_predictions=None):
    st.subheader("Прогноз по тикеру")

    if predictions is None or predictions.empty:
        st.info("Файлы финальной 24ч-модели пока не найдены.")
        return

    thresholds = sorted(predictions["threshold"].dropna().unique())
    movement_threshold = thresholds[0]
    signal_threshold = st.slider(
        "Порог сигнала",
        min_value=0.50,
        max_value=0.90,
        value=0.70,
        step=0.05,
        key="forecast_signal_threshold",
    )

    direction_feature_sets = (
        sorted(direction_predictions["feature_set"].dropna().unique())
        if direction_predictions is not None and not direction_predictions.empty
        else []
    )
    direction_models = (
        sorted(direction_predictions["model"].dropna().unique())
        if direction_predictions is not None and not direction_predictions.empty
        else []
    )
    direction_feature_set = None
    direction_model_name = None
    if direction_feature_sets and direction_models:
        direction_cols = st.columns([1.2, 1.1, 1])
        with direction_cols[0]:
            direction_feature_set = st.selectbox(
                "Direction-признаки",
                direction_feature_sets,
                index=direction_feature_sets.index("news_memory") if "news_memory" in direction_feature_sets else 0,
                format_func=format_direction_feature_set,
                key="forecast_direction_features",
            )
        with direction_cols[1]:
            direction_model_name = st.selectbox(
                "Direction-модель",
                direction_models,
                index=direction_models.index("logreg_balanced") if "logreg_balanced" in direction_models else 0,
                format_func=format_news_24h_model_name,
                key="forecast_direction_model",
            )
        with direction_cols[2]:
            direction_confidence_min = st.slider(
                "Мин. уверенность direction",
                min_value=0.00,
                max_value=0.30,
                value=0.10,
                step=0.05,
                key="forecast_direction_confidence_min",
            )
    else:
        direction_confidence_min = 0.0

    selected = predictions[predictions["threshold"] == movement_threshold].copy()
    if direction_feature_set is not None and direction_model_name is not None:
        selected = apply_direction_model(
            selected,
            direction_predictions,
            direction_feature_set,
            direction_model_name,
        )

    if selected.empty:
        st.warning("Для выбранного порога нет новостных оценок.")
        return

    selected_signals = selected[selected["event_proba"] >= signal_threshold].copy()
    selected_signals = selected_signals.sort_values("start_time", ascending=False)
    ticker_counts = selected["security"].value_counts()
    ticker_signal_counts = selected_signals["security"].value_counts()
    ticker = st.selectbox(
        "Тикер",
        ticker_counts.index.tolist(),
        format_func=lambda value: (
            f"{value} ({ticker_signal_counts.get(value, 0)} сильных сигналов)"
        ),
        key="forecast_ticker",
    )
    days_back = st.slider(
        "Показывать дней на графике",
        min_value=30,
        max_value=520,
        value=180,
        step=30,
        key="forecast_days_back",
    )

    ticker_predictions = selected[selected["security"] == ticker].copy()
    ticker_predictions = ticker_predictions.sort_values("start_time", ascending=False)
    ticker_signals = selected_signals[selected_signals["security"] == ticker].copy()
    latest_prediction = ticker_predictions.iloc[0]

    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(
            plot_news_24h_prediction_card(ticker, latest_prediction, signal_threshold),
            width="stretch",
            key="ticker_forecast_card_chart",
        )
        st.metric("Вероятность события", f"{latest_prediction['event_proba'] * 100:.1f}%")
        if latest_prediction["direction_confidence"] >= direction_confidence_min:
            st.metric("Вероятность роста", f"{latest_prediction['direction_up_proba'] * 100:.1f}%")
        else:
            st.metric("Направление", "не показываем")
        st.metric(
            "Уверенность направления",
            f"{latest_prediction['direction_confidence'] * 100:.1f} п.п.",
        )
        st.caption(
            "Последняя новостная оценка: "
            f"{latest_prediction['start_time'].strftime('%Y-%m-%d %H:%M')}"
        )
        if ticker_signals.empty:
            st.caption("Сильных сигналов по выбранному порогу для тикера нет.")
        else:
            latest_signal_time = ticker_signals["start_time"].max().strftime("%Y-%m-%d %H:%M")
            st.caption(f"Последний сильный сигнал: {latest_signal_time}")

    with col2:
        st.plotly_chart(
            plot_price_chart_with_news_24h_signals(
                stocks=stocks,
                signals=ticker_signals,
                ticker=ticker,
                days_back=days_back,
                direction_confidence_min=direction_confidence_min,
            ),
            width="stretch",
            key="ticker_forecast_price_signals_chart",
        )

    st.caption("Сигналы выбранного тикера")
    st.dataframe(
        get_news_24h_signals_table(
            ticker_signals.sort_values("start_time", ascending=False),
            direction_confidence_min,
        ),
        width="stretch",
        hide_index=True,
    )


def get_magnitude_score(row):
    if pd.notna(row.get("proba_strong_ge_2pct")):
        return float(row["proba_strong_ge_2pct"])
    if pd.notna(row.get("predicted_magnitude")):
        return float(row["predicted_magnitude"])
    return 0.0


def make_magnitude_table(predictions):
    table = predictions.copy()
    table["score"] = table.apply(get_magnitude_score, axis=1)
    cols = [
        "start_time",
        "security",
        "news_count",
        "news_title_example",
        "future_abs_return_24h",
        "target_magnitude_label",
        "predicted_label",
        "proba_strong_ge_2pct",
        "predicted_magnitude",
        "score",
    ]
    existing_cols = [col for col in cols if col in table.columns]
    table = table[existing_cols].sort_values("score", ascending=False).copy()
    return table.rename(columns={
        "start_time": "Дата",
        "security": "Тикер",
        "news_count": "Новостей",
        "news_title_example": "Пример новости",
        "future_abs_return_24h": "Факт движения 24ч",
        "target_magnitude_label": "Факт диапазон",
        "predicted_label": "Прогноз диапазон",
        "proba_strong_ge_2pct": "Вероятность 2%+",
        "predicted_magnitude": "Прогноз движения",
        "score": "Скор",
    })


def plot_magnitude_probabilities(row):
    proba_cols = ["proba_<1%", "proba_1-2%", "proba_2-4%", "proba_>4%"]
    if not all(col in row.index and pd.notna(row[col]) for col in proba_cols):
        return go.Figure()

    labels = ["<1%", "1-2%", "2-4%", ">4%"]
    values = [float(row[col]) * 100 for col in proba_cols]
    colors = ["#95a5a6", "#f1c40f", "#ff9f43", "#ff4f45"]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=values,
            marker_color=colors,
            text=[f"{value:.1f}%" for value in values],
            textposition="outside",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        height=330,
        title="Вероятности диапазонов движения",
        yaxis_title="Вероятность, %",
        xaxis_title="Диапазон движения за 24ч",
        margin=dict(l=20, r=20, t=70, b=40),
    )
    fig.update_yaxes(range=[0, max(100, max(values) * 1.15)])
    return fig


def plot_price_chart_with_magnitude(
    stocks,
    predictions,
    ticker,
    days_back,
    scenario_direction,
    max_scenario_lines,
    scenario_horizon_hours=24,
):
    ticker_stocks_full = stocks[stocks["security"] == ticker].copy()
    ticker_predictions = predictions[predictions["security"] == ticker].copy()

    if ticker_stocks_full.empty:
        return go.Figure()

    ticker_stocks_full = ticker_stocks_full.sort_values("datetime")
    ticker_stocks = ticker_stocks_full.copy()
    end_date = ticker_stocks["datetime"].max()
    start_date = end_date - pd.Timedelta(days=days_back)
    ticker_stocks = ticker_stocks[ticker_stocks["datetime"] >= start_date]
    ticker_predictions = ticker_predictions[ticker_predictions["start_time"] >= start_date]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ticker_stocks["datetime"],
            y=ticker_stocks["close"],
            mode="lines",
            name="Цена закрытия",
            line=dict(color="#aeb7bd", width=2),
        )
    )

    if not ticker_predictions.empty:
        marker_data = ticker_predictions.merge(
            ticker_stocks[["datetime", "close"]],
            left_on="start_time",
            right_on="datetime",
            how="left",
        ).dropna(subset=["close"])
        if not marker_data.empty:
            future_rows = []
            for row in marker_data.itertuples(index=False):
                window = ticker_stocks_full[
                    (ticker_stocks_full["datetime"] > row.start_time)
                    & (ticker_stocks_full["datetime"] <= row.start_time + pd.Timedelta(hours=24))
                ]
                if window.empty or not hasattr(row, "close"):
                    future_rows.append({
                        "actual_direction": "н/д",
                        "actual_direction_symbol": "circle",
                        "actual_direction_color": "#aeb7bd",
                        "actual_max_return": None,
                        "actual_min_return": None,
                    })
                    continue

                max_return = window["high"].max() / row.close - 1
                min_return = window["low"].min() / row.close - 1
                is_up = abs(max_return) >= abs(min_return)
                future_rows.append({
                    "actual_direction": "рост" if is_up else "снижение",
                    "actual_direction_symbol": "triangle-up" if is_up else "triangle-down",
                    "actual_direction_color": "#2ecc71" if is_up else "#ff4f45",
                    "actual_max_return": max_return,
                    "actual_min_return": min_return,
                })

            marker_data = pd.concat(
                [marker_data.reset_index(drop=True), pd.DataFrame(future_rows)],
                axis=1,
            )
            marker_data["score"] = marker_data.apply(get_magnitude_score, axis=1)
            for col in [
                "news_title_example",
                "future_abs_return_24h",
                "target_magnitude_label",
                "predicted_label",
                "proba_strong_ge_2pct",
            ]:
                if col not in marker_data.columns:
                    marker_data[col] = ""
            marker_data["future_abs_return_label"] = marker_data["future_abs_return_24h"].map(
                lambda value: f"{value:.1%}" if pd.notna(value) and value != "" else ""
            )
            marker_data["proba_strong_label"] = marker_data["proba_strong_ge_2pct"].map(
                lambda value: f"{value:.1%}" if pd.notna(value) and value != "" else ""
            )
            marker_data["actual_max_return_label"] = marker_data["actual_max_return"].map(
                lambda value: f"{value:.1%}" if pd.notna(value) else ""
            )
            marker_data["actual_min_return_label"] = marker_data["actual_min_return"].map(
                lambda value: f"{value:.1%}" if pd.notna(value) else ""
            )
            marker_data["scenario_magnitude"] = marker_data["predicted_magnitude"]
            if "proba_strong_ge_2pct" in marker_data.columns:
                class_based_magnitude = (
                    marker_data["proba_1-2%"].fillna(0) * 0.015
                    + marker_data["proba_2-4%"].fillna(0) * 0.030
                    + marker_data["proba_>4%"].fillna(0) * 0.050
                )
                marker_data["scenario_magnitude"] = marker_data["scenario_magnitude"].fillna(
                    class_based_magnitude
                )
            marker_data["scenario_magnitude"] = marker_data["scenario_magnitude"].fillna(0.02)
            if scenario_direction == "growth":
                marker_data["scenario_direction_sign"] = 1
                marker_data["scenario_direction_label"] = "рост"
            elif scenario_direction == "fall":
                marker_data["scenario_direction_sign"] = -1
                marker_data["scenario_direction_label"] = "снижение"
            else:
                marker_data["scenario_direction_sign"] = marker_data["actual_direction"].map(
                    {"рост": 1, "снижение": -1}
                ).fillna(1)
                marker_data["scenario_direction_label"] = marker_data["actual_direction"]

            marker_data["scenario_target_price"] = marker_data["close"] * (
                1 + marker_data["scenario_direction_sign"] * marker_data["scenario_magnitude"]
            )
            marker_data["scenario_return_label"] = (
                marker_data["scenario_direction_sign"] * marker_data["scenario_magnitude"]
            ).map(lambda value: f"{value:+.1%}")
            marker_data = marker_data.sort_values("start_time").reset_index(drop=True)
            marker_data["next_signal_time"] = marker_data["start_time"].shift(-1)
            marker_data["scenario_full_end_time"] = (
                marker_data["start_time"] + pd.Timedelta(hours=scenario_horizon_hours)
            )
            marker_data["scenario_end_time"] = marker_data["scenario_full_end_time"]
            has_next_signal = (
                marker_data["next_signal_time"].notna()
                & (marker_data["next_signal_time"] < marker_data["scenario_full_end_time"])
            )
            marker_data.loc[has_next_signal, "scenario_end_time"] = marker_data.loc[
                has_next_signal,
                "next_signal_time",
            ]
            marker_data["scenario_elapsed_hours"] = (
                marker_data["scenario_end_time"] - marker_data["start_time"]
            ).dt.total_seconds() / 3600
            marker_data["scenario_visible_return"] = (
                marker_data["scenario_direction_sign"]
                * marker_data["scenario_magnitude"]
                * (marker_data["scenario_elapsed_hours"] / scenario_horizon_hours)
            )
            marker_data["scenario_visible_price"] = marker_data["close"] * (
                1 + marker_data["scenario_visible_return"]
            )
            marker_data["scenario_visible_return_label"] = marker_data[
                "scenario_visible_return"
            ].map(lambda value: f"{value:+.1%}")
            marker_data["scenario_elapsed_label"] = marker_data["scenario_elapsed_hours"].map(
                lambda value: f"{value:.0f}ч" if value >= 1 else f"{value:.1f}ч"
            )
            marker_data["scenario_end_label"] = "полный горизонт 24ч"
            marker_data.loc[has_next_signal, "scenario_end_label"] = (
                "перестроен через "
                + marker_data.loc[has_next_signal, "scenario_elapsed_label"]
            )
            if scenario_direction != "actual":
                marker_data["display_symbol"] = "diamond"
                marker_data["display_color"] = "#aeb7bd"
            else:
                marker_data["display_symbol"] = marker_data["actual_direction_symbol"]
                marker_data["display_color"] = marker_data["actual_direction_color"]
            scenario_data = marker_data.sort_values("start_time", ascending=False).head(
                max_scenario_lines
            )
            legend_names_shown = set()
            for scenario in scenario_data.itertuples(index=False):
                is_growth = scenario.scenario_direction_sign >= 0
                color = "rgba(46, 204, 113, 0.85)" if is_growth else "rgba(255, 79, 69, 0.85)"
                name = "Прогнозный отрезок magnitude: рост" if is_growth else "Прогнозный отрезок magnitude: снижение"
                show_legend = name not in legend_names_shown
                legend_names_shown.add(name)
                fig.add_trace(
                    go.Scatter(
                        x=[
                            scenario.start_time,
                            scenario.scenario_end_time,
                        ],
                        y=[
                            scenario.close,
                            scenario.scenario_visible_price,
                        ],
                        mode="lines+markers",
                        name=name,
                        line=dict(color=color, width=3),
                        marker=dict(size=[4, 9], symbol=["circle", "diamond"], color=color),
                        opacity=0.9,
                        showlegend=show_legend,
                        hovertemplate=(
                            "Сценарий magnitude на 24ч<br>"
                            f"Показанный участок: {scenario.close:.4f} -> {scenario.scenario_visible_price:.4f}<br>"
                            f"Движение на отрезке: {scenario.scenario_visible_return_label}<br>"
                            f"Полный 24ч таргет: {scenario.close:.4f} -> {scenario.scenario_target_price:.4f}<br>"
                            f"Полный 24ч сценарий: {scenario.scenario_return_label}<br>"
                            f"Статус: {scenario.scenario_end_label}"
                            "<extra></extra>"
                        ),
                    )
                )
            fig.add_trace(
                go.Scatter(
                    x=marker_data["start_time"],
                    y=marker_data["close"],
                    mode="markers",
                    name="Magnitude-сигнал",
                    marker=dict(
                        size=8 + marker_data["score"] * 16,
                        color=marker_data["display_color"],
                        symbol=marker_data["display_symbol"],
                        line=dict(width=1, color="#f5f5f5"),
                    ),
                    customdata=marker_data[[
                        "news_title_example",
                        "future_abs_return_label",
                        "target_magnitude_label",
                        "predicted_label",
                        "proba_strong_label",
                        "actual_direction",
                        "actual_max_return_label",
                        "actual_min_return_label",
                        "scenario_end_label",
                    ]].fillna("").to_numpy(),
                    hovertemplate=(
                        "<b>%{x|%Y-%m-%d %H:%M}</b><br>"
                        "Цена: %{y:.4f}<br>"
                        "Новость: %{customdata[0]}<br>"
                        "Факт движение: %{customdata[1]}<br>"
                        "Факт диапазон: %{customdata[2]}<br>"
                        "Прогноз диапазон: %{customdata[3]}<br>"
                        "Вероятность 2%+: %{customdata[4]}<br>"
                        "Факт направление: %{customdata[5]}<br>"
                        "Макс. рост 24ч: %{customdata[6]}<br>"
                        "Макс. снижение 24ч: %{customdata[7]}<br>"
                        "Прогнозный отрезок: %{customdata[8]}"
                        "<extra></extra>"
                    ),
                )
            )
            if scenario_direction == "actual":
                for direction, color, symbol, name in [
                    ("рост", "#2ecc71", "triangle-up", "Факт: рост"),
                    ("снижение", "#ff4f45", "triangle-down", "Факт: снижение"),
                ]:
                    fig.add_trace(
                        go.Scatter(
                            x=[None],
                            y=[None],
                            mode="markers",
                            name=name,
                            marker=dict(
                                size=12,
                                color=color,
                                symbol=symbol,
                                line=dict(width=1, color="#f5f5f5"),
                            ),
                            showlegend=True,
                        )
                    )

    fig.update_layout(
        template="plotly_dark",
        height=560,
        title=f"Котировки {ticker} и magnitude-сигналы",
        xaxis_title="Дата",
        yaxis_title="Цена",
        margin=dict(l=20, r=20, t=70, b=40),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.12)")
    return fig


def add_calendar_24h_returns(stocks, predictions):
    if predictions.empty:
        return predictions.copy()

    base_columns = ["security", "datetime", "close"]
    prices = stocks[base_columns].dropna().sort_values(["security", "datetime"]).copy()
    rows = []
    for security, group in predictions.groupby("security", sort=False):
        ticker_prices = prices[prices["security"] == security]
        if ticker_prices.empty:
            part = group.copy()
            part["calendar_actual_price_24h"] = pd.NA
            part["calendar_return_24h"] = pd.NA
            part["calendar_actual_time_24h"] = pd.NaT
            rows.append(part)
            continue

        left = group.sort_values("start_time").copy()
        right = ticker_prices.rename(columns={
            "datetime": "calendar_actual_time_24h",
            "close": "calendar_actual_price_24h",
        })
        left["calendar_target_time_24h"] = left["start_time"] + pd.Timedelta(hours=24)
        merged = pd.merge_asof(
            left.sort_values("calendar_target_time_24h"),
            right[["calendar_actual_time_24h", "calendar_actual_price_24h"]],
            left_on="calendar_target_time_24h",
            right_on="calendar_actual_time_24h",
            direction="nearest",
            tolerance=pd.Timedelta(hours=12),
        )
        merged["calendar_return_24h"] = merged["calendar_actual_price_24h"] / merged["close"] - 1
        rows.append(merged.sort_values("start_time"))

    return pd.concat(rows, ignore_index=True)


def make_return_24h_table(predictions):
    table = predictions.sort_values("start_time", ascending=False).copy()
    table["Прогноз доходности"] = (table["predicted_return"] * 100).round(2)
    actual_return_col = (
        "calendar_return_24h" if "calendar_return_24h" in table.columns else "return_after_24h"
    )
    actual_price_col = (
        "calendar_actual_price_24h"
        if "calendar_actual_price_24h" in table.columns
        else "actual_price"
    )
    table["Факт доходности"] = (table[actual_return_col] * 100).round(2)
    table["Ошибка, п.п."] = (
        (table["predicted_return"] - table[actual_return_col]).abs() * 100
    ).round(2)
    table["Сигнал"] = table["predicted_return"].map(
        lambda value: "рост" if value > 0 else "снижение"
    )
    columns = [
        "news_title_example",
        "start_time",
        "security",
        "Сигнал",
        "Прогноз доходности",
        "Факт доходности",
        "Ошибка, п.п.",
        "close",
        "predicted_price",
        actual_price_col,
        "calendar_actual_time_24h",
        "news_count",
    ]
    existing_columns = [col for col in columns if col in table.columns]
    return table[existing_columns].rename(columns={
        "start_time": "Дата",
        "security": "Тикер",
        "close": "Цена в момент новости",
        "predicted_price": "Прогноз цены модели",
        actual_price_col: "Факт цены около +24ч",
        "calendar_actual_time_24h": "Время факта",
        "news_count": "Новостей в час",
        "news_title_example": "Пример новости",
    })


def get_future_rows_for_model(future_predictions, feature_set, model_name, signal_column):
    if future_predictions is None or future_predictions.empty:
        return pd.DataFrame()

    future = future_predictions[
        (future_predictions["feature_set"] == feature_set)
        & (future_predictions["model"] == model_name)
    ].copy()
    if future.empty:
        return future

    if signal_column not in future.columns:
        signal_column = "predicted_return"
    future["forecast_strength"] = future[signal_column].abs()
    if "forecast_target_time_24h" in future.columns:
        future["forecast_target_time_24h"] = pd.to_datetime(
            future["forecast_target_time_24h"],
            errors="coerce",
        )
    if "is_future_forecast" not in future.columns:
        future["is_future_forecast"] = True
    return future


def render_future_24h_forecast_block(
    stocks,
    future_predictions,
    feature_set,
    model_name,
    signal_threshold,
    title,
    key_prefix,
    signal_column="predicted_return",
):
    st.markdown(f"#### {title}")

    if future_predictions is None or future_predictions.empty:
        st.info("Future-прогнозы пока не рассчитаны. Запусти `python updates/scripts/build_future_24h_forecasts.py`.")
        return

    future = future_predictions[
        (future_predictions["feature_set"] == feature_set)
        & (future_predictions["model"] == model_name)
    ].copy()
    if future.empty:
        st.info("Для выбранной модели нет future-прогнозов.")
        return

    if signal_column not in future.columns:
        signal_column = "predicted_return"
    future["forecast_strength"] = future[signal_column].abs()
    signal_rows = future[future["forecast_strength"] >= signal_threshold].copy()
    chart_rows = signal_rows if not signal_rows.empty else future.copy()

    strongest = future.sort_values("forecast_strength", ascending=False).iloc[0]
    metric_cols = st.columns(4)
    metric_cols[0].metric("Future-строк", f"{len(future)}")
    metric_cols[1].metric("Самый сильный тикер", strongest["security"])
    metric_cols[2].metric("Прогноз", f"{strongest[signal_column]:+.2%}")
    metric_cols[3].metric("Старт", strongest["start_time"].strftime("%Y-%m-%d %H:%M"))
    st.caption(
        "На графике future-линия строится от последней известной цены к прогнозной цене "
        "примерно через 24 часа: зеленая — рост, красная — снижение, серая — движение "
        "по модулю ниже выбранного порога."
    )

    ticker_counts = chart_rows["security"].value_counts()
    controls = st.columns([1.2, 1, 1])
    with controls[0]:
        ticker = st.selectbox(
            "Тикер future-прогноза",
            ticker_counts.index.tolist(),
            format_func=lambda value: f"{value} ({ticker_counts[value]} прогнозов)",
            key=f"{key_prefix}_future_ticker",
        )
    with controls[1]:
        days_back = st.slider(
            "Дней истории",
            min_value=7,
            max_value=120,
            value=30,
            step=7,
            key=f"{key_prefix}_future_days_back",
        )
    with controls[2]:
        max_lines = st.slider(
            "Future-отрезков",
            min_value=1,
            max_value=30,
            value=min(10, max(1, len(chart_rows))),
            step=1,
            key=f"{key_prefix}_future_max_lines",
        )

    st.plotly_chart(
        plot_price_chart_with_return_24h(
            stocks=stocks,
            predictions=chart_rows,
            ticker=ticker,
            days_back=days_back,
            signal_threshold=signal_threshold,
            max_lines=max_lines,
            signal_column=signal_column,
            show_neutral_forecasts=True,
        ),
        width="stretch",
        key=f"{key_prefix}_future_return_chart",
    )

    st.caption("Future-прогнозы выбранной модели")
    st.dataframe(
        make_return_24h_table(chart_rows).head(100),
        width="stretch",
        hide_index=True,
    )


def plot_price_chart_with_return_24h(
    stocks,
    predictions,
    ticker,
    days_back,
    signal_threshold,
    max_lines,
    signal_column="predicted_return",
    show_neutral_forecasts=False,
):
    ticker_stocks_full = stocks[stocks["security"] == ticker].copy()
    ticker_predictions = predictions[predictions["security"] == ticker].copy()

    if ticker_stocks_full.empty:
        return go.Figure()

    ticker_stocks_full = ticker_stocks_full.sort_values("datetime")
    end_date = ticker_stocks_full["datetime"].max()
    start_date = end_date - pd.Timedelta(days=days_back)
    ticker_stocks = ticker_stocks_full[ticker_stocks_full["datetime"] >= start_date].copy()
    ticker_predictions = ticker_predictions[ticker_predictions["start_time"] >= start_date].copy()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=ticker_stocks["datetime"],
            y=ticker_stocks["close"],
            mode="lines",
            name="Цена закрытия",
            line=dict(color="#aeb7bd", width=2),
        )
    )

    if ticker_predictions.empty:
        fig.update_layout(
            template="plotly_dark",
            height=560,
            title=f"Котировки {ticker} и прогноз доходности около +24ч",
            xaxis_title="Дата",
            yaxis_title="Цена",
            margin=dict(l=20, r=20, t=70, b=40),
        )
        return fig

    marker_data = ticker_predictions.merge(
        ticker_stocks[["datetime", "close"]],
        left_on="start_time",
        right_on="datetime",
        how="left",
        suffixes=("", "_chart"),
    ).dropna(subset=["close_chart"])

    if marker_data.empty:
        return fig

    marker_data["close"] = marker_data["close_chart"]
    if "is_future_forecast" in marker_data.columns:
        marker_data["is_future_forecast"] = marker_data["is_future_forecast"].fillna(False).astype(bool)
    else:
        marker_data["is_future_forecast"] = False
    if signal_column not in marker_data.columns:
        signal_column = "predicted_return"
    marker_data["predicted_return_for_chart"] = marker_data[signal_column]
    marker_data["is_signal"] = marker_data["predicted_return_for_chart"].abs() >= signal_threshold
    if "news_title_example" not in marker_data.columns:
        marker_data["news_title_example"] = ""
    marker_data["news_title_example"] = marker_data["news_title_example"].fillna("")
    if not show_neutral_forecasts:
        marker_data = marker_data[marker_data["is_signal"]].copy()

    if not marker_data.empty:
        marker_data = marker_data.sort_values("start_time").reset_index(drop=True)
        marker_data["next_signal_time"] = marker_data["start_time"].shift(-1)
        marker_data["full_end_time"] = marker_data["start_time"] + pd.Timedelta(hours=24)
        if "forecast_target_time_24h" in marker_data.columns:
            marker_data["forecast_target_time_24h"] = pd.to_datetime(
                marker_data["forecast_target_time_24h"],
                errors="coerce",
            )
            if "is_future_forecast" in marker_data.columns:
                future_flags = marker_data["is_future_forecast"].fillna(False).astype(bool)
            else:
                future_flags = pd.Series(False, index=marker_data.index)
            future_mask = future_flags & marker_data["forecast_target_time_24h"].notna()
            marker_data.loc[future_mask, "full_end_time"] = marker_data.loc[
                future_mask,
                "forecast_target_time_24h",
            ]
        marker_data["end_time"] = marker_data["full_end_time"]
        interrupted = (
            marker_data["next_signal_time"].notna()
            & (marker_data["next_signal_time"] < marker_data["full_end_time"])
        )
        marker_data.loc[interrupted, "end_time"] = marker_data.loc[
            interrupted,
            "next_signal_time",
        ]
        marker_data["elapsed_hours"] = (
            marker_data["end_time"] - marker_data["start_time"]
        ).dt.total_seconds() / 3600
        marker_data["visible_return"] = marker_data["predicted_return_for_chart"] * marker_data["elapsed_hours"] / 24
        marker_data["visible_price"] = marker_data["close"] * (1 + marker_data["visible_return"])
        marker_data["target_price"] = marker_data["close"] * (1 + marker_data["predicted_return_for_chart"])
        marker_data["predicted_return_label"] = marker_data["predicted_return_for_chart"].map(
            lambda value: f"{value:+.2%}"
        )
        marker_data["visible_return_label"] = marker_data["visible_return"].map(
            lambda value: f"{value:+.2%}"
        )
        actual_return_col = (
            "calendar_return_24h"
            if "calendar_return_24h" in marker_data.columns
            else "return_after_24h"
        )
        marker_data["actual_return_label"] = marker_data[actual_return_col].map(
            lambda value: f"{value:+.2%}" if pd.notna(value) else ""
        )
        marker_data["elapsed_label"] = marker_data["elapsed_hours"].map(
            lambda value: f"{value:.0f}ч" if value >= 1 else f"{value:.1f}ч"
        )
        marker_data["status"] = "полный горизонт визуализации"
        marker_data.loc[interrupted, "status"] = (
            "перестроен через " + marker_data.loc[interrupted, "elapsed_label"]
        )

        scenario_data = marker_data.sort_values("start_time", ascending=False).head(max_lines)
        shown_names = set()
        for signal in scenario_data.itertuples(index=False):
            is_future = bool(getattr(signal, "is_future_forecast", False))
            if abs(signal.predicted_return_for_chart) < signal_threshold:
                color = "rgba(174, 183, 189, 0.8)"
                direction_name = "стабильно"
            elif signal.predicted_return_for_chart > 0:
                color = "rgba(46, 204, 113, 0.9)"
                direction_name = "рост"
            else:
                color = "rgba(255, 79, 69, 0.9)"
                direction_name = "снижение"
            name = (
                f"Будущий прогноз: {direction_name}"
                if is_future
                else f"Прогноз доходности: {direction_name}"
            )
            show_legend = name not in shown_names
            shown_names.add(name)
            fig.add_trace(
                go.Scatter(
                    x=[signal.start_time, signal.end_time],
                    y=[signal.close, signal.visible_price],
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=4 if is_future else 3, dash="dash" if is_future else "solid"),
                    showlegend=show_legend,
                    hovertemplate=(
                        "Прогноз доходности около +24ч<br>"
                        f"Показанный участок: {signal.close:.4f} -> {signal.visible_price:.4f}<br>"
                        f"Движение на отрезке: {signal.visible_return_label}<br>"
                        f"Полный таргет: {signal.close:.4f} -> {signal.target_price:.4f}<br>"
                        f"Прогноз модели: {signal.predicted_return_label}<br>"
                        f"Факт около +24ч: {signal.actual_return_label}<br>"
                        f"Новость: {signal.news_title_example}<br>"
                        f"Статус: {signal.status}"
                        "<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        template="plotly_dark",
        height=560,
        title=f"Котировки {ticker} и прогноз доходности около +24ч",
        xaxis_title="Дата",
        yaxis_title="Цена",
        margin=dict(l=20, r=20, t=70, b=40),
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.12)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.12)")
    return fig


def render_return_24h_page(stocks, return_results, return_predictions, future_predictions=None):
    st.subheader("Прогноз доходности около +24 часов")

    if return_results is None or return_predictions is None or return_predictions.empty:
        st.info(
            "Файлы эксперимента с регрессией доходности пока не найдены. "
            "Нужен `updates/model_outputs/fixed_horizon_return_experiment_fast`."
        )
        return

    predictions_24h = return_predictions[return_predictions["horizon"] == 24].copy()
    results_24h = return_results[return_results["horizon"] == 24].copy()
    if predictions_24h.empty:
        st.warning("В prediction-файле нет горизонта 24 часа.")
        return

    feature_sets = sorted(predictions_24h["feature_set"].dropna().unique())
    model_names = sorted(predictions_24h["model"].dropna().unique())
    default_feature_name = (
        "news_plus_imoex"
        if "news_plus_imoex" in feature_sets
        else "news_plus_price"
    )
    default_feature = (
        feature_sets.index(default_feature_name)
        if default_feature_name in feature_sets
        else 0
    )
    default_model = model_names.index("catboost") if "catboost" in model_names else 0

    feature_set = st.session_state.get(
        "return24_feature_set",
        feature_sets[default_feature],
    )
    model_name = st.session_state.get(
        "return24_model",
        model_names[default_model],
    )
    if feature_set not in feature_sets:
        feature_set = feature_sets[default_feature]
    if model_name not in model_names:
        model_name = model_names[default_model]

    signal_threshold = st.slider(
        "Порог сценария по модулю",
        min_value=0.001,
        max_value=0.020,
        value=0.001,
        step=0.001,
        format="%.3f",
        key="return24_signal_threshold",
    )
    selected = predictions_24h[
        (predictions_24h["feature_set"] == feature_set)
        & (predictions_24h["model"] == model_name)
    ].copy()
    selected = add_calendar_24h_returns(stocks, selected)
    selected_results = results_24h[
        (results_24h["feature_set"] == feature_set)
        & (results_24h["model"] == model_name)
    ].copy()

    selected["is_signal"] = selected["predicted_return"].abs() >= signal_threshold
    signal_rows = selected[selected["is_signal"]].copy()
    future_rows = get_future_rows_for_model(
        future_predictions,
        feature_set,
        model_name,
        "predicted_return",
    )

    if selected.empty:
        st.warning("Для выбранной модели нет строк прогнозов.")
        return

    if signal_rows.empty and future_rows.empty:
        st.warning(
            "Для выбранного порога модель не дала сценариев. "
            "Снизь порог или выбери другую модель."
        )
        chart_candidates = selected.copy()
    else:
        chart_candidates = pd.concat([signal_rows, future_rows], ignore_index=True)

    ticker_counts = chart_candidates["security"].value_counts()
    chart_controls = st.columns([1.2, 1, 1])
    with chart_controls[0]:
        default_ticker = None
        if not future_rows.empty:
            default_ticker = (
                future_rows.sort_values("forecast_strength", ascending=False)
                .iloc[0]["security"]
            )
        ticker_options = ticker_counts.index.tolist()
        default_ticker_index = (
            ticker_options.index(default_ticker)
            if default_ticker in ticker_options
            else 0
        )
        selected_ticker = st.selectbox(
            "Тикер для графика",
            ticker_options,
            index=default_ticker_index,
            format_func=lambda value: f"{value} ({ticker_counts[value]} строк)",
            key="return24_ticker",
        )
    with chart_controls[1]:
        days_back = st.slider(
            "Показывать дней",
            min_value=30,
            max_value=520,
            value=360,
            step=30,
            key="return24_days_back",
        )
    with chart_controls[2]:
        max_lines = st.slider(
            "Отрезков",
            min_value=1,
            max_value=30,
            value=10,
            step=1,
            key="return24_max_lines",
        )

    st.plotly_chart(
        plot_price_chart_with_return_24h(
            stocks=stocks,
            predictions=chart_candidates,
            ticker=selected_ticker,
            days_back=days_back,
            signal_threshold=signal_threshold,
            max_lines=max_lines,
            signal_column="predicted_return",
            show_neutral_forecasts=not future_rows.empty,
        ),
        width="stretch",
        key="return24_calendar_chart",
    )

    st.markdown("#### Параметры модели и качество")
    model_controls = st.columns([1.2, 1])
    with model_controls[0]:
        st.selectbox(
            "Набор признаков",
            feature_sets,
            index=feature_sets.index(feature_set),
            format_func=format_magnitude_feature_set,
            key="return24_feature_set",
        )
    with model_controls[1]:
        st.selectbox(
            "Модель",
            model_names,
            index=model_names.index(model_name),
            format_func=format_news_24h_model_name,
            key="return24_model",
        )

    latest = selected.sort_values("start_time", ascending=False).iloc[0]
    metric_cols = st.columns(6)
    if not selected_results.empty:
        row = selected_results.iloc[0]
        metric_cols[0].metric("MAE", f"{row['mae_pct']:.2f}%")
        metric_cols[1].metric("RMSE", f"{row['rmse_pct']:.2f}%")
        metric_cols[2].metric("R2", f"{row['r2']:.3f}")
    else:
        metric_cols[0].metric("MAE", "н/д")
        metric_cols[1].metric("RMSE", "н/д")
        metric_cols[2].metric("R2", "н/д")

    metric_cols[3].metric("Сценарии", f"{len(signal_rows)}")
    metric_cols[4].metric("Future", f"{len(future_rows)}")
    if signal_rows.empty:
        metric_cols[5].metric("Hit-rate сценариев", "н/д")
    else:
        hit_rate = (
            (signal_rows["predicted_return"] >= 0)
            == (signal_rows["calendar_return_24h"] >= 0)
        ).mean()
        metric_cols[5].metric("Hit-rate сценариев", f"{hit_rate:.1%}")

    info_cols = st.columns(4)
    info_cols[0].metric("Последняя строка", latest["security"])
    info_cols[1].metric("Прогноз модели", f"{latest['predicted_return']:+.2%}")
    latest_calendar_return = latest.get("calendar_return_24h")
    info_cols[2].metric(
        "Факт около +24ч",
        f"{latest_calendar_return:+.2%}" if pd.notna(latest_calendar_return) else "н/д",
    )
    info_cols[3].metric("Дата", latest["start_time"].strftime("%Y-%m-%d %H:%M"))

    with st.expander("Как выбирать параметры"):
        st.markdown(
            """
            **Для защиты:** `Новости + цена/рынок`, `CatBoost`, порог `0.001`, 180-360 дней истории,
            10-20 отрезков. Так видно и исторические сценарии, и future-прогноз.

            **Порог сценария** отвечает только за визуализацию: чем выше порог, тем меньше
            отрезков останется на графике. Метрики модели от него не меняются.

            **Отрезки** — это количество последних сценариев на графике. Если график перегружен,
            поставь 10. Если нужно показать больше примеров, поставь 20-30.
            """
        )

    with st.expander("Как читать прогноз доходности"):
        st.markdown(
            """
            Зеленый отрезок означает прогноз роста за сутки, красный — прогноз снижения.
            Future-прогнозы рисуются на этом же графике пунктиром; если прогноз ниже
            порога по модулю, он показывается серым как стабильный сценарий.

            Линия показывает сценарий до цены, соответствующей прогнозной доходности.
            Если раньше появляется новый прогноз по тому же тикеру, старый отрезок
            обрезается и начинается новый.
            """
        )

    st.caption("Строки со сценариями по выбранному порогу")
    st.dataframe(
        make_return_24h_table(signal_rows).head(300),
        width="stretch",
        hide_index=True,
    )


def render_best_calendar_return_page(stocks, calendar_results, calendar_predictions):
    st.subheader("Best model 24h: прогноз доходности около +24 часов")

    if calendar_results is None or calendar_predictions is None or calendar_predictions.empty:
        st.info(
            "Файлы сравнения календарных 24ч-моделей пока не найдены. "
            "Запусти `python updates/scripts/run_calendar_24h_return_experiment.py --fast --max-train-rows 5000`."
        )
        return

    test_results = calendar_results[calendar_results["split"] == "test"].copy()
    test_predictions = calendar_predictions[calendar_predictions["split"] == "test"].copy()
    if test_results.empty or test_predictions.empty:
        st.warning("В файлах календарного 24ч-эксперимента нет test-части.")
        return

    best_overall = test_results.sort_values(["mae_pct", "rmse_pct"]).iloc[0]
    news_candidates = test_results[test_results["feature_set"] == "news_plus_price"]
    best_news = (
        news_candidates.sort_values(["mae_pct", "rmse_pct"]).iloc[0]
        if not news_candidates.empty
        else best_overall
    )

    mode = st.selectbox(
        "Какую модель показать",
        ["best_news", "best_overall"],
        format_func=lambda value: {
            "best_news": "Лучшая модель с новостями",
            "best_overall": "Лучшая модель по MAE",
        }[value],
        key="best_calendar_mode",
    )
    selected_row = best_news if mode == "best_news" else best_overall
    feature_set = selected_row["feature_set"]
    model_name = selected_row["model"]

    selected_predictions = test_predictions[
        (test_predictions["feature_set"] == feature_set)
        & (test_predictions["model"] == model_name)
    ].copy()
    selected_predictions["return_after_24h"] = selected_predictions["calendar_return_24h"]
    selected_predictions["actual_price"] = selected_predictions["calendar_actual_price_24h"]

    signal_threshold = st.slider(
        "Порог сценария по модулю",
        min_value=0.001,
        max_value=0.020,
        value=0.001,
        step=0.001,
        format="%.3f",
        key="best_calendar_signal_threshold",
    )
    signal_rows = selected_predictions[
        selected_predictions["predicted_return"].abs() >= signal_threshold
    ].copy()

    metric_cols = st.columns(6)
    metric_cols[0].metric("Модель", format_news_24h_model_name(model_name))
    metric_cols[1].metric("Признаки", format_magnitude_feature_set(feature_set))
    metric_cols[2].metric("MAE", f"{selected_row['mae_pct']:.2f}%")
    metric_cols[3].metric("RMSE", f"{selected_row['rmse_pct']:.2f}%")
    metric_cols[4].metric("Hit-rate", f"{selected_row['direction_hit_rate']:.1%}")
    if signal_rows.empty:
        metric_cols[5].metric("Сценарии", "0")
    else:
        signal_hit_rate = (
            (signal_rows["predicted_return"] >= 0)
            == (signal_rows["calendar_return_24h"] >= 0)
        ).mean()
        metric_cols[5].metric("Сценарии", f"{len(signal_rows)} / {signal_hit_rate:.1%}")

    with st.expander("Что выбрано и почему"):
        st.markdown(
            f"""
            **Лучшая по MAE на test:** `{best_overall['feature_set']} + {best_overall['model']}`,
            MAE = `{best_overall['mae_pct']:.2f}%`.

            **Лучшая модель с новостями:** `{best_news['feature_set']} + {best_news['model']}`,
            MAE = `{best_news['mae_pct']:.2f}%`.

            В дипломной логике важна именно вторая модель: она использует новости и рыночные
            признаки вместе. Модель получает агрегаты новостей в час публикации, признаки
            источников, текстовые/тональные признаки и рыночное состояние акции в этот момент.
            После этого она оценивает будущую доходность около `+24 календарных часов`.
            """
        )

    comparison = test_results.sort_values(["mae_pct", "rmse_pct"]).copy()
    comparison["Набор признаков"] = comparison["feature_set"].map(format_magnitude_feature_set)
    comparison["Модель"] = comparison["model"].map(format_news_24h_model_name)
    fig = px.bar(
        comparison.head(12).sort_values("mae_pct", ascending=True),
        x="mae_pct",
        y="Модель",
        color="Набор признаков",
        orientation="h",
        text="mae_pct",
        labels={"mae_pct": "MAE, п.п.", "Модель": "Модель"},
        title="Сравнение моделей на test по MAE",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(template="plotly_dark", height=430, margin=dict(l=20, r=20, t=70, b=30))
    st.plotly_chart(fig, width="stretch", key="best_calendar_model_comparison_chart")

    chart_candidates = signal_rows if not signal_rows.empty else selected_predictions
    ticker_counts = chart_candidates["security"].value_counts()
    chart_controls = st.columns([1.2, 1, 1])
    with chart_controls[0]:
        selected_ticker = st.selectbox(
            "Тикер для графика",
            ticker_counts.index.tolist(),
            format_func=lambda value: f"{value} ({ticker_counts[value]} строк)",
            key="best_calendar_ticker",
        )
    with chart_controls[1]:
        days_back = st.slider(
            "Показывать дней",
            min_value=30,
            max_value=520,
            value=360,
            step=30,
            key="best_calendar_days_back",
        )
    with chart_controls[2]:
        max_lines = st.slider(
            "Отрезков",
            min_value=1,
            max_value=30,
            value=12,
            step=1,
            key="best_calendar_max_lines",
        )

    st.plotly_chart(
        plot_price_chart_with_return_24h(
            stocks=stocks,
            predictions=selected_predictions,
            ticker=selected_ticker,
            days_back=days_back,
            signal_threshold=signal_threshold,
            max_lines=max_lines,
        ),
        width="stretch",
        key="best_calendar_return_chart",
    )

    st.caption("Test-сценарии выбранной модели")
    st.dataframe(
        make_return_24h_table(signal_rows).head(300),
        width="stretch",
        hide_index=True,
    )


def render_abnormal_return_page(stocks, abnormal_results, abnormal_predictions, future_predictions=None):
    st.subheader("Event study: abnormal return около +24 часов")

    if abnormal_results is None or abnormal_predictions is None or abnormal_predictions.empty:
        st.info(
            "Файлы abnormal-return эксперимента пока не найдены. "
            "Запусти `python updates/scripts/run_abnormal_24h_return_experiment.py --fast --max-train-rows 5000`."
        )
        return

    test_results = abnormal_results[abnormal_results["split"] == "test"].copy()
    test_predictions = abnormal_predictions[abnormal_predictions["split"] == "test"].copy()
    if test_results.empty or test_predictions.empty:
        st.warning("В abnormal-return файлах нет test-части.")
        return

    feature_sets = sorted(test_results["feature_set"].dropna().unique())
    model_names = sorted(test_results["model"].dropna().unique())
    default_feature = (
        feature_sets.index("news_plus_price")
        if "news_plus_price" in feature_sets
        else 0
    )
    default_model = (
        model_names.index("hist_gradient_boosting")
        if "hist_gradient_boosting" in model_names
        else 0
    )

    controls = st.columns([1.2, 1, 1])
    with controls[0]:
        feature_set = st.selectbox(
            "Набор признаков",
            feature_sets,
            index=default_feature,
            format_func=format_magnitude_feature_set,
            key="abnormal_feature_set",
        )
    with controls[1]:
        model_name = st.selectbox(
            "Модель",
            model_names,
            index=default_model,
            format_func=format_news_24h_model_name,
            key="abnormal_model",
        )
    with controls[2]:
        signal_threshold = st.slider(
            "Порог abnormal-сценария",
            min_value=0.001,
            max_value=0.020,
            value=0.001,
            step=0.001,
            format="%.3f",
            key="abnormal_signal_threshold",
        )

    selected_results = test_results[
        (test_results["feature_set"] == feature_set)
        & (test_results["model"] == model_name)
    ].copy()
    selected_predictions = test_predictions[
        (test_predictions["feature_set"] == feature_set)
        & (test_predictions["model"] == model_name)
    ].copy()
    if selected_results.empty or selected_predictions.empty:
        st.warning("Для выбранной abnormal-return модели нет данных.")
        return

    selected_predictions["return_after_24h"] = selected_predictions["calendar_return_24h"]
    selected_predictions["actual_price"] = selected_predictions["calendar_actual_price_24h"]
    if "predicted_raw_return" in selected_predictions.columns:
        if "predicted_abnormal_return" not in selected_predictions.columns:
            selected_predictions["predicted_abnormal_return"] = selected_predictions["predicted_return"]
        selected_predictions["predicted_return"] = selected_predictions["predicted_abnormal_return"]

    abnormal_signal_rows = selected_predictions[
        selected_predictions["predicted_abnormal_return"].abs()
        >= signal_threshold
    ].copy()
    future_rows = get_future_rows_for_model(
        future_predictions,
        feature_set,
        model_name,
        "predicted_abnormal_return",
    )

    row = selected_results.iloc[0]
    metric_cols = st.columns(6)
    metric_cols[0].metric("MAE abnormal", f"{row['mae_pct']:.2f}%")
    metric_cols[1].metric("RMSE abnormal", f"{row['rmse_pct']:.2f}%")
    metric_cols[2].metric("R2 abnormal", f"{row['r2']:.3f}")
    metric_cols[3].metric("Hit-rate abnormal", f"{row['direction_hit_rate']:.1%}")
    if abnormal_signal_rows.empty:
        metric_cols[4].metric("Сценарии", "0")
        metric_cols[5].metric("Future", f"{len(future_rows)}")
    else:
        abnormal_pred_col = (
            "predicted_abnormal_return"
            if "predicted_abnormal_return" in abnormal_signal_rows.columns
            else "predicted_return"
        )
        signal_hit_rate = (
            (abnormal_signal_rows[abnormal_pred_col] >= 0)
            == (abnormal_signal_rows["abnormal_return_24h"] >= 0)
        ).mean()
        metric_cols[4].metric("Сценарии", f"{len(abnormal_signal_rows)}")
        metric_cols[5].metric("Future", f"{len(future_rows)}")

    with st.expander("Что показывает abnormal return"):
        st.markdown(
            """
            **Abnormal return** — это доходность акции за окно около `+24ч` минус доходность
            индекса IMOEX за то же окно. Так мы пытаемся отделить реакцию конкретной акции
            на новость от общего движения рынка.

            На графике линия строится **только по прогнозируемому abnormal-движению**.
            Мы не добавляем фактическую будущую доходность IMOEX, потому что это было бы
            подглядыванием в будущее для визуализации.
            """
        )

    comparison = test_results.sort_values(["mae_pct", "rmse_pct"]).copy()
    comparison["Набор признаков"] = comparison["feature_set"].map(format_magnitude_feature_set)
    comparison["Модель"] = comparison["model"].map(format_news_24h_model_name)
    fig = px.bar(
        comparison.head(12).sort_values("mae_pct", ascending=True),
        x="mae_pct",
        y="Модель",
        color="Набор признаков",
        orientation="h",
        text="mae_pct",
        labels={"mae_pct": "MAE abnormal, п.п.", "Модель": "Модель"},
        title="Сравнение abnormal-return моделей на test",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    fig.update_layout(template="plotly_dark", height=430, margin=dict(l=20, r=20, t=70, b=30))
    st.plotly_chart(fig, width="stretch", key="abnormal_model_comparison_chart")

    if abnormal_signal_rows.empty and future_rows.empty:
        chart_candidates = selected_predictions
    else:
        chart_candidates = pd.concat([abnormal_signal_rows, future_rows], ignore_index=True)
    ticker_counts = chart_candidates["security"].value_counts()
    chart_controls = st.columns([1.2, 1, 1])
    with chart_controls[0]:
        default_ticker = None
        if not future_rows.empty:
            default_ticker = (
                future_rows.sort_values("forecast_strength", ascending=False)
                .iloc[0]["security"]
            )
        ticker_options = ticker_counts.index.tolist()
        default_ticker_index = (
            ticker_options.index(default_ticker)
            if default_ticker in ticker_options
            else 0
        )
        selected_ticker = st.selectbox(
            "Тикер для графика",
            ticker_options,
            index=default_ticker_index,
            format_func=lambda value: f"{value} ({ticker_counts[value]} строк)",
            key="abnormal_ticker",
        )
    with chart_controls[1]:
        days_back = st.slider(
            "Показывать дней",
            min_value=30,
            max_value=520,
            value=360,
            step=30,
            key="abnormal_days_back",
        )
    with chart_controls[2]:
        max_lines = st.slider(
            "Отрезков",
            min_value=1,
            max_value=30,
            value=12,
            step=1,
            key="abnormal_max_lines",
        )

    st.plotly_chart(
        plot_price_chart_with_return_24h(
            stocks=stocks,
            predictions=chart_candidates,
            ticker=selected_ticker,
            days_back=days_back,
            signal_threshold=signal_threshold,
            max_lines=max_lines,
            signal_column="predicted_abnormal_return",
            show_neutral_forecasts=not future_rows.empty,
        ),
        width="stretch",
        key="abnormal_return_chart",
    )

    table_rows = abnormal_signal_rows.copy()
    if "predicted_abnormal_return" in table_rows.columns:
        table_rows["Прогноз abnormal, %"] = (
            table_rows["predicted_abnormal_return"] * 100
        ).round(2)
        table_rows["Факт abnormal, %"] = (table_rows["abnormal_return_24h"] * 100).round(2)
        table_rows["Доходность IMOEX, %"] = (table_rows["market_return_24h"] * 100).round(2)

    st.caption("Abnormal-return сценарии выбранной модели")
    st.dataframe(
        make_return_24h_table(table_rows).head(300),
        width="stretch",
        hide_index=True,
    )


def render_market_news_attribution_page(attribution, driver_summary, ticker_summary):
    st.subheader("Что двигало цену: рынок или новость")

    if attribution is None or attribution.empty:
        st.info(
            "Файлы разложения движения пока не найдены. "
            "Запусти `python updates/scripts/build_market_news_attribution.py`."
        )
        return

    data = attribution.copy()
    data["start_time"] = pd.to_datetime(data["start_time"], errors="coerce")

    controls = st.columns([1, 1, 1])
    tickers = ["Все"] + sorted(data["security"].dropna().unique().tolist())
    with controls[0]:
        ticker = st.selectbox("Тикер", tickers, key="attribution_ticker")
    with controls[1]:
        min_abs_return = st.slider(
            "Мин. движение акции",
            min_value=0.0,
            max_value=0.05,
            value=0.01,
            step=0.005,
            format="%.3f",
            key="attribution_min_abs_return",
        )
    with controls[2]:
        driver_options = ["Все"] + sorted(data["dominant_driver"].dropna().unique().tolist())
        driver = st.selectbox("Тип движения", driver_options, key="attribution_driver")

    filtered = data[data["calendar_return_24h"].abs() >= min_abs_return].copy()
    if ticker != "Все":
        filtered = filtered[filtered["security"] == ticker]
    if driver != "Все":
        filtered = filtered[filtered["dominant_driver"] == driver]

    metric_cols = st.columns(4)
    metric_cols[0].metric("Наблюдений", f"{len(filtered)}")
    if not filtered.empty:
        metric_cols[1].metric("Среднее |движение|", f"{filtered['calendar_return_24h'].abs().mean():.2%}")
        metric_cols[2].metric("Средняя |факторная часть|", f"{filtered['market_component_24h'].abs().mean():.2%}")
        metric_cols[3].metric("Средняя |спец. часть|", f"{filtered['specific_return_24h'].abs().mean():.2%}")
    else:
        metric_cols[1].metric("Среднее |движение|", "н/д")
        metric_cols[2].metric("Средняя |факторная часть|", "н/д")
        metric_cols[3].metric("Средняя |спец. часть|", "н/д")

    with st.expander("Как читать разложение"):
        st.markdown(
            """
            Здесь не делается прогноз. Это **event-study разложение факта**:

            `доходность акции за 24ч = beta к IMOEX * доходность IMOEX + beta к Brent * доходность Brent + beta к Gold * доходность Gold + специфическая часть`.

            Если основная часть движения объясняется общими факторами, событие помечается как `рынок`.
            Если большая часть движения остается в специфической компоненте, событие помечается как
            `тикер/новость`. Brent используется как дневная спотовая цена, Gold — как месячный
            макрофактор, протянутый на дневной календарь.
            """
        )

    if driver_summary is not None and not driver_summary.empty:
        summary = driver_summary.copy()
        summary["Среднее |движение|"] = (summary["avg_abs_return_24h"] * 100).round(2)
        summary["Средняя |спец. часть|"] = (summary["avg_abs_specific_24h"] * 100).round(2)
        fig = px.bar(
            summary,
            x="dominant_driver",
            y="rows",
            color="dominant_driver",
            text="rows",
            labels={"dominant_driver": "Тип движения", "rows": "Количество"},
            title="Сколько событий объясняется рынком, тикером или смешанно",
        )
        fig.update_layout(template="plotly_dark", height=420, showlegend=False)
        st.plotly_chart(fig, width="stretch", key="market_news_driver_bar_chart")

    if not filtered.empty:
        scatter = filtered.copy()
        scatter["Доходность акции, %"] = scatter["calendar_return_24h"] * 100
        scatter["Факторная часть, %"] = scatter["market_component_24h"] * 100
        scatter["Специфическая часть, %"] = scatter["specific_return_24h"] * 100
        fig = px.scatter(
            scatter,
            x="Факторная часть, %",
            y="Специфическая часть, %",
            color="dominant_driver",
            hover_data=["security", "start_time", "news_title_example", "Доходность акции, %"],
            title="Разложение движения: рынок против специфики тикера",
        )
        fig.add_hline(y=0, line_color="rgba(255,255,255,0.35)")
        fig.add_vline(x=0, line_color="rgba(255,255,255,0.35)")
        fig.update_layout(template="plotly_dark", height=560)
        st.plotly_chart(fig, width="stretch", key="market_news_factor_scatter_chart")

    table = filtered.sort_values("start_time", ascending=False).copy()
    table["Доходность акции, %"] = (table["calendar_return_24h"] * 100).round(2)
    table["Доходность IMOEX, %"] = (table["market_return_24h"] * 100).round(2)
    if "brent_return_24h" in table.columns:
        table["Доходность Brent, %"] = (table["brent_return_24h"] * 100).round(2)
    if "gold_return_24h" in table.columns:
        table["Доходность Gold, %"] = (table["gold_return_24h"] * 100).round(2)
    if "imoex_component_24h" in table.columns:
        table["Часть IMOEX, %"] = (table["imoex_component_24h"] * 100).round(2)
    if "brent_component_24h" in table.columns:
        table["Часть Brent, %"] = (table["brent_component_24h"] * 100).round(2)
    if "gold_component_24h" in table.columns:
        table["Часть Gold, %"] = (table["gold_component_24h"] * 100).round(2)
    table["Факторная часть, %"] = (table["market_component_24h"] * 100).round(2)
    table["Специфическая часть, %"] = (table["specific_return_24h"] * 100).round(2)
    table["Beta к IMOEX"] = table["market_beta_imoex"].round(2)
    if "market_beta_brent" in table.columns:
        table["Beta к Brent"] = table["market_beta_brent"].round(2)
    if "market_beta_gold" in table.columns:
        table["Beta к Gold"] = table["market_beta_gold"].round(2)
    columns = [
        "start_time",
        "security",
        "dominant_driver",
        "Доходность акции, %",
        "Доходность IMOEX, %",
        "Доходность Brent, %",
        "Доходность Gold, %",
        "Часть IMOEX, %",
        "Часть Brent, %",
        "Часть Gold, %",
        "Факторная часть, %",
        "Специфическая часть, %",
        "Beta к IMOEX",
        "Beta к Brent",
        "Beta к Gold",
        "news_count",
        "news_title_example",
    ]
    st.caption("Новости и разложение движения")
    st.dataframe(
        table[[col for col in columns if col in table.columns]].rename(columns={
            "start_time": "Дата",
            "security": "Тикер",
            "dominant_driver": "Тип движения",
            "news_count": "Новостей в час",
            "news_title_example": "Пример новости",
        }).head(500),
        width="stretch",
        hide_index=True,
    )


def render_magnitude_page(stocks, magnitude_results, magnitude_predictions, source_analysis):
    st.subheader("Magnitude: оценка силы движения после новости")

    if magnitude_results is None or magnitude_predictions is None or magnitude_predictions.empty:
        st.info(
            "Файлы magnitude-эксперимента пока не найдены. "
            "Запусти `python updates/scripts/run_magnitude_estimation_experiment.py`."
        )
        return

    task_options = sorted(magnitude_predictions["task"].dropna().unique())
    task = st.selectbox(
        "Постановка",
        task_options,
        index=task_options.index("magnitude_classification")
        if "magnitude_classification" in task_options
        else 0,
        format_func=format_magnitude_task,
    )

    task_predictions = magnitude_predictions[magnitude_predictions["task"] == task].copy()
    feature_sets = sorted(task_predictions["feature_set"].dropna().unique())
    model_names = sorted(task_predictions["model"].dropna().unique())

    default_feature = (
        feature_sets.index("price_only_on_news")
        if "price_only_on_news" in feature_sets
        else 0
    )
    default_model = (
        model_names.index("mlp")
        if task == "magnitude_classification" and "mlp" in model_names
        else model_names.index("catboost")
        if "catboost" in model_names
        else 0
    )

    controls = st.columns([1.5, 1.1, 1])
    with controls[0]:
        feature_set = st.selectbox(
            "Набор признаков",
            feature_sets,
            index=default_feature,
            format_func=format_magnitude_feature_set,
        )
    with controls[1]:
        model_name = st.selectbox(
            "Модель",
            model_names,
            index=default_model,
            format_func=format_news_24h_model_name,
        )
    with controls[2]:
        top_share = st.selectbox(
            "Топ сигналов",
            [0.05, 0.10, 0.20],
            index=1,
            format_func=lambda value: f"{int(value * 100)}%",
        )

    selected_results = magnitude_results[
        (magnitude_results["task"] == task)
        & (magnitude_results["feature_set"] == feature_set)
        & (magnitude_results["model"] == model_name)
    ].copy()
    selected_predictions = task_predictions[
        (task_predictions["feature_set"] == feature_set)
        & (task_predictions["model"] == model_name)
    ].copy()

    if selected_predictions.empty:
        st.warning("Для выбранной модели нет prediction-таблицы.")
        return

    selected_predictions["score"] = selected_predictions.apply(get_magnitude_score, axis=1)
    selected_predictions = selected_predictions.sort_values("score", ascending=False)
    top_count = max(1, int(len(selected_predictions) * top_share))
    top_predictions = selected_predictions.head(top_count).copy()

    metric_cols = st.columns(5)
    if not selected_results.empty:
        row = selected_results.iloc[0]
        if task == "magnitude_classification":
            metric_cols[0].metric("Accuracy", f"{row['accuracy']:.3f}")
            metric_cols[1].metric("Balanced Accuracy", f"{row['balanced_accuracy']:.3f}")
            metric_cols[2].metric("F1 macro", f"{row['f1_macro']:.3f}")
        else:
            metric_cols[0].metric("MAE", f"{row['mae']:.3%}")
            metric_cols[1].metric("RMSE", f"{row['rmse']:.3%}")
            metric_cols[2].metric("Spearman", f"{row['spearman']:.3f}")
        metric_cols[3].metric("Strong Precision@5%", f"{row['strong_precision_at_5pct']:.1%}")
        metric_cols[4].metric("Strong Precision@10%", f"{row['strong_precision_at_10pct']:.1%}")

    with st.expander("Как читать magnitude"):
        st.markdown(
            """
            **Magnitude** не обещает точную будущую цену. Это оценка силы реакции после новости.

            В классификации модель относит будущую реакцию к диапазонам: `<1%`, `1-2%`,
            `2-4%` или `>4%`. Для интерфейса особенно важна вероятность `2%+`: она
            показывает, насколько модель считает событие сильным.

            На графике прогнозный отрезок показывает сценарный уровень цены. Например,
            если цена в момент новости равна 100, а magnitude-сценарий равен +4%, линия
            идет к уровню 104 за 24 часа. Если раньше появляется новый сигнал по этому
            же тикеру, старый отрезок обрезается и дальше строится новый сценарий.

            **Strong Precision@10%** означает, какая доля реальных движений `2%+`
            оказалась среди самых сильных 10% magnitude-сигналов.
            """
        )

    latest = selected_predictions.sort_values("start_time", ascending=False).iloc[0]
    card_cols = st.columns([1, 1.3, 1.7])
    with card_cols[0]:
        st.metric("Последняя оценка", latest["security"])
        if pd.notna(latest.get("proba_strong_ge_2pct")):
            st.metric("Вероятность 2%+", f"{latest['proba_strong_ge_2pct']:.1%}")
        if pd.notna(latest.get("predicted_magnitude")):
            st.metric("Прогноз движения", f"{latest['predicted_magnitude']:.2%}")
        if pd.notna(latest.get("predicted_label")):
            st.metric("Прогноз диапазон", str(latest["predicted_label"]))
        st.caption(f"Дата: {latest['start_time'].strftime('%Y-%m-%d %H:%M')}")
    with card_cols[1]:
        if task == "magnitude_classification":
            st.plotly_chart(
                plot_magnitude_probabilities(latest),
                width="stretch",
                key="magnitude_probability_chart",
            )
        else:
            st.markdown("**Интерпретация регрессии**")
            st.info(
                "Регрессионная модель оценивает ожидаемый модуль движения. "
                "Для защиты ее лучше использовать как ранжирование силы реакции, "
                "а не как точный прогноз будущей доходности."
            )
    with card_cols[2]:
        if source_analysis is not None and not source_analysis.empty:
            source_plot = source_analysis.copy()
            source_plot["Источник"] = (
                source_plot["source_feature"]
                .str.replace("source_count_", "", regex=False)
                .str.replace("_", ".")
            )
            fig = px.bar(
                source_plot.sort_values("share_ge_2pct"),
                x="share_ge_2pct",
                y="Источник",
                orientation="h",
                text="news_events",
                labels={
                    "share_ge_2pct": "Доля событий 2%+",
                    "Источник": "Источник",
                },
                title="Качество источников по сильным движениям",
            )
            fig.update_layout(template="plotly_dark", height=330, margin=dict(l=20, r=20, t=70, b=30))
            fig.update_xaxes(tickformat=".0%")
            st.plotly_chart(fig, width="stretch", key="magnitude_source_quality_chart")

    ticker_counts = top_predictions["security"].value_counts()
    chart_controls = st.columns([1.2, 1, 1, 1])
    with chart_controls[0]:
        selected_ticker = st.selectbox(
            "Тикер для графика",
            ticker_counts.index.tolist(),
            format_func=lambda value: f"{value} ({ticker_counts[value]} top-сигналов)",
            key="magnitude_ticker",
        )
    with chart_controls[1]:
        days_back = st.slider(
            "Показывать дней",
            min_value=30,
            max_value=520,
            value=360,
            step=30,
            key="magnitude_days_back",
        )
    with chart_controls[2]:
        scenario_direction = st.selectbox(
            "Направление сценария",
            ["growth", "fall", "actual"],
            index=2,
            format_func=lambda value: {
                "growth": "Рост",
                "fall": "Снижение",
                "actual": "По факту теста",
            }[value],
            key="magnitude_scenario_direction",
        )
    with chart_controls[3]:
        max_scenario_lines = st.slider(
            "Отрезков",
            min_value=1,
            max_value=25,
            value=8,
            step=1,
            key="magnitude_max_scenario_lines",
        )

    st.plotly_chart(
        plot_price_chart_with_magnitude(
            stocks=stocks,
            predictions=top_predictions,
            ticker=selected_ticker,
            days_back=days_back,
            scenario_direction=scenario_direction,
            max_scenario_lines=max_scenario_lines,
        ),
        width="stretch",
        key="magnitude_price_scenarios_chart",
    )

    st.caption("Топ magnitude-сигналов выбранной модели")
    st.dataframe(
        make_magnitude_table(top_predictions).head(200),
        width="stretch",
        hide_index=True,
    )


def render_models_page(
    comparison_results,
    comparison_signals,
    direction_memory_results=None,
    old_direction_results=None,
):
    st.subheader("Сравнение моделей 24ч")

    if comparison_results is None or comparison_results.empty:
        st.info("Файл сравнения 24ч-моделей не найден.")
        return

    event_results = comparison_results[comparison_results["task"] == "event_24h"].copy()
    direction_results = comparison_results[
        comparison_results["task"] == "direction_inside_true_events"
    ].copy()

    threshold_options = sorted(event_results["threshold"].dropna().unique())
    selected_threshold = st.selectbox(
        "Порог движения для сравнения",
        threshold_options,
        format_func=lambda value: f"{value:.1%}",
    )

    event_table = event_results[event_results["threshold"] == selected_threshold].copy()
    event_table["Набор признаков"] = event_table["feature_set"].map(format_news_24h_feature_set)
    event_table["Модель"] = event_table["model"].map(format_news_24h_model_name)
    event_table = event_table.sort_values(
        ["precision_at_5pct", "pr_auc", "roc_auc"],
        ascending=False,
    )

    direction_tables_for_chart = []
    direction_table = pd.DataFrame()
    direction_memory_table = pd.DataFrame()
    old_direction_table = pd.DataFrame()

    if not direction_results.empty:
        direction_table = direction_results[
            direction_results["threshold"] == selected_threshold
        ].copy()
        direction_table["Набор признаков"] = direction_table["feature_set"].map(
            format_news_24h_feature_set
        )
        direction_table["Модель"] = direction_table["model"].map(format_news_24h_model_name)
        direction_table["Постановка"] = "24ч: направление внутри event"
        direction_table["Горизонт"] = "24ч"
        direction_table["Порог события"] = direction_table["threshold"].map(
            lambda value: f"{value:.1%}"
        )
        direction_table = direction_table.sort_values(
            ["roc_auc", "pr_auc", "precision_at_5pct"],
            ascending=False,
        )
        direction_tables_for_chart.append(direction_table)

    if direction_memory_results is not None and not direction_memory_results.empty:
        direction_memory_table = direction_memory_results.copy()
        direction_memory_table["Набор признаков"] = direction_memory_table["feature_set"].map(
            format_direction_feature_set
        )
        direction_memory_table["Модель"] = direction_memory_table["model"].map(
            format_news_24h_model_name
        )
        direction_memory_table["Постановка"] = "24ч: first-hit direction"
        direction_memory_table["Горизонт"] = "24ч"
        direction_memory_table["Порог события"] = "2.0%"
        direction_memory_table = direction_memory_table.sort_values(
            ["roc_auc", "precision_at_5pct"],
            ascending=False,
        )
        direction_tables_for_chart.append(direction_memory_table)

    if old_direction_results is not None and not old_direction_results.empty:
        old_direction_table = old_direction_results.copy()
        old_direction_table["Набор признаков"] = old_direction_table["feature_set"]
        old_direction_table["Модель"] = old_direction_table["model"].map(
            format_news_24h_model_name
        )
        old_direction_table["Постановка"] = old_direction_table["source"]
        old_direction_table["Горизонт"] = old_direction_table["horizon"].map(
            lambda value: f"{int(value)}ч" if pd.notna(value) else "н/д"
        )
        old_direction_table["Порог события"] = old_direction_table["event_threshold"].map(
            lambda value: f"{value:.1%}" if pd.notna(value) else "н/д"
        )
        old_direction_table = old_direction_table.sort_values(
            ["roc_auc", "precision_at_5pct"],
            ascending=False,
        )
        direction_tables_for_chart.append(old_direction_table)

    if direction_tables_for_chart:
        direction_chart_table = pd.concat(direction_tables_for_chart, ignore_index=True)
    else:
        direction_chart_table = pd.DataFrame()

    st.plotly_chart(
        plot_event_model_comparison(event_table),
        width="stretch",
        key="models_event_comparison_chart",
    )
    st.plotly_chart(
        plot_direction_model_comparison(direction_chart_table),
        width="stretch",
        key="models_direction_comparison_chart",
    )

    st.caption(
        "Важно: старые direction-метрики не сравниваются один-в-один с новой 24ч-постановкой. "
        "Там другой горизонт, другой порог события и местами другой способ формирования target."
    )

    st.markdown("**Таблицы метрик**")

    st.markdown("**Event-модель: будет ли сильное движение за 24 часа**")
    st.dataframe(
        event_table[
            [
                "Набор признаков",
                "Модель",
                "positive_rate",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "precision_at_5pct",
                "precision_at_10pct",
            ]
        ].rename(columns={
            "positive_rate": "Доля событий",
            "precision": "Precision",
            "recall": "Recall",
            "f1": "F1",
            "roc_auc": "ROC-AUC",
            "pr_auc": "PR-AUC",
            "precision_at_5pct": "Precision@5%",
            "precision_at_10pct": "Precision@10%",
        }),
        width="stretch",
        hide_index=True,
    )

    if not direction_table.empty:
        st.markdown("**Direction-модель: направление внутри реальных событий**")
        st.dataframe(
            direction_table[
                [
                    "Набор признаков",
                    "Модель",
                    "positive_rate",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "pr_auc",
                    "precision_at_5pct",
                ]
            ].rename(columns={
                "positive_rate": "Доля роста",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC",
                "precision_at_5pct": "Precision@5%",
            }),
            width="stretch",
            hide_index=True,
        )

    if comparison_signals is not None and not comparison_signals.empty:
        signal_table = comparison_signals[
            comparison_signals["threshold"] == selected_threshold
        ].copy()
        signal_table["Набор признаков"] = signal_table["feature_set"].map(
            format_news_24h_feature_set
        )
        signal_table["Модель"] = signal_table["model"].map(format_news_24h_model_name)
        signal_table = signal_table.sort_values(
            ["true_event_rate", "signals"],
            ascending=False,
        )

        st.markdown("**Практическая проверка сигналов при разных порогах**")
        st.dataframe(
            signal_table[
                [
                    "Набор признаков",
                    "Модель",
                    "event_cutoff",
                    "signals",
                    "true_event_rate",
                    "hit_rate",
                    "avg_event_proba",
                    "avg_direction_confidence",
                    "avg_best_side_return",
                ]
            ].rename(columns={
                "event_cutoff": "Порог сигнала",
                "signals": "Сигналов",
                "true_event_rate": "Доля реальных событий",
                "hit_rate": "Угадано направление",
                "avg_event_proba": "Средняя вероятность события",
                "avg_direction_confidence": "Средняя уверенность направления",
                "avg_best_side_return": "Среднее движение по стороне",
            }),
            width="stretch",
            hide_index=True,
        )

    st.caption(
        "Эти таблицы выше показывают глобальные метрики модели. "
        "Порог уверенности direction влияет на практические метрики выбранных сигналов на первой и второй вкладке."
    )

    if not direction_memory_table.empty:
        st.markdown("**Улучшенные direction-модели: новости и память тикера**")
        st.dataframe(
            direction_memory_table[
                [
                    "Набор признаков",
                    "Модель",
                    "positive_rate",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "pr_auc",
                    "precision_at_5pct",
                    "precision_at_10pct",
                ]
            ].rename(columns={
                "positive_rate": "Доля роста",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC",
                "precision_at_5pct": "Precision@5%",
                "precision_at_10pct": "Precision@10%",
            }),
            width="stretch",
            hide_index=True,
        )

    if not old_direction_table.empty:
        st.markdown("**Старые direction-модели из архива**")
        st.dataframe(
            old_direction_table[
                [
                    "Постановка",
                    "Горизонт",
                    "Порог события",
                    "Набор признаков",
                    "Модель",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "pr_auc",
                    "precision_at_5pct",
                    "precision_at_10pct",
                ]
            ].rename(columns={
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "roc_auc": "ROC-AUC",
                "pr_auc": "PR-AUC",
                "precision_at_5pct": "Precision@5%",
                "precision_at_10pct": "Precision@10%",
            }).head(30),
            width="stretch",
            hide_index=True,
        )


st.set_page_config(
    page_title="MOEX News 24h Signals",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Новостные сигналы MOEX на горизонте 24 часа")
st.caption("Модель ищет новости, после которых цена может показать сильное движение в течение суток")

missing_data_files = get_missing_main_data_files()
if missing_data_files:
    st.warning("Не найдены основные файлы данных. Их можно загрузить из интерфейса.")
    st.write("Отсутствуют:")
    for file_path, description in missing_data_files:
        st.write(f"- `{file_path}` — {description}")

    with st.sidebar:
        st.header("Первичная загрузка")
        update_market = st.checkbox("Загрузить котировки MOEX", value=True)
        update_news = st.checkbox("Загрузить новости и признаки", value=True)
        market_date_from = st.date_input(
            "Котировки с даты",
            value=datetime(2020, 1, 1).date(),
        )
        news_date_from = st.date_input(
            "Новости с даты",
            value=datetime(2020, 1, 1).date(),
        )

        if st.button("Загрузить данные"):
            with st.spinner("Загружаю данные. Первый запуск может занять много времени..."):
                update_project_data(
                    update_market,
                    update_news,
                    market_date_from=str(market_date_from),
                    news_date_from=str(news_date_from),
                    use_heavy_news_features=False,
                )
            st.cache_data.clear()
            st.rerun()

    st.info("После загрузки котировок и новостных признаков приложение откроет основной интерфейс.")
    st.stop()

stocks = load_stocks(STOCKS_PATH, get_file_mtime(STOCKS_PATH))
news = load_news(NEWS_PATH, get_file_mtime(NEWS_PATH))

with st.sidebar:
    st.header("Покрытие данных")
    st.write(f"**Котировки:** {get_date_range_label(stocks, 'datetime')}")
    st.write(f"**Новости:** {get_date_range_label(news, 'published_at')}")

    st.header("Обновление данных")
    update_market = st.checkbox("Обновить котировки MOEX", value=True)
    update_news = st.checkbox("Обновить новости и признаки", value=True)
    use_heavy_news_features = st.checkbox(
        "Считать RuBERT/sentiment признаки",
        value=False,
        help=(
            "Для текущей 24ч-модели это не требуется. "
            "Включай только если нужен полный старый набор текстовых признаков."
        ),
    )

    if st.button("Обновить данные"):
        with st.spinner("Обновляю данные..."):
            update_project_data(
                update_market,
                update_news,
                use_heavy_news_features=use_heavy_news_features,
            )
            st.cache_data.clear()
            st.rerun()

    if st.button("Переобучить 24ч-модель"):
        with st.spinner("Переобучаю 24ч-модель..."):
            try:
                completed = retrain_news_24h_model()
            except subprocess.TimeoutExpired:
                st.error("Обучение остановлено по таймауту 30 минут.")
            except Exception as error:
                st.error(f"Не удалось запустить обучение: {error}")
            else:
                if completed.returncode == 0:
                    st.success("24ч-модель переобучена.")
                    st.cache_data.clear()
                    st.rerun()
                else:
                    st.error("Обучение завершилось с ошибкой.")
                    st.code(completed.stderr[-4000:] or completed.stdout[-4000:])

try:
    news_24h_results = load_results(
        NEWS_24H_RESULTS_PATH,
        get_file_mtime(NEWS_24H_RESULTS_PATH),
    )
    news_24h_signal_slices = load_results(
        NEWS_24H_SIGNALS_PATH,
        get_file_mtime(NEWS_24H_SIGNALS_PATH),
    )
    news_24h_predictions = load_news_24h_predictions(
        NEWS_24H_PREDICTIONS_PATH,
        get_file_mtime(NEWS_24H_PREDICTIONS_PATH),
    )
except Exception:
    news_24h_results = None
    news_24h_signal_slices = None
    news_24h_predictions = None

try:
    comparison_results = load_results(
        NEWS_24H_COMPARISON_RESULTS_PATH,
        get_file_mtime(NEWS_24H_COMPARISON_RESULTS_PATH),
    )
    comparison_signal_slices = load_results(
        NEWS_24H_COMPARISON_SIGNALS_PATH,
        get_file_mtime(NEWS_24H_COMPARISON_SIGNALS_PATH),
    )
    comparison_predictions = load_news_24h_predictions(
        NEWS_24H_COMPARISON_PREDICTIONS_PATH,
        get_file_mtime(NEWS_24H_COMPARISON_PREDICTIONS_PATH),
    )
except Exception:
    comparison_results = news_24h_results
    comparison_signal_slices = news_24h_signal_slices
    comparison_predictions = news_24h_predictions

try:
    direction_memory_results = load_results(
        NEWS_MEMORY_DIRECTION_RESULTS_PATH,
        get_file_mtime(NEWS_MEMORY_DIRECTION_RESULTS_PATH),
    )
    direction_memory_predictions = load_news_24h_predictions(
        NEWS_MEMORY_DIRECTION_PREDICTIONS_PATH,
        get_file_mtime(NEWS_MEMORY_DIRECTION_PREDICTIONS_PATH),
    )
    direction_memory_slices = load_results(
        NEWS_MEMORY_DIRECTION_SIGNALS_PATH,
        get_file_mtime(NEWS_MEMORY_DIRECTION_SIGNALS_PATH),
    )
except Exception:
    direction_memory_results = None
    direction_memory_predictions = None
    direction_memory_slices = None

try:
    magnitude_results = load_results(
        MAGNITUDE_RESULTS_PATH,
        get_file_mtime(MAGNITUDE_RESULTS_PATH),
    )
    magnitude_predictions = load_magnitude_predictions(
        MAGNITUDE_PREDICTIONS_PATH,
        get_file_mtime(MAGNITUDE_PREDICTIONS_PATH),
    )
    magnitude_source_analysis = load_results(
        MAGNITUDE_SOURCE_ANALYSIS_PATH,
        get_file_mtime(MAGNITUDE_SOURCE_ANALYSIS_PATH),
    )
except Exception:
    magnitude_results = None
    magnitude_predictions = None
    magnitude_source_analysis = None

try:
    return_24h_results = load_results(
        RETURN_24H_RESULTS_PATH,
        get_file_mtime(RETURN_24H_RESULTS_PATH),
    )
    return_24h_predictions = load_news_24h_predictions(
        RETURN_24H_PREDICTIONS_PATH,
        get_file_mtime(RETURN_24H_PREDICTIONS_PATH),
    )
except Exception:
    return_24h_results = None
    return_24h_predictions = None

try:
    calendar_return_24h_results = load_results(
        CALENDAR_RETURN_24H_RESULTS_PATH,
        get_file_mtime(CALENDAR_RETURN_24H_RESULTS_PATH),
    )
    calendar_return_24h_predictions = load_news_24h_predictions(
        CALENDAR_RETURN_24H_PREDICTIONS_PATH,
        get_file_mtime(CALENDAR_RETURN_24H_PREDICTIONS_PATH),
    )
except Exception:
    calendar_return_24h_results = None
    calendar_return_24h_predictions = None

try:
    abnormal_return_24h_results = load_results(
        ABNORMAL_RETURN_24H_RESULTS_PATH,
        get_file_mtime(ABNORMAL_RETURN_24H_RESULTS_PATH),
    )
    abnormal_return_24h_predictions = load_news_24h_predictions(
        ABNORMAL_RETURN_24H_PREDICTIONS_PATH,
        get_file_mtime(ABNORMAL_RETURN_24H_PREDICTIONS_PATH),
    )
except Exception:
    abnormal_return_24h_results = None
    abnormal_return_24h_predictions = None

try:
    future_return_24h_predictions = load_news_24h_predictions(
        FUTURE_RETURN_24H_PREDICTIONS_PATH,
        get_file_mtime(FUTURE_RETURN_24H_PREDICTIONS_PATH),
    )
except Exception:
    future_return_24h_predictions = None

try:
    future_abnormal_24h_predictions = load_news_24h_predictions(
        FUTURE_ABNORMAL_24H_PREDICTIONS_PATH,
        get_file_mtime(FUTURE_ABNORMAL_24H_PREDICTIONS_PATH),
    )
except Exception:
    future_abnormal_24h_predictions = None

try:
    market_news_attribution = load_news_24h_predictions(
        MARKET_NEWS_ATTRIBUTION_PATH,
        get_file_mtime(MARKET_NEWS_ATTRIBUTION_PATH),
    )
    market_news_driver_summary = load_results(
        MARKET_NEWS_DRIVER_SUMMARY_PATH,
        get_file_mtime(MARKET_NEWS_DRIVER_SUMMARY_PATH),
    )
    market_news_ticker_summary = load_results(
        MARKET_NEWS_TICKER_SUMMARY_PATH,
        get_file_mtime(MARKET_NEWS_TICKER_SUMMARY_PATH),
    )
except Exception:
    market_news_attribution = None
    market_news_driver_summary = None
    market_news_ticker_summary = None

old_direction_file_mtimes = tuple(
    get_file_mtime(file_path)
    for file_path, _source_name in OLD_DIRECTION_RESULT_FILES
)
old_direction_results = load_old_direction_results(old_direction_file_mtimes)

tab_events_24h, tab_return_24h = st.tabs(["События 24ч", "Доходность 24ч"])

with tab_events_24h:
    render_news_24h_page(
        stocks=stocks,
        results=comparison_results,
        signal_slices=comparison_signal_slices,
        predictions=comparison_predictions,
        direction_results=direction_memory_results,
        direction_predictions=direction_memory_predictions,
    )

with tab_return_24h:
    render_return_24h_page(
        stocks=stocks,
        return_results=return_24h_results,
        return_predictions=return_24h_predictions,
        future_predictions=future_return_24h_predictions,
    )
