import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.predictions import EVENT_DISPLAY_THRESHOLD, get_prediction_color


def plot_prediction_card(ticker, predictions):
    row = predictions[predictions["security"] == ticker].iloc[0]

    label = row["prediction"]
    color = get_prediction_color(label)

    fig = go.Figure()

    fig.add_trace(go.Indicator(
        mode="number+gauge",
        value=row["event_proba"] * 100,
        title={
            "text": (
                f"<b>{ticker}</b><br>"
                f"Прогноз: <span style='color:{color}'>{label}</span><br>"
                f"Вероятность значимого движения"
            )
        },
        number={
            "suffix": "%",
            "font": {"size": 48, "color": color}
        },
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color}
        }
    ))

    fig.update_layout(
        height=300,
        template="plotly_white",
        margin=dict(l=30, r=30, t=80, b=30)
    )

    return fig


def plot_price_chart(stocks, ticker, last_n=200):
    df = stocks[stocks["security"] == ticker].copy()
    df = df.sort_values("datetime").tail(last_n)

    fig = go.Figure()

    fig.add_trace(go.Candlestick(
        x=df["datetime"],
        open=df["open"],
        high=df["high"],
        low=df["low"],
        close=df["close"],
        name="Котировки"
    ))

    fig.update_layout(
        title=f"Котировки {ticker}",
        xaxis_title="Дата",
        yaxis_title="Цена",
        height=500,
        template="plotly_white",
        xaxis_rangeslider_visible=False
    )

    return fig


def plot_prediction_heatmap(predictions, top_n=15):
    df = predictions.copy()
    df = df[df["event_proba"] >= EVENT_DISPLAY_THRESHOLD].copy()

    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            height=360,
            title="Нет тикеров с вероятностью события выше 60%",
        )
        fig.add_annotation(
            text="Direction не рассчитывается, пока event-модель не дала сильный сигнал.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
        return fig

    df["event_percent"] = df["event_proba"] * 100

    df = df.sort_values("event_percent", ascending=False).head(top_n)

    df["direction_percent"] = df["direction_proba"] * 100
    heatmap_df = df[
        ["security", "event_percent", "direction_percent"]
    ].set_index("security")

    fig = px.imshow(
        heatmap_df,
        text_auto=".1f",
        aspect="auto",
        color_continuous_scale="RdYlGn",
        zmin=0,
        zmax=100,
        title=f"Тикеры с вероятностью события выше 60%"
    )

    fig.update_layout(
        template="plotly_white",
        height=520,
        xaxis_title="Показатель",
        yaxis_title="Тикер",
        coloraxis_colorbar_title="Вероятность, %"
    )

    fig.update_xaxes(
        ticktext=["Событие", "Рост"],
        tickvals=["event_percent", "direction_percent"]
    )

    return fig


def get_top_signals_table(predictions, top_n=15):
    df = predictions.copy()
    df = df[df["event_proba"] >= EVENT_DISPLAY_THRESHOLD].copy()

    df["event_percent"] = (df["event_proba"] * 100).round(1)
    df["direction_percent"] = (df["direction_proba"] * 100).round(1)

    df = df.sort_values("event_percent", ascending=False).head(top_n)

    df = df[
        [
            "security",
            "prediction",
            "event_percent",
            "direction_percent",
            "close"
        ]
    ].rename(columns={
        "security": "Тикер",
        "prediction": "Прогноз",
        "event_percent": "Вероятность события, %",
        "direction_percent": "Вероятность роста, %",
        "close": "Последняя цена"
    })

    return df


def get_full_signals_table(predictions):
    df = predictions.copy()

    df["event_percent"] = (df["event_proba"] * 100).round(1)
    df["direction_percent"] = (df["direction_proba"] * 100).round(1)
    df["direction_status"] = df["direction_available"].map({
        True: "считается",
        False: "не считается",
    })

    if "direction_model" not in df.columns:
        df["direction_model"] = ""

    df = df.sort_values("event_percent", ascending=False)

    df = df[
        [
            "security",
            "prediction",
            "event_percent",
            "direction_status",
            "direction_percent",
            "direction_model",
            "close",
        ]
    ].rename(columns={
        "security": "Тикер",
        "prediction": "Прогноз",
        "event_percent": "Вероятность события, %",
        "direction_status": "Direction",
        "direction_percent": "Вероятность роста, %",
        "direction_model": "Модель direction",
        "close": "Последняя цена",
    })

    return df


def plot_experiment_metrics(results, task=None, title=None):
    df = results.copy()
    if task is not None and "task" in df.columns:
        df = df[df["task"] == task].copy()

    if df.empty:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_white",
            height=520,
            title=title or "Нет данных для выбранной задачи",
        )
        return fig

    hover_columns = [
        "experiment",
        "task",
        "feature_set",
        "model",
        "precision",
        "recall",
        "balanced_accuracy",
        "pr_auc",
        "precision_at_1pct",
        "lift_at_1pct",
        "precision_at_5pct",
        "lift_at_5pct",
        "decision_threshold",
        "threshold_tuning_score_val",
        "n_features",
        "n_train",
        "n_val",
        "n_test",
        "train_positive_share",
        "val_positive_share",
        "test_positive_share",
        "horizon",
        "event_threshold",
        "volatility_multiplier",
        "target_mode",
        "split_name",
        "train_period",
        "val_period",
        "test_period",
    ]
    hover_data = {
        column: ":.4f" if pd.api.types.is_numeric_dtype(df[column]) else True
        for column in hover_columns
        if column in df.columns
    }

    fig = px.scatter(
        df,
        x="f1",
        y="roc_auc",
        color="model" if "model" in df.columns else None,
        symbol="feature_set" if "feature_set" in df.columns else None,
        hover_name="experiment" if "experiment" in df.columns else None,
        hover_data=hover_data,
        title=title or "ROC-AUC и F1 по экспериментам",
        labels={
            "f1": "F1",
            "roc_auc": "ROC-AUC",
            "model": "Модель",
            "feature_set": "Набор признаков",
        },
    )

    fig.update_traces(marker=dict(size=12, opacity=0.82, line=dict(width=0.7)))
    fig.update_layout(
        template="plotly_white",
        height=560,
        xaxis_title="F1",
        yaxis_title="ROC-AUC",
        legend_title_text="Модель / признаки",
    )
    fig.update_xaxes(range=[0, 1], gridcolor="rgba(128,128,128,0.2)")
    fig.update_yaxes(range=[0, 1], gridcolor="rgba(128,128,128,0.2)")

    return fig

def plot_price_chart_with_model_signals(
    stocks,
    signals,
    ticker,
    last_n=300,
    event_threshold=0.12,
    direction_confidence_min=0.0,
):
    import plotly.graph_objects as go

    price_df = stocks[stocks["security"] == ticker].copy()
    price_df = price_df.sort_values("datetime").tail(last_n)

    if price_df.empty:
        return go.Figure()

    price_df["datetime"] = pd.to_datetime(price_df["datetime"])

    signals_df = signals[signals["security"] == ticker].copy()

    if not signals_df.empty:
        signals_df["datetime"] = pd.to_datetime(signals_df["datetime"])

        date_min = price_df["datetime"].min()
        date_max = price_df["datetime"].max()

        signals_df = signals_df[
            (signals_df["datetime"] >= date_min)
            & (signals_df["datetime"] <= date_max)
        ].copy()

        if "event_proba" in signals_df.columns:
            signals_df = signals_df[
                signals_df["event_proba"] >= event_threshold
            ].copy()

        if "direction_proba" in signals_df.columns and direction_confidence_min > 0:
            signals_df = signals_df[
                (signals_df["direction_proba"] - 0.5).abs() >= direction_confidence_min
            ].copy()

        if "signal_direction_available" in signals_df.columns:
            signals_df = signals_df[
                signals_df["signal_direction_available"].fillna(False).astype(bool)
            ].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=price_df["datetime"],
            open=price_df["open"],
            high=price_df["high"],
            low=price_df["low"],
            close=price_df["close"],
            name="Котировки",
        )
    )

    if not signals_df.empty:
        up_signals = signals_df[
            signals_df.get("direction_pred", 1) == 1
        ].copy()

        down_signals = signals_df[
            signals_df.get("direction_pred", 1) == 0
        ].copy()

        if not up_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=up_signals["datetime"],
                    y=up_signals["close"],
                    mode="markers",
                    name="Сигнал: рост",
                    marker=dict(
                        size=11,
                        color="green",
                        symbol="triangle-up",
                        line=dict(width=1, color="black"),
                    ),
                    customdata=up_signals[
                        [
                            "event_proba",
                            "direction_proba",
                            "future_return",
                        ]
                    ].round(4),
                    hovertemplate=(
                        "Дата: %{x}<br>"
                        "Цена: %{y}<br>"
                        "P(event): %{customdata[0]}<br>"
                        "P(up): %{customdata[1]}<br>"
                        "Future return: %{customdata[2]}<extra></extra>"
                    ),
                )
            )

        if not down_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=down_signals["datetime"],
                    y=down_signals["close"],
                    mode="markers",
                    name="Сигнал: снижение",
                    marker=dict(
                        size=11,
                        color="red",
                        symbol="triangle-down",
                        line=dict(width=1, color="black"),
                    ),
                    customdata=down_signals[
                        [
                            "event_proba",
                            "direction_proba",
                            "future_return",
                        ]
                    ].round(4),
                    hovertemplate=(
                        "Дата: %{x}<br>"
                        "Цена: %{y}<br>"
                        "P(event): %{customdata[0]}<br>"
                        "P(up): %{customdata[1]}<br>"
                        "Future return: %{customdata[2]}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=f"Котировки {ticker} и сигналы модели",
        xaxis_title="Дата",
        yaxis_title="Цена",
        height=560,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def plot_price_chart_with_news_24h_signals(
    stocks,
    signals,
    ticker,
    last_n=300,
    days_back=None,
    direction_confidence_min=0.0,
):
    price_df = stocks[stocks["security"] == ticker].copy()
    price_df = price_df.sort_values("datetime")

    if price_df.empty:
        return go.Figure()

    price_df["datetime"] = pd.to_datetime(price_df["datetime"])
    if days_back is not None:
        date_min = price_df["datetime"].max() - pd.Timedelta(days=days_back)
        price_df = price_df[price_df["datetime"] >= date_min].copy()
    else:
        price_df = price_df.tail(last_n)

    signals_df = signals[signals["security"] == ticker].copy()
    if not signals_df.empty:
        signals_df["start_time"] = pd.to_datetime(signals_df["start_time"], errors="coerce")
        date_min = price_df["datetime"].min()
        date_max = price_df["datetime"].max()
        signals_df = signals_df[
            (signals_df["start_time"] >= date_min)
            & (signals_df["start_time"] <= date_max)
        ].copy()

    fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=price_df["datetime"],
            open=price_df["open"],
            high=price_df["high"],
            low=price_df["low"],
            close=price_df["close"],
            name="Котировки",
        )
    )

    if not signals_df.empty:
        confident = signals_df["direction_confidence"] >= direction_confidence_min
        up_signals = signals_df[
            confident & (signals_df["direction_up_proba"] >= 0.5)
        ].copy()
        down_signals = signals_df[
            confident & (signals_df["direction_up_proba"] < 0.5)
        ].copy()
        neutral_signals = signals_df[~confident].copy()

        if not neutral_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=neutral_signals["start_time"],
                    y=neutral_signals["close"],
                    mode="markers",
                    name="Событие без направления",
                    marker=dict(
                        size=10,
                        color="#aeb7bd",
                        symbol="circle",
                        line=dict(width=1, color="black"),
                    ),
                    customdata=neutral_signals[
                        [
                            "event_proba",
                            "direction_up_proba",
                            "direction_confidence",
                            "future_max_return_24h",
                            "future_min_return_24h",
                            "news_title_example",
                        ]
                    ],
                    hovertemplate=(
                        "Дата: %{x}<br>"
                        "Цена: %{y}<br>"
                        "P(event): %{customdata[0]:.3f}<br>"
                        "P(up): %{customdata[1]:.3f}<br>"
                        "Direction confidence: %{customdata[2]:.3f}<br>"
                        "Max 24h: %{customdata[3]:.2%}<br>"
                        "Min 24h: %{customdata[4]:.2%}<br>"
                        "%{customdata[5]}<extra></extra>"
                    ),
                )
            )

        if not up_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=up_signals["start_time"],
                    y=up_signals["close"],
                    mode="markers",
                    name="Сигнал: рост",
                    marker=dict(
                        size=12,
                        color="green",
                        symbol="triangle-up",
                        line=dict(width=1, color="black"),
                    ),
                    customdata=up_signals[
                        [
                            "event_proba",
                            "direction_up_proba",
                            "direction_confidence",
                            "future_max_return_24h",
                            "future_min_return_24h",
                            "news_title_example",
                        ]
                    ],
                    hovertemplate=(
                        "Дата: %{x}<br>"
                        "Цена: %{y}<br>"
                        "P(event): %{customdata[0]:.3f}<br>"
                        "P(up): %{customdata[1]:.3f}<br>"
                        "Direction confidence: %{customdata[2]:.3f}<br>"
                        "Max 24h: %{customdata[3]:.2%}<br>"
                        "Min 24h: %{customdata[4]:.2%}<br>"
                        "%{customdata[5]}<extra></extra>"
                    ),
                )
            )

        if not down_signals.empty:
            fig.add_trace(
                go.Scatter(
                    x=down_signals["start_time"],
                    y=down_signals["close"],
                    mode="markers",
                    name="Сигнал: снижение",
                    marker=dict(
                        size=12,
                        color="red",
                        symbol="triangle-down",
                        line=dict(width=1, color="black"),
                    ),
                    customdata=down_signals[
                        [
                            "event_proba",
                            "direction_up_proba",
                            "direction_confidence",
                            "future_max_return_24h",
                            "future_min_return_24h",
                            "news_title_example",
                        ]
                    ],
                    hovertemplate=(
                        "Дата: %{x}<br>"
                        "Цена: %{y}<br>"
                        "P(event): %{customdata[0]:.3f}<br>"
                        "P(up): %{customdata[1]:.3f}<br>"
                        "Direction confidence: %{customdata[2]:.3f}<br>"
                        "Max 24h: %{customdata[3]:.2%}<br>"
                        "Min 24h: %{customdata[4]:.2%}<br>"
                        "%{customdata[5]}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=f"Новостные 24ч сигналы: {ticker}",
        xaxis_title="Дата",
        yaxis_title="Цена",
        height=560,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig


def get_news_24h_signals_table(signals, direction_confidence_min=0.0):
    df = signals.copy()
    if df.empty:
        return df

    df["event_percent"] = (df["event_proba"] * 100).round(1)
    df["direction_percent"] = (df["direction_up_proba"] * 100).round(1)
    df["direction_confidence_percent"] = (df["direction_confidence"] * 100).round(1)
    df["max_growth_percent"] = (df["future_max_return_24h"] * 100).round(2)
    df["max_fall_percent"] = (df["future_min_return_24h"] * 100).round(2)
    df["max_move_percent"] = (df["future_abs_return_24h"] * 100).round(2)
    df["prediction"] = df.apply(
        lambda row: "Событие без направления"
        if row["direction_confidence"] < direction_confidence_min
        else "Рост"
        if row["direction_up_proba"] >= 0.5
        else "Снижение",
        axis=1,
    )

    columns = [
        "security",
        "start_time",
        "prediction",
        "event_percent",
        "direction_percent",
        "direction_confidence_percent",
        "max_growth_percent",
        "max_fall_percent",
        "max_move_percent",
        "target_event_24h",
        "news_count",
        "news_title_example",
    ]

    df = df[[column for column in columns if column in df.columns]]
    df = df.rename(columns={
        "security": "Тикер",
        "start_time": "Время сигнала",
        "prediction": "Прогноз",
        "event_percent": "Вероятность события, %",
        "direction_percent": "Вероятность роста, %",
        "direction_confidence_percent": "Уверенность направления, п.п.",
        "max_growth_percent": "Макс. рост 24ч, %",
        "max_fall_percent": "Макс. падение 24ч, %",
        "max_move_percent": "Макс. движение 24ч, %",
        "target_event_24h": "Факт события",
        "news_count": "Новостей в час",
        "news_title_example": "Пример новости",
    })

    return df
