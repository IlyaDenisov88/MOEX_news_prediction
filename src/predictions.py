import numpy as np
import pandas as pd


EVENT_DISPLAY_THRESHOLD = 0.60


def get_prediction_label(
    event_proba,
    direction_proba,
    event_threshold=0.5,
    direction_threshold=0.5,
    direction_confidence_min=0.0,
):
    if event_proba < event_threshold:
        return "Нейтрально"

    if abs(direction_proba - direction_threshold) < direction_confidence_min:
        return "Нейтрально"

    if direction_proba >= direction_threshold:
        return "Рост"

    return "Падение"


def get_prediction_label_from_signal(row):
    event_pred = row.get("event_pred")
    direction_pred = row.get("direction_pred")
    event_proba = row["event_proba"]

    if event_proba < EVENT_DISPLAY_THRESHOLD:
        return "Нейтрально"

    if pd.notna(event_pred):
        if int(event_pred) == 0:
            return "Нейтрально"

        if row["direction_proba"] < 0.5:
            return "Падение"

        return "Рост"

    return get_prediction_label(
        row["event_proba"],
        row["direction_proba"],
        event_threshold=0.5,
        direction_threshold=0.5,
        direction_confidence_min=0.06,
    )


def get_prediction_color(label):
    if label == "Рост":
        return "#2ecc71"

    if label == "Падение":
        return "#e74c3c"

    return "#95a5a6"


def create_demo_predictions(stocks):
    latest = (
        stocks
        .sort_values("datetime")
        .groupby("security")
        .tail(1)
        .copy()
    )

    np.random.seed(42)

    latest["event_proba"] = np.random.uniform(0.15, 0.85, size=len(latest))
    latest["direction_proba"] = np.random.uniform(0.15, 0.85, size=len(latest))

    latest["prediction"] = latest.apply(
        lambda row: get_prediction_label(
            row["event_proba"],
            row["direction_proba"],
            event_threshold=0.5,
            direction_threshold=0.5
        ),
        axis=1
    )
    latest["direction_available"] = latest["event_proba"] >= EVENT_DISPLAY_THRESHOLD

    return latest


def create_predictions_from_signals(stocks, signals):
    if signals is None or len(signals) == 0:
        return create_demo_predictions(stocks)

    latest_prices = (
        stocks
        .sort_values("datetime")
        .groupby("security")
        .tail(1)
        .copy()
    )

    latest_signals = signals.copy()
    latest_signals["datetime"] = pd.to_datetime(latest_signals["datetime"], errors="coerce")
    latest_signals = (
        latest_signals
        .dropna(subset=["datetime"])
        .sort_values("datetime")
        .groupby("security")
        .tail(1)
    )

    predictions = latest_prices.merge(
        latest_signals[
            [
                "security",
                "event_proba",
                "direction_proba",
                "event_pred",
                "direction_pred",
            ]
        ],
        on="security",
        how="left",
    )

    predictions["event_proba"] = predictions["event_proba"].fillna(0.0)
    predictions["direction_proba"] = predictions["direction_proba"].fillna(0.5)
    predictions["direction_available"] = predictions["event_proba"] >= EVENT_DISPLAY_THRESHOLD

    if "event_pred" in predictions.columns:
        predictions["direction_available"] = (
            predictions["direction_available"]
            & predictions["event_pred"].fillna(0).astype(int).eq(1)
        )

    predictions["prediction"] = predictions.apply(get_prediction_label_from_signal, axis=1)

    return predictions
