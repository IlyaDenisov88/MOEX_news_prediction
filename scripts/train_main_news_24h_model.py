import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parents[1]
UPDATES_SCRIPTS_DIR = ROOT / "updates" / "scripts"
sys.path.insert(0, str(UPDATES_SCRIPTS_DIR))

from run_news_24h_event_experiment import (  # noqa: E402
    build_dataset,
    feature_columns,
    metric_row,
    split_dataset,
)


RANDOM_STATE = 42


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks-path", default="data/all_1h_stocks_2020_2026.csv")
    parser.add_argument("--news-path", default="data/news_features_2020_2026.parquet")
    parser.add_argument("--out-dir", default="data/final_news_24h_model_outputs")
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--train-end", default="2024-01-01")
    parser.add_argument("--val-end", default="2025-01-01")
    return parser.parse_args()


def make_event_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.05,
            l2_regularization=0.10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def make_direction_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.05,
            l2_regularization=0.10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])


def build_predictions(test, event_model, direction_model, features, threshold):
    event_proba = event_model.predict_proba(test[features])[:, 1]
    direction_proba = direction_model.predict_proba(test[features])[:, 1]
    prediction = test[[
        "security",
        "start_time",
        "news_count",
        "news_title_example",
        "future_max_return_24h",
        "future_min_return_24h",
        "future_abs_return_24h",
        "target_direction_24h",
        "close",
    ]].copy()
    prediction["threshold"] = threshold
    prediction["feature_set"] = "price_only_on_news"
    prediction["model"] = "hist_gradient_boosting"
    prediction["target_event_24h"] = (
        prediction["future_abs_return_24h"] >= threshold
    ).astype(int)
    prediction["event_proba"] = event_proba
    prediction["direction_up_proba"] = direction_proba
    prediction["direction_confidence"] = (prediction["direction_up_proba"] - 0.5).abs()
    return prediction


def build_signal_slices(predictions):
    rows = []
    for event_cutoff in [0.50, 0.60, 0.70, 0.80]:
        selected = predictions[predictions["event_proba"] >= event_cutoff].copy()
        if selected.empty:
            continue

        pred_up = selected["direction_up_proba"] >= 0.5
        actual_up = selected["target_direction_24h"] == 1
        signed_return = selected["future_max_return_24h"].where(
            pred_up,
            -selected["future_min_return_24h"],
        )

        rows.append({
            "threshold": float(selected["threshold"].iloc[0]),
            "feature_set": "price_only_on_news",
            "model": "hist_gradient_boosting",
            "event_cutoff": event_cutoff,
            "signals": int(len(selected)),
            "true_event_rate": float(selected["target_event_24h"].mean()),
            "hit_rate": float((pred_up == actual_up).mean()),
            "avg_event_proba": float(selected["event_proba"].mean()),
            "avg_direction_confidence": float(selected["direction_confidence"].mean()),
            "avg_best_side_return": float(signed_return.mean()),
        })

    return pd.DataFrame(rows)


def main():
    args = parse_args()
    out_dir = ROOT / args.out_dir
    models_dir = out_dir / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(ROOT / args.stocks_path, ROOT / args.news_path)
    train, val, test = split_dataset(dataset, args.train_end, args.val_end)
    _, price_cols = feature_columns(dataset)

    for part in [train, val, test]:
        part["target_event_24h"] = (
            part["future_abs_return_24h"] >= args.threshold
        ).astype(int)

    event_model = make_event_model()
    event_model.fit(train[price_cols], train["target_event_24h"])

    event_train = train[train["target_event_24h"] == 1]
    direction_model = make_direction_model()
    direction_model.fit(event_train[price_cols], event_train["target_direction_24h"])

    predictions = build_predictions(test, event_model, direction_model, price_cols, args.threshold)
    event_metrics = metric_row(test["target_event_24h"], predictions["event_proba"])
    event_metrics.update({
        "task": "event_24h",
        "threshold": args.threshold,
        "feature_set": "price_only_on_news",
        "model": "hist_gradient_boosting",
        "val_pr_auc": average_precision_score(
            val["target_event_24h"],
            event_model.predict_proba(val[price_cols])[:, 1],
        ),
    })

    event_test = test[test["target_event_24h"] == 1]
    direction_metrics = metric_row(
        event_test["target_direction_24h"],
        direction_model.predict_proba(event_test[price_cols])[:, 1],
    )
    direction_metrics.update({
        "task": "direction_inside_true_events",
        "threshold": args.threshold,
        "feature_set": "price_only_on_news",
        "model": "hist_gradient_boosting",
        "val_pr_auc": None,
    })

    results = pd.DataFrame([event_metrics, direction_metrics])
    signal_slices = build_signal_slices(predictions)

    results.to_csv(out_dir / "news_24h_results.csv", index=False)
    predictions.to_csv(out_dir / "news_24h_test_predictions.csv", index=False)
    signal_slices.to_csv(out_dir / "news_24h_signal_slices.csv", index=False)

    dump(event_model, models_dir / "main_news_24h_event_model.joblib")
    dump(direction_model, models_dir / "main_news_24h_direction_model.joblib")

    metadata = {
        "model_role": "main_news_24h_model",
        "event_threshold": args.threshold,
        "horizon": "next_24_hours_after_news",
        "feature_set": "price_only_on_news",
        "model": "hist_gradient_boosting",
        "train_period": f"< {args.train_end}",
        "validation_period": f"{args.train_end} .. {args.val_end}",
        "test_period": f">= {args.val_end}",
        "features": price_cols,
        "rows": {
            "train": int(len(train)),
            "validation": int(len(val)),
            "test": int(len(test)),
        },
        "event_metrics": event_metrics,
        "direction_metrics": direction_metrics,
    }
    (out_dir / "news_24h_model_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("Saved main 24h model to", out_dir)
    print(results.to_string(index=False))
    print(signal_slices.to_string(index=False))


if __name__ == "__main__":
    main()
