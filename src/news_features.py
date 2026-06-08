from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.functional import softmax


RUBERT_MODEL_NAME = "DeepPavlov/rubert-base-cased-sentence"
SENTIMENT_MODEL_NAME = "mxlcw/rubert-tiny2-russian-financial-sentiment"

BASE_NEWS_COLUMNS = [
    "source",
    "published_at",
    "date",
    "time",
    "title",
    "link",
    "ticker",
    "rubric",
]

SENTIMENT_LABELS = {
    0: "neutral",
    1: "positive",
    2: "negative",
}


def get_device():
    if torch.cuda.is_available():
        return "cuda"

    if torch.backends.mps.is_available():
        return "mps"

    return "cpu"


def load_raw_news(raw_news_path):
    path = Path(raw_news_path)

    if not path.exists():
        raise FileNotFoundError(f"Файл с новостями не найден: {raw_news_path}")

    df = pd.read_csv(path)

    for column in BASE_NEWS_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df[BASE_NEWS_COLUMNS].copy()

    if "published_at" in df.columns:
        df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    else:
        df["published_at"] = pd.to_datetime(df["date"], errors="coerce")

    df = df.dropna(subset=["published_at"])
    df["title"] = df["title"].fillna("").astype(str)
    df = df[df["title"].str.len() > 0].copy()

    df["date"] = df["published_at"].dt.date.astype(str)
    df["time"] = df["published_at"].dt.strftime("%H:%M")

    df = df.drop_duplicates(subset=["title", "link"])
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)

    return df


def build_rubert_embeddings(
    texts,
    model_name=RUBERT_MODEL_NAME,
    batch_size=64,
    device=None,
):
    if device is None:
        device = get_device()

    model = SentenceTransformer(
        model_name,
        device=device,
    )

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )

    return embeddings


def build_sentiment_features(
    texts,
    model_name=SENTIMENT_MODEL_NAME,
    batch_size=64,
    max_length=256,
    device=None,
):
    if device is None:
        device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    model.to(device)
    model.eval()

    results = []

    clean_texts = [str(text) if pd.notna(text) else "" for text in texts]

    for start in tqdm(range(0, len(clean_texts), batch_size), desc="Sentiment"):
        batch_texts = clean_texts[start:start + batch_size]

        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )

        encoded = {key: value.to(device) for key, value in encoded.items()}

        with torch.no_grad():
            outputs = model(**encoded)
            probabilities = softmax(outputs.logits, dim=1).detach().cpu().numpy()

        for row in probabilities:
            neutral_probability = float(row[0])
            positive_probability = float(row[1])
            negative_probability = float(row[2])

            sentiment_class_id = int(row.argmax())
            sentiment_label = SENTIMENT_LABELS[sentiment_class_id]

            results.append(
                {
                    "sentiment_neutral": neutral_probability,
                    "sentiment_positive": positive_probability,
                    "sentiment_negative": negative_probability,
                    "sentiment_label": sentiment_label,
                    "sentiment_score": positive_probability - negative_probability,
                }
            )

    return pd.DataFrame(results)


def embeddings_to_dataframe(
    embeddings,
    prefix="rubert",
):
    return pd.DataFrame(
        embeddings,
        columns=[f"{prefix}_{index}" for index in range(embeddings.shape[1])],
    )


def build_news_features(
    raw_news_path,
    output_path,
    use_rubert=True,
    use_sentiment=True,
    rubert_batch_size=64,
    sentiment_batch_size=64,
    sentiment_max_length=256,
):
    df = load_raw_news(raw_news_path)
    texts = df["title"].fillna("").astype(str).tolist()

    parts = [df.reset_index(drop=True)]

    device = get_device()
    print(f"[INFO] Используется устройство: {device}")
    print(f"[INFO] Новостей для обработки: {len(texts)}")

    if use_rubert:
        print("[INFO] Строю RuBERT-эмбеддинги...")
        embeddings = build_rubert_embeddings(
            texts=texts,
            batch_size=rubert_batch_size,
            device=device,
        )

        embedding_df = embeddings_to_dataframe(
            embeddings=embeddings,
            prefix="rubert",
        )

        parts.append(embedding_df.reset_index(drop=True))

    if use_sentiment:
        print("[INFO] Считаю финансовую тональность...")
        sentiment_df = build_sentiment_features(
            texts=texts,
            batch_size=sentiment_batch_size,
            max_length=sentiment_max_length,
            device=device,
        )

        parts.append(sentiment_df.reset_index(drop=True))

    result = pd.concat(parts, axis=1)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    result.to_parquet(output_file, index=False)

    print(f"[OK] news_features сохранён: {output_file}")
    print(f"[OK] строк: {len(result)}")
    print(f"[OK] колонок: {len(result.columns)}")

    return result

def build_news_features_from_dataframe(
    news_df,
    use_rubert=True,
    use_sentiment=True,
    rubert_batch_size=64,
    sentiment_batch_size=64,
    sentiment_max_length=256,
):
    df = news_df.copy()

    for column in BASE_NEWS_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df[BASE_NEWS_COLUMNS].copy()

    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["published_at"])

    df["title"] = df["title"].fillna("").astype(str)
    df = df[df["title"].str.len() > 0].copy()

    df["date"] = df["published_at"].dt.date.astype(str)
    df["time"] = df["published_at"].dt.strftime("%H:%M")

    df = df.drop_duplicates(subset=["title", "link"])
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)

    texts = df["title"].fillna("").astype(str).tolist()

    parts = [df.reset_index(drop=True)]

    device = get_device()
    print(f"[INFO] Используется устройство: {device}")
    print(f"[INFO] Новостей для обработки: {len(texts)}")

    if len(texts) == 0:
        return df

    if use_rubert:
        print("[INFO] Строю RuBERT-эмбеддинги...")
        embeddings = build_rubert_embeddings(
            texts=texts,
            batch_size=rubert_batch_size,
            device=device,
        )

        embedding_df = embeddings_to_dataframe(
            embeddings=embeddings,
            prefix="rubert",
        )

        parts.append(embedding_df.reset_index(drop=True))

    if use_sentiment:
        print("[INFO] Считаю финансовую тональность...")
        sentiment_df = build_sentiment_features(
            texts=texts,
            batch_size=sentiment_batch_size,
            max_length=sentiment_max_length,
            device=device,
        )

        parts.append(sentiment_df.reset_index(drop=True))

    result = pd.concat(parts, axis=1)

    return result
