from pathlib import Path

import pandas as pd

from src.news_data import update_news_file, normalize_news
from src.news_features import build_news_features_from_dataframe


FEATURE_DEDUP_COLUMNS = [
    "source",
    "published_at",
    "title",
    "link",
]


def make_news_key(df):
    source = df["source"].fillna("").astype(str)
    published_at = pd.to_datetime(df["published_at"], errors="coerce").astype(str)
    title = df["title"].fillna("").astype(str)
    link = df["link"].fillna("").astype(str)

    return source + "||" + published_at + "||" + title + "||" + link


def load_existing_features(features_path):
    path = Path(features_path)

    if not path.exists():
        return pd.DataFrame()

    return pd.read_parquet(path)


def get_new_rows_for_features(
    raw_news_df,
    existing_features_df,
):
    raw_news_df = normalize_news(raw_news_df)

    if raw_news_df.empty:
        return raw_news_df

    if existing_features_df.empty:
        return raw_news_df

    existing_features_df = existing_features_df.copy()

    for column in FEATURE_DEDUP_COLUMNS:
        if column not in existing_features_df.columns:
            existing_features_df[column] = None

    raw_news_df["_news_key"] = make_news_key(raw_news_df)
    existing_features_df["_news_key"] = make_news_key(existing_features_df)

    existing_keys = set(existing_features_df["_news_key"].dropna().astype(str))

    new_rows = raw_news_df[
        ~raw_news_df["_news_key"].astype(str).isin(existing_keys)
    ].copy()

    new_rows = new_rows.drop(columns=["_news_key"], errors="ignore")

    return new_rows.reset_index(drop=True)


def merge_features(
    existing_features_df,
    new_features_df,
):
    if existing_features_df.empty:
        full_df = new_features_df.copy()
    elif new_features_df.empty:
        full_df = existing_features_df.copy()
    else:
        full_df = pd.concat(
            [existing_features_df, new_features_df],
            ignore_index=True,
        )

    if full_df.empty:
        return full_df

    for column in FEATURE_DEDUP_COLUMNS:
        if column not in full_df.columns:
            full_df[column] = None

    full_df["_news_key"] = make_news_key(full_df)

    full_df = full_df.drop_duplicates(
        subset=["_news_key"],
        keep="last",
    )

    full_df = full_df.drop(columns=["_news_key"], errors="ignore")

    full_df["published_at"] = pd.to_datetime(
        full_df["published_at"],
        errors="coerce",
    )

    full_df = full_df.sort_values(
        "published_at",
        ascending=False,
    ).reset_index(drop=True)

    return full_df


def update_news_features_pipeline(
    raw_news_path,
    features_path,
    date_from=None,
    date_till=None,
    use_rubert=True,
    use_sentiment=True,
    rubert_batch_size=64,
    sentiment_batch_size=64,
    sentiment_max_length=256,
):
    print("[STEP 1] Обновляю сырой файл новостей...")

    raw_news_df = update_news_file(
        output_path=raw_news_path,
        date_from=date_from,
        date_till=date_till,
    )

    print(f"[OK] Всего строк в raw news: {len(raw_news_df)}")

    print("[STEP 2] Загружаю существующий parquet с фичами...")

    existing_features_df = load_existing_features(features_path)

    if existing_features_df.empty:
        print("[INFO] Существующий parquet не найден или пустой.")
    else:
        print(f"[OK] Строк в существующем parquet: {len(existing_features_df)}")

    print("[STEP 3] Ищу новые новости для feature generation...")

    new_rows_df = get_new_rows_for_features(
        raw_news_df=raw_news_df,
        existing_features_df=existing_features_df,
    )

    print(f"[INFO] Новых строк для обработки: {len(new_rows_df)}")

    if new_rows_df.empty:
        print("[OK] Новых новостей нет. Parquet не требует обновления.")
        return existing_features_df

    print("[STEP 4] Считаю признаки для новых новостей...")

    new_features_df = build_news_features_from_dataframe(
        news_df=new_rows_df,
        use_rubert=use_rubert,
        use_sentiment=use_sentiment,
        rubert_batch_size=rubert_batch_size,
        sentiment_batch_size=sentiment_batch_size,
        sentiment_max_length=sentiment_max_length,
    )

    print(f"[OK] Новых feature-строк: {len(new_features_df)}")

    print("[STEP 5] Объединяю старые и новые признаки...")

    full_features_df = merge_features(
        existing_features_df=existing_features_df,
        new_features_df=new_features_df,
    )

    output_file = Path(features_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    full_features_df.to_parquet(output_file, index=False)

    print(f"[OK] Обновлён parquet: {output_file}")
    print(f"[OK] Всего строк в parquet: {len(full_features_df)}")
    print(f"[OK] Всего колонок в parquet: {len(full_features_df.columns)}")

    return full_features_df
