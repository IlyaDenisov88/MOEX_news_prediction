import re
import pandas as pd


def filter_news_by_period(news, period="week"):
    df = news.copy()
    df["published_at"] = pd.to_datetime(df["published_at"])

    max_date = df["published_at"].max()

    if period == "day":
        start_date = max_date - pd.Timedelta(days=1)
    elif period == "week":
        start_date = max_date - pd.Timedelta(days=7)
    elif period == "month":
        start_date = max_date - pd.Timedelta(days=30)
    elif period == "all":
        start_date = df["published_at"].min()
    else:
        raise ValueError("period должен быть: day, week, month или all")

    return df[df["published_at"] >= start_date].copy()


def build_keyword_pattern(keywords):
    pattern_parts = []

    for word in keywords:
        word = str(word).lower().strip()
        escaped = re.escape(word)

        if re.fullmatch(r"[a-zA-Z0-9]+", word):
            pattern_parts.append(rf"\b{escaped}\b")
        else:
            pattern_parts.append(escaped)

    return "|".join(pattern_parts)


def get_news_by_ticker_keywords(news, ticker, ticker_keywords, period="week", last_n=10):
    ticker = ticker.upper().strip()
    keywords = ticker_keywords.get(ticker, [ticker.lower()])

    df = filter_news_by_period(news, period=period)

    pattern = build_keyword_pattern(keywords)

    df = df[
        df["title"]
        .fillna("")
        .astype(str)
        .str.lower()
        .str.contains(pattern, regex=True)
    ].copy()

    df = df.sort_values("published_at", ascending=False).head(last_n)

    return df
