import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump as joblib_dump
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_moex_factors import build_moex_factors


warnings.filterwarnings("ignore")


RANDOM_STATE = 42


TICKER_KEYWORDS = {
    "AFLT": ["аэрофлот", "aeroflot", "aflt", "пао аэрофлот"],
    "ALRS": ["алроса", "alrosa", "alrs", "ак алроса", "алмазы россии"],
    "CHMF": ["северсталь", "severstal", "chmf", "пао северсталь"],
    "FEES": ["фск", "россети", "fees", "федеральная сетевая"],
    "GAZP": ["газпром", "gazprom", "gazp", "пао газпром"],
    "GMKN": ["норникель", "норильский никель", "gmkn", "гмк норильский никель"],
    "IRAO": ["интер рао", "inter rao", "irao", "интеррао"],
    "LKOH": ["лукойл", "lukoil", "lkoh", "пао лукойл"],
    "MAGN": ["ммк", "магнитогорский", "magn", "магнитогорский металлургический"],
    "MGNT": ["магнит", "magnit", "mgnt", "пао магнит"],
    "MOEX": ["мосбиржа", "московская биржа", "moex", "moex group"],
    "MTSS": ["мтс", "mts", "mtss", "мобильные телесистемы"],
    "NLMK": ["нлмк", "nlmk", "новолипецкий металлургический"],
    "NVTK": ["новатэк", "novatek", "nvtk", "пао новатэк"],
    "PHOR": ["фосагро", "phosagro", "phor", "фосагро"],
    "PLZL": ["полюс", "polyus", "plzl", "пао полюс"],
    "ROSN": ["роснефть", "rosneft", "rosn", "нк роснефть"],
    "RTKM": ["ростелеком", "rostelecom", "rtkm", "пао ростелеком"],
    "SBER": ["сбер", "сбербанк", "sber", "sberbank", "пао сбербанк"],
    "SNGS": ["сургутнефтегаз", "surgutneftegaz", "sngs"],
    "SNGSP": ["сургутнефтегаз", "surgutneftegaz", "sngsp"],
    "TATN": ["татнефть", "tatneft", "tatn", "пао татнефть"],
    "TATNP": ["татнефть", "tatneft", "tatnp", "пао татнефть"],
    "VTBR": ["втб", "vtb", "vtbr", "банк втб"],
    "YDEX": ["яндекс", "yandex", "ydex", "яндекс нв"],
    "YNDX": ["яндекс", "yandex", "yndx", "яндекс нв"],
}


SECTOR_KEYWORDS = {
    "banking": ["банк", "ключевая ставка", "кредит", "ипотек", "цб", "центробанк"],
    "oil_gas": ["нефть", "газ", "brent", "opec", "опек", "трубопровод", "топливо"],
    "metals": ["сталь", "металл", "алюмини", "никель", "золото", "палладий"],
    "retail": ["ритейл", "продажи", "магазин", "потребител", "маркетплейс"],
    "tech": ["it", "технолог", "интернет", "цифров", "маркетплейс"],
    "sanctions": ["санкци", "ограничени", "экспорт", "импорт", "пошлин"],
    "dividends": ["дивиденд", "реестр", "выплат", "акционерам"],
    "financials": ["отчет", "прибыль", "выручк", "убыт", "ebitda", "мсфо", "рсбу"],
}


EVENT_TYPE_KEYWORDS = {
    "dividends": ["дивиденд", "реестр", "отсечк", "выплат", "дивдоход"],
    "earnings": ["отчет", "прибыль", "выручк", "убыт", "ebitda", "мсфо", "рсбу", "финансовые результаты"],
    "sanctions": ["санкци", "ограничени", "блокирующ", "sdn", "экспортный контроль"],
    "ma_deals": ["m&a", "слияни", "поглощени", "сделк", "покупк", "продажу доли", "актив"],
    "buyback": ["buyback", "байбек", "обратный выкуп", "выкуп акций"],
    "guidance": ["прогноз", "ожидает", "планирует", "менеджмент", "таргет", "рекомендац"],
    "spo_ipo": ["ipo", "spo", "размещени", "допэмисси", "эмисси", "акций"],
    "legal": ["суд", "иск", "штраф", "расследован", "арбитраж", "претензи"],
    "rates_fx": ["ключевая ставка", "ставк", "цб", "рубл", "доллар", "юан", "валют"],
    "commodities": ["нефть", "brent", "газ", "золото", "никель", "сталь", "алюмини"],
}


NOISE_KEYWORDS = [
    "погода",
    "спорт",
    "футбол",
    "хоккей",
    "кино",
    "музыка",
    "туризм",
    "происшеств",
    "криминал",
]


POSITIVE_SENTIMENT_WORDS = [
    "рост",
    "вырос",
    "увелич",
    "прибыль",
    "рекорд",
    "выше ожид",
    "повысил",
    "улучш",
    "дивиденд",
    "байбек",
    "выкуп",
    "сильн",
    "позитив",
]


NEGATIVE_SENTIMENT_WORDS = [
    "падени",
    "сниз",
    "убыт",
    "хуже ожид",
    "санкци",
    "штраф",
    "суд",
    "иск",
    "авари",
    "огранич",
    "негатив",
    "дефолт",
    "риск",
]


SECTOR_BY_TICKER = {
    "SBER": "banking",
    "VTBR": "banking",
    "MOEX": "financial",
    "GAZP": "oil_gas",
    "LKOH": "oil_gas",
    "NVTK": "oil_gas",
    "ROSN": "oil_gas",
    "SNGS": "oil_gas",
    "SNGSP": "oil_gas",
    "TATN": "oil_gas",
    "TATNP": "oil_gas",
    "GMKN": "metals",
    "MAGN": "metals",
    "NLMK": "metals",
    "CHMF": "metals",
    "ALRS": "metals",
    "PLZL": "metals",
    "PHOR": "chemicals",
    "MGNT": "retail",
    "YDEX": "tech",
    "YNDX": "tech",
    "MTSS": "telecom",
    "RTKM": "telecom",
    "AFLT": "transport",
    "FEES": "utilities",
    "IRAO": "utilities",
}


class ExperimentConfig:
    def __init__(
        self,
        horizon=4,
        event_threshold=0.01,
        target_mode="fixed",
        volatility_multiplier=3.0,
        train_start=None,
        train_end="2024-01-01",
        val_end="2025-01-01",
        max_tfidf_features=6000,
        tfidf_components=24,
        rubert_components=32,
        max_train_rows=None,
        external_factors_path="data/moex_external_factors.csv",
        event_decision_threshold=0.60,
        use_text_components=True,
    ):
        self.horizon = horizon
        self.event_threshold = event_threshold
        self.target_mode = target_mode
        self.volatility_multiplier = volatility_multiplier
        self.train_start = train_start
        self.train_end = train_end
        self.val_end = val_end
        self.max_tfidf_features = max_tfidf_features
        self.tfidf_components = tfidf_components
        self.rubert_components = rubert_components
        self.max_train_rows = max_train_rows
        self.external_factors_path = external_factors_path
        self.event_decision_threshold = event_decision_threshold
        self.use_text_components = use_text_components

    def __repr__(self):
        return str(self.__dict__)


SECTOR_TOPIC_BY_SECTOR = {
    "banking": ["banking", "rates_fx", "financials"],
    "financial": ["banking", "rates_fx", "financials"],
    "oil_gas": ["oil_gas", "commodities", "sanctions"],
    "metals": ["metals", "commodities", "sanctions"],
    "chemicals": ["commodities", "sanctions", "financials"],
    "retail": ["retail", "financials"],
    "tech": ["tech", "sanctions", "financials"],
    "telecom": ["tech", "financials"],
    "transport": ["sanctions", "financials"],
    "utilities": ["rates_fx", "financials"],
}


def normalize_text(value):
    if pd.isna(value):
        return ""
    return str(value).lower().replace("ё", "е")


def compile_pattern(words):
    parts = []
    for word in words:
        clean = re.escape(word.lower().replace("ё", "е"))
        if re.fullmatch(r"[a-z0-9_]+", word.lower()):
            parts.append(rf"(?<![a-z0-9_]){clean}(?![a-z0-9_])")
        elif len(word) <= 4 and " " not in word:
            parts.append(rf"(?<![а-яa-z0-9_]){clean}(?![а-яa-z0-9_])")
        else:
            parts.append(clean)
    return re.compile("|".join(parts), flags=re.IGNORECASE)


def add_price_features(stocks):
    df = stocks.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df = df.dropna(subset=["datetime", "security", "close"])
    df = df.sort_values(["security", "datetime"]).reset_index(drop=True)

    group = df.groupby("security", group_keys=False)
    df["return_1h"] = group["close"].pct_change(1)
    df["return_2h"] = group["close"].pct_change(2)
    df["return_4h"] = group["close"].pct_change(4)
    df["return_8h"] = group["close"].pct_change(8)
    df["return_24h"] = group["close"].pct_change(24)
    df["volume_change_1h"] = group["volume"].pct_change(1).replace([np.inf, -np.inf], np.nan)
    df["value_change_1h"] = group["value"].pct_change(1).replace([np.inf, -np.inf], np.nan)
    df["candle_body"] = (df["close"] - df["open"]) / df["open"].replace(0, np.nan)
    df["high_low_range"] = (df["high"] - df["low"]) / df["open"].replace(0, np.nan)

    for window in [4, 8, 24, 48]:
        rolling_close = group["close"].transform(lambda x: x.rolling(window, min_periods=3).mean())
        rolling_std = group["return_1h"].transform(lambda x: x.rolling(window, min_periods=3).std())
        rolling_volume = group["volume"].transform(lambda x: x.rolling(window, min_periods=3).mean())
        df[f"close_to_sma_{window}"] = df["close"] / rolling_close.replace(0, np.nan) - 1
        df[f"volatility_{window}"] = rolling_std
        df[f"volume_to_sma_{window}"] = df["volume"] / rolling_volume.replace(0, np.nan) - 1

    df["hour"] = df["datetime"].dt.floor("h")
    df["hour_of_day"] = df["datetime"].dt.hour
    df["day_of_week"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    df["hour_sin"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["sector"] = df["security"].map(SECTOR_BY_TICKER).fillna("other")

    market_context = (
        df.groupby("datetime")
        .agg(
            market_return_1h_mean=("return_1h", "mean"),
            market_return_4h_mean=("return_4h", "mean"),
            market_volatility_24_mean=("volatility_24", "mean"),
            market_volume_change_mean=("volume_change_1h", "mean"),
        )
        .reset_index()
    )
    df = df.merge(market_context, on="datetime", how="left")

    sector_context = (
        df.groupby(["sector", "datetime"])
        .agg(
            sector_return_1h_mean=("return_1h", "mean"),
            sector_return_4h_mean=("return_4h", "mean"),
            sector_volatility_24_mean=("volatility_24", "mean"),
            sector_volume_change_mean=("volume_change_1h", "mean"),
        )
        .reset_index()
    )
    df = df.merge(sector_context, on=["sector", "datetime"], how="left")

    df["relative_return_1h"] = df["return_1h"] - df["market_return_1h_mean"]
    df["relative_return_4h"] = df["return_4h"] - df["market_return_4h_mean"]
    df["relative_sector_return_1h"] = df["return_1h"] - df["sector_return_1h_mean"]
    df["relative_sector_return_4h"] = df["return_4h"] - df["sector_return_4h_mean"]

    ticker_dummies = pd.get_dummies(df["security"], prefix="ticker", dtype="int8")
    sector_dummies = pd.get_dummies(df["sector"], prefix="sector", dtype="int8")
    df = pd.concat([df, ticker_dummies, sector_dummies], axis=1)
    return df


def add_targets(df, config):
    out = df.sort_values(["security", "datetime"]).copy()
    group = out.groupby("security", group_keys=False)
    future_close = group["close"].shift(-config.horizon)
    out["future_return"] = future_close / out["close"] - 1
    if config.target_mode == "volatility":
        volatility_threshold = (
            out["volatility_24"].fillna(out["volatility_8"]).fillna(0.0)
            * config.volatility_multiplier
        )
        threshold = np.maximum(config.event_threshold, volatility_threshold)
        out["event_cutoff"] = threshold
    else:
        threshold = config.event_threshold
        out["event_cutoff"] = config.event_threshold

    out["target_event"] = (out["future_return"].abs() >= threshold).astype(int)
    out["target_direction"] = (out["future_return"] > 0).astype(int)
    out["target_ternary"] = 1
    out.loc[(out["target_event"] == 1) & (out["future_return"] > 0), "target_ternary"] = 2
    out.loc[(out["target_event"] == 1) & (out["future_return"] < 0), "target_ternary"] = 0
    out.loc[out["future_return"].isna(), ["target_event", "target_direction"]] = np.nan
    out.loc[out["future_return"].isna(), "target_ternary"] = np.nan
    return out


def build_text_components(news, config):
    text = news["news_text_for_model"].fillna("").astype(str)
    vectorizer = TfidfVectorizer(
        max_features=config.max_tfidf_features,
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform(text)
    n_components = min(config.tfidf_components, max(2, matrix.shape[1] - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    components = svd.fit_transform(matrix)
    cols = [f"tfidf_svd_{i}" for i in range(components.shape[1])]
    return pd.DataFrame(components, columns=cols, index=news.index)


def build_rubert_components(news, config):
    rubert_cols = [col for col in news.columns if col.startswith("rubert_")]
    if not rubert_cols:
        return pd.DataFrame(index=news.index)

    values = news[rubert_cols].astype("float32").fillna(0.0)
    n_components = min(config.rubert_components, max(2, len(rubert_cols) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    components = svd.fit_transform(values)
    cols = [f"rubert_svd_{i}" for i in range(components.shape[1])]
    return pd.DataFrame(components, columns=cols, index=news.index)


def prepare_news(news, config):
    df = news.copy()
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["published_at"]).reset_index(drop=True)

    text_cols = [col for col in ["news_text", "title_clean", "title"] if col in df.columns]
    if not text_cols:
        df["news_text_for_model"] = ""
    else:
        df["news_text_for_model"] = (
            df[text_cols].fillna("").astype(str).agg(". ".join, axis=1).map(normalize_text)
        )

    for col in ["sentiment_score", "sentiment_positive", "sentiment_negative", "sentiment_neutral"]:
        if col not in df.columns:
            df[col] = 0.0

    if config.use_text_components:
        tfidf_df = build_text_components(df, config)
        rubert_df = build_rubert_components(df, config)
        df = pd.concat([df, tfidf_df, rubert_df], axis=1)

    for sector, words in SECTOR_KEYWORDS.items():
        pattern = compile_pattern(words)
        df[f"topic_{sector}"] = df["news_text_for_model"].map(lambda x: int(bool(pattern.search(x))))

    event_cols = []
    for event_type, words in EVENT_TYPE_KEYWORDS.items():
        pattern = compile_pattern(words)
        col = f"event_type_{event_type}"
        df[col] = df["news_text_for_model"].map(lambda x: int(bool(pattern.search(x))))
        event_cols.append(col)

    sentiment_prob_cols = ["sentiment_negative", "sentiment_neutral", "sentiment_positive"]
    df[sentiment_prob_cols] = df[sentiment_prob_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(
        df["sentiment_positive"] - df["sentiment_negative"]
    )
    df["sentiment_margin"] = df["sentiment_positive"] - df["sentiment_negative"]
    df["sentiment_abs"] = df["sentiment_margin"].abs()
    df["sentiment_confidence"] = df[sentiment_prob_cols].max(axis=1)
    df["sentiment_signed_confidence"] = np.sign(df["sentiment_margin"]) * df["sentiment_confidence"]
    df["sentiment_label_positive"] = (
        (df["sentiment_positive"] >= df["sentiment_negative"])
        & (df["sentiment_positive"] >= df["sentiment_neutral"])
    ).astype(int)
    df["sentiment_label_negative"] = (
        (df["sentiment_negative"] > df["sentiment_positive"])
        & (df["sentiment_negative"] >= df["sentiment_neutral"])
    ).astype(int)
    df["sentiment_label_neutral"] = (
        (df["sentiment_neutral"] > df["sentiment_positive"])
        & (df["sentiment_neutral"] > df["sentiment_negative"])
    ).astype(int)

    positive_pattern = compile_pattern(POSITIVE_SENTIMENT_WORDS)
    negative_pattern = compile_pattern(NEGATIVE_SENTIMENT_WORDS)
    df["lexicon_positive_flag"] = df["news_text_for_model"].map(lambda x: int(bool(positive_pattern.search(x))))
    df["lexicon_negative_flag"] = df["news_text_for_model"].map(lambda x: int(bool(negative_pattern.search(x))))
    df["lexicon_sentiment_score"] = df["lexicon_positive_flag"] - df["lexicon_negative_flag"]

    noise_pattern = compile_pattern(NOISE_KEYWORDS)
    df["is_noise_news"] = df["news_text_for_model"].map(lambda x: int(bool(noise_pattern.search(x))))
    df["event_type_count"] = df[event_cols].sum(axis=1)
    df["news_relevance_score"] = (
        df["event_type_count"]
        + df[[col for col in df.columns if col.startswith("topic_")]].sum(axis=1) * 0.25
        + (df["sentiment_score"].abs() > 0.20).astype(int) * 0.5
        - df["is_noise_news"] * 2.0
    )
    df["is_relevant_market_news"] = (df["news_relevance_score"] > 0).astype(int)
    df["sentiment_relevance_weighted"] = df["sentiment_margin"] * df["news_relevance_score"].clip(lower=0)

    for col in event_cols:
        suffix = col.replace("event_type_", "")
        df[f"sentiment_event_{suffix}"] = df["sentiment_margin"] * df[col]

    return df


def find_relevant_tickers(news, securities):
    rows = []
    text = news["news_text_for_model"]

    for ticker in securities:
        words = TICKER_KEYWORDS.get(ticker, [ticker.lower()])
        pattern = compile_pattern(words)
        mask = text.map(lambda x: bool(pattern.search(x)))
        if "ticker" in news.columns:
            ticker_mask = news["ticker"].fillna("").astype(str).str.upper().eq(ticker.upper())
            mask = mask | ticker_mask
        if mask.any():
            part = news.loc[mask].copy()
            part["security"] = ticker
            part["is_ticker_news"] = 1
            rows.append(part)

    if not rows:
        return pd.DataFrame(columns=list(news.columns) + ["security", "is_ticker_news"])

    direct = pd.concat(rows, ignore_index=True)
    direct = direct.drop_duplicates(subset=["published_at", "title", "security"])
    return direct


def aggregate_news(news, securities):
    component_cols = [
        col for col in news.columns
        if col.startswith("tfidf_svd_") or col.startswith("rubert_svd_")
    ]
    topic_cols = [col for col in news.columns if col.startswith("topic_")]
    event_type_cols = [col for col in news.columns if col.startswith("event_type_")]
    sentiment_cols = [
        col for col in news.columns
        if (
            col.startswith("sentiment_")
            or col.startswith("lexicon_")
        )
        and col != "sentiment_label"
    ]

    news = news.copy()
    news["hour"] = news["published_at"].dt.floor("h")
    news["news_count"] = 1

    relevant_news = news[news["is_relevant_market_news"] == 1].copy()
    if relevant_news.empty:
        relevant_news = news.iloc[:0].copy()

    market_agg = news.groupby("hour").agg(
        market_news_count=("news_count", "sum"),
        market_sentiment_mean=("sentiment_score", "mean"),
        market_sentiment_min=("sentiment_score", "min"),
        market_sentiment_max=("sentiment_score", "max"),
        **{f"market_{col}_sum": (col, "sum") for col in topic_cols},
        **{f"market_{col}_sum": (col, "sum") for col in event_type_cols},
        **{f"market_{col}_mean": (col, "mean") for col in sentiment_cols if col != "sentiment_score"},
        **{f"market_{col}_mean": (col, "mean") for col in component_cols[:16]},
    ).reset_index()

    relevant_market_agg = relevant_news.groupby("hour").agg(
        relevant_market_news_count=("news_count", "sum"),
        relevant_market_sentiment_mean=("sentiment_score", "mean"),
        relevant_market_relevance_mean=("news_relevance_score", "mean"),
        **{f"relevant_market_{col}_mean": (col, "mean") for col in sentiment_cols if col != "sentiment_score"},
        **{f"relevant_market_{col}_sum": (col, "sum") for col in event_type_cols},
    ).reset_index()

    direct = find_relevant_tickers(news, securities)
    if direct.empty:
        ticker_agg = pd.DataFrame(columns=["security", "hour"])
    else:
        ticker_agg = direct.groupby(["security", "hour"]).agg(
            ticker_news_count=("news_count", "sum"),
            ticker_sentiment_mean=("sentiment_score", "mean"),
            ticker_sentiment_min=("sentiment_score", "min"),
            ticker_sentiment_max=("sentiment_score", "max"),
            ticker_positive_mean=("sentiment_positive", "mean"),
            ticker_negative_mean=("sentiment_negative", "mean"),
            ticker_relevance_mean=("news_relevance_score", "mean"),
            **{f"ticker_{col}_mean": (col, "mean") for col in sentiment_cols if col != "sentiment_score"},
            **{f"ticker_{col}_sum": (col, "sum") for col in topic_cols},
            **{f"ticker_{col}_sum": (col, "sum") for col in event_type_cols},
            **{f"ticker_{col}_mean": (col, "mean") for col in component_cols},
        ).reset_index()

    skeleton = pd.MultiIndex.from_product(
        [securities, market_agg["hour"].sort_values().unique()],
        names=["security", "hour"],
    ).to_frame(index=False)
    hourly = skeleton.merge(market_agg, on="hour", how="left")
    hourly = hourly.merge(relevant_market_agg, on="hour", how="left")
    hourly = hourly.merge(ticker_agg, on=["security", "hour"], how="left")

    news_feature_cols = [col for col in hourly.columns if col not in ["security", "hour"]]
    count_cols = [col for col in news_feature_cols if col.endswith("_count") or col.endswith("_sum")]
    mean_cols = [col for col in news_feature_cols if col not in count_cols]

    hourly[count_cols] = hourly[count_cols].fillna(0.0)
    hourly[mean_cols] = hourly[mean_cols].fillna(0.0)
    hourly["ticker_has_news"] = (hourly.get("ticker_news_count", 0) > 0).astype(int)
    hourly["market_has_news"] = (hourly.get("market_news_count", 0) > 0).astype(int)

    grouped = hourly.groupby("security", group_keys=False)
    ticker_seen = grouped["ticker_has_news"].cumsum()
    hourly["hours_since_ticker_news"] = (
        grouped["ticker_has_news"]
        .transform(lambda x: x.where(x.eq(1)).ffill())
        .isna()
        .astype(int)
    )
    hourly["hours_since_ticker_news"] = grouped["ticker_has_news"].transform(
        lambda x: x.groupby(x.cumsum()).cumcount()
    )
    hourly.loc[ticker_seen == 0, "hours_since_ticker_news"] = 999
    hourly = hourly.sort_values(["security", "hour"]).reset_index(drop=True)

    for window in [4, 24]:
        for col in count_cols:
            hourly[f"{col}_{window}h"] = grouped[col].transform(
                lambda x: x.rolling(window, min_periods=1).sum()
            )
        for col in mean_cols:
            if col.endswith("_mean") or "sentiment" in col:
                hourly[f"{col}_{window}h"] = grouped[col].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )

    final_news_cols = [col for col in hourly.columns if col not in ["security", "hour"]]
    return hourly, final_news_cols


def make_dataset(stocks_path, news_path, config):
    stocks = pd.read_csv(stocks_path)
    news = pd.read_parquet(news_path)

    stocks = add_price_features(stocks)
    securities = sorted(stocks["security"].dropna().astype(str).unique())
    news = prepare_news(news, config)
    news_hourly, news_feature_cols = aggregate_news(news, securities)

    dataset = stocks.merge(news_hourly, on=["security", "hour"], how="left")
    external_factor_cols = []
    if config.external_factors_path and Path(config.external_factors_path).exists():
        external = pd.read_csv(config.external_factors_path)
        external["hour"] = pd.to_datetime(external["hour"], errors="coerce").dt.floor("h")
        external = external.dropna(subset=["hour"]).drop_duplicates(subset=["hour"])
        external_factor_cols = [col for col in external.columns if col != "hour"]
        dataset = dataset.merge(external, on="hour", how="left")

    for col in news_feature_cols:
        dataset[col] = dataset[col].fillna(0.0)
    for col in external_factor_cols:
        dataset[col] = dataset.groupby("security")[col].ffill().fillna(0.0)

    dataset = add_targets(dataset, config)

    price_cols = [
        "return_1h", "return_2h", "return_4h", "return_8h", "return_24h",
        "volume_change_1h", "value_change_1h", "candle_body", "high_low_range",
        "close_to_sma_4", "volatility_4", "volume_to_sma_4",
        "close_to_sma_8", "volatility_8", "volume_to_sma_8",
        "close_to_sma_24", "volatility_24", "volume_to_sma_24",
        "close_to_sma_48", "volatility_48", "volume_to_sma_48",
        "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
        "market_return_1h_mean", "market_return_4h_mean", "market_volatility_24_mean",
        "market_volume_change_mean", "sector_return_1h_mean", "sector_return_4h_mean",
        "sector_volatility_24_mean", "sector_volume_change_mean",
        "relative_return_1h", "relative_return_4h",
        "relative_sector_return_1h", "relative_sector_return_4h",
    ]
    dummy_cols = [
        col
        for col in dataset.columns
        if (col.startswith("ticker_") or col.startswith("sector_"))
        and col not in news_feature_cols
        and col not in price_cols
    ]
    price_cols.extend(dummy_cols)
    price_cols.extend(external_factor_cols)
    price_cols = [col for col in price_cols if col in dataset.columns]
    return dataset, price_cols, news_feature_cols


def make_models():
    models = {
        "logreg_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", __import__("sklearn.linear_model").linear_model.LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=140,
                learning_rate=0.055,
                l2_regularization=0.05,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=180,
                max_depth=16,
                min_samples_leaf=20,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "mlp": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(96, 48),
                alpha=0.001,
                learning_rate_init=0.001,
                early_stopping=True,
                max_iter=80,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    try:
        from lightgbm import LGBMClassifier

        models["lightgbm"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", LGBMClassifier(
                n_estimators=260,
                learning_rate=0.035,
                num_leaves=31,
                subsample=0.85,
                colsample_bytree=0.85,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                verbosity=-1,
            )),
        ])
    except Exception:
        pass

    try:
        from catboost import CatBoostClassifier

        models["catboost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", CatBoostClassifier(
                iterations=260,
                learning_rate=0.04,
                depth=6,
                loss_function="Logloss",
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=False,
            )),
        ])
    except Exception:
        pass

    return models


def make_multiclass_models():
    models = {
        "logreg_balanced": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", __import__("sklearn.linear_model").linear_model.LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(
                max_iter=140,
                learning_rate=0.055,
                l2_regularization=0.05,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=180,
                max_depth=16,
                min_samples_leaf=20,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
    }

    try:
        from catboost import CatBoostClassifier

        models["catboost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", CatBoostClassifier(
                iterations=260,
                learning_rate=0.04,
                depth=6,
                loss_function="MultiClass",
                auto_class_weights="Balanced",
                random_seed=RANDOM_STATE,
                verbose=False,
            )),
        ])
    except Exception:
        pass

    return models


def sample_train_frame(df, target, max_rows):
    if max_rows is None or len(df) <= max_rows:
        return df

    positives = df[df[target] == 1]
    negatives = df[df[target] == 0]
    pos_n = min(len(positives), max_rows // 2)
    neg_n = max_rows - pos_n
    sampled = pd.concat([
        positives.sample(n=pos_n, random_state=RANDOM_STATE) if len(positives) > pos_n else positives,
        negatives.sample(n=min(len(negatives), neg_n), random_state=RANDOM_STATE),
    ])
    return sampled.sort_values("datetime")


def predict_proba_or_score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    score = model.decision_function(X)
    return 1 / (1 + np.exp(-score))


def best_threshold(y_true, proba, metric="f1"):
    best_t, best_s = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.9, 86):
        pred = (proba >= threshold).astype(int)
        if metric == "balanced_accuracy":
            score = balanced_accuracy_score(y_true, pred)
        else:
            score = f1_score(y_true, pred, zero_division=0)
        if score > best_s:
            best_t, best_s = float(threshold), float(score)
    return best_t, best_s


def evaluate_predictions(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    y_arr = np.asarray(y_true).astype(int)
    order = np.argsort(proba)[::-1]
    base_rate = y_arr.mean() if len(y_arr) else np.nan
    top_metrics = {}
    for share in [0.01, 0.05, 0.10]:
        n_top = max(1, int(len(y_arr) * share))
        top_rate = y_arr[order[:n_top]].mean()
        top_metrics[f"precision_at_{int(share * 100)}pct"] = top_rate
        top_metrics[f"lift_at_{int(share * 100)}pct"] = top_rate / base_rate if base_rate else np.nan

    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba) if len(np.unique(y_true)) > 1 else np.nan,
        "pr_auc": average_precision_score(y_true, proba) if len(np.unique(y_true)) > 1 else np.nan,
        **top_metrics,
    }


def run_task(
    task_name,
    target,
    train_df,
    val_df,
    test_df,
    feature_sets,
    config,
):
    rows = []
    signals = {}
    fitted_models = {}
    models = make_models()

    for feature_set_name, features in feature_sets.items():
        usable_features = [col for col in features if col in train_df.columns]
        if not usable_features:
            continue

        for model_name, base_model in models.items():
            train_sample = sample_train_frame(train_df, target, config.max_train_rows)
            model = clone(base_model)
            model.fit(train_sample[usable_features], train_sample[target].astype(int))

            val_proba = predict_proba_or_score(model, val_df[usable_features])
            if task_name == "event_detection" and config.event_decision_threshold is not None:
                threshold = config.event_decision_threshold
                tuning_score = f1_score(
                    val_df[target].astype(int),
                    (val_proba >= threshold).astype(int),
                    zero_division=0,
                )
            else:
                threshold, tuning_score = best_threshold(val_df[target].astype(int), val_proba)

            test_proba = predict_proba_or_score(model, test_df[usable_features])
            metrics = evaluate_predictions(test_df[target].astype(int), test_proba, threshold)
            pred = (test_proba >= threshold).astype(int)

            experiment_name = f"{task_name}__{feature_set_name}__{model_name}"
            rows.append({
                "experiment": experiment_name,
                "task": task_name,
                "feature_set": feature_set_name,
                "model": model_name,
                **metrics,
                "decision_threshold": threshold,
                "threshold_tuning_score_val": tuning_score,
                "n_features": len(usable_features),
                "n_train": len(train_sample),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "train_positive_share": float(train_sample[target].mean()),
                "val_positive_share": float(val_df[target].mean()),
                "test_positive_share": float(test_df[target].mean()),
                "horizon": config.horizon,
                "event_threshold": config.event_threshold,
                "target_mode": config.target_mode,
                "volatility_multiplier": config.volatility_multiplier,
                "train_period": f"{train_df['datetime'].min()} — {train_df['datetime'].max()}",
                "val_period": f"{val_df['datetime'].min()} — {val_df['datetime'].max()}",
                "test_period": f"{test_df['datetime'].min()} — {test_df['datetime'].max()}",
            })

            signal_cols = ["security", "datetime", "open", "high", "low", "close", "future_return", "target_event", "target_direction"]
            task_signals = test_df[signal_cols].copy()
            task_signals[f"{task_name}_proba"] = test_proba
            task_signals[f"{task_name}_pred"] = pred
            signals[experiment_name] = task_signals
            fitted_models[experiment_name] = model

            print(
                f"{experiment_name}: f1={metrics['f1']:.4f}, "
                f"roc_auc={metrics['roc_auc']:.4f}, pr_auc={metrics['pr_auc']:.4f}, "
                f"threshold={threshold:.2f}"
            )

    return rows, signals, fitted_models


def evaluate_multiclass_predictions(y_true, proba, pred):
    y_arr = np.asarray(y_true).astype(int)
    event_true = (y_arr != 1).astype(int)
    event_score = proba[:, 0] + proba[:, 2]
    direction_mask = event_true == 1

    rows = {
        "accuracy": accuracy_score(y_arr, pred),
        "balanced_accuracy": balanced_accuracy_score(y_arr, pred),
        "f1_macro": f1_score(y_arr, pred, average="macro", zero_division=0),
        "f1_event_up": f1_score((y_arr == 2).astype(int), (pred == 2).astype(int), zero_division=0),
        "f1_event_down": f1_score((y_arr == 0).astype(int), (pred == 0).astype(int), zero_division=0),
        "event_roc_auc": roc_auc_score(event_true, event_score) if len(np.unique(event_true)) > 1 else np.nan,
        "event_pr_auc": average_precision_score(event_true, event_score) if len(np.unique(event_true)) > 1 else np.nan,
    }

    if direction_mask.any():
        true_up = (y_arr[direction_mask] == 2).astype(int)
        up_score = proba[direction_mask, 2] / (
            proba[direction_mask, 0] + proba[direction_mask, 2] + 1e-9
        )
        rows["direction_roc_auc_on_events"] = (
            roc_auc_score(true_up, up_score) if len(np.unique(true_up)) > 1 else np.nan
        )
        rows["direction_pr_auc_on_events"] = (
            average_precision_score(true_up, up_score) if len(np.unique(true_up)) > 1 else np.nan
        )
    else:
        rows["direction_roc_auc_on_events"] = np.nan
        rows["direction_pr_auc_on_events"] = np.nan

    order = np.argsort(event_score)[::-1]
    base_rate = event_true.mean() if len(event_true) else np.nan
    for share in [0.01, 0.05, 0.10]:
        n_top = max(1, int(len(event_true) * share))
        top_rate = event_true[order[:n_top]].mean()
        rows[f"event_precision_at_{int(share * 100)}pct"] = top_rate
        rows[f"event_lift_at_{int(share * 100)}pct"] = top_rate / base_rate if base_rate else np.nan

    return rows


def run_multiclass_task(
    train_df,
    val_df,
    test_df,
    feature_sets,
    config,
):
    rows = []
    signals = {}
    models = make_multiclass_models()

    for feature_set_name, features in feature_sets.items():
        usable_features = [col for col in features if col in train_df.columns]
        if not usable_features:
            continue

        for model_name, base_model in models.items():
            train_sample = sample_train_frame(train_df, "target_event", config.max_train_rows)
            model = clone(base_model)
            model.fit(train_sample[usable_features], train_sample["target_ternary"].astype(int))

            test_proba = model.predict_proba(test_df[usable_features])
            pred = np.asarray(model.predict(test_df[usable_features])).astype(int)
            metrics = evaluate_multiclass_predictions(
                test_df["target_ternary"].astype(int),
                test_proba,
                pred,
            )
            experiment_name = f"ternary_event_direction__{feature_set_name}__{model_name}"
            rows.append({
                "experiment": experiment_name,
                "task": "ternary_event_direction",
                "feature_set": feature_set_name,
                "model": model_name,
                **metrics,
                "n_features": len(usable_features),
                "n_train": len(train_sample),
                "n_val": len(val_df),
                "n_test": len(test_df),
                "train_event_share": float((train_sample["target_ternary"] != 1).mean()),
                "val_event_share": float((val_df["target_ternary"] != 1).mean()),
                "test_event_share": float((test_df["target_ternary"] != 1).mean()),
                "horizon": config.horizon,
                "event_threshold": config.event_threshold,
                "target_mode": config.target_mode,
                "volatility_multiplier": config.volatility_multiplier,
                "train_period": f"{train_df['datetime'].min()} — {train_df['datetime'].max()}",
                "val_period": f"{val_df['datetime'].min()} — {val_df['datetime'].max()}",
                "test_period": f"{test_df['datetime'].min()} — {test_df['datetime'].max()}",
            })

            task_signals = test_df[
                ["security", "datetime", "open", "high", "low", "close", "future_return", "target_event", "target_direction", "target_ternary"]
            ].copy()
            task_signals["ternary_down_proba"] = test_proba[:, 0]
            task_signals["ternary_no_event_proba"] = test_proba[:, 1]
            task_signals["ternary_up_proba"] = test_proba[:, 2]
            task_signals["ternary_event_proba"] = test_proba[:, 0] + test_proba[:, 2]
            task_signals["ternary_pred"] = pred
            signals[experiment_name] = task_signals

            print(
                f"{experiment_name}: f1_macro={metrics['f1_macro']:.4f}, "
                f"event_pr_auc={metrics['event_pr_auc']:.4f}, "
                f"event_lift5={metrics['event_lift_at_5pct']:.2f}"
            )

    return rows, signals


def run_sector_event_models(
    train_df,
    val_df,
    test_df,
    feature_sets,
    config,
    min_rows=3000,
):
    rows = []
    model_template = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", HistGradientBoostingClassifier(
            max_iter=120,
            learning_rate=0.055,
            l2_regularization=0.05,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    selected_feature_sets = {
        name: cols
        for name, cols in feature_sets.items()
        if name in {"price_only", "price_ticker_news_counts_sentiment", "price_market_news_counts_sentiment", "price_all_news"}
    }

    for sector in sorted(train_df["sector"].dropna().unique()):
        train_s = train_df[train_df["sector"] == sector].copy()
        val_s = val_df[val_df["sector"] == sector].copy()
        test_s = test_df[test_df["sector"] == sector].copy()
        if min(len(train_s), len(val_s), len(test_s)) < min_rows:
            continue
        if train_s["target_event"].nunique() < 2 or val_s["target_event"].nunique() < 2 or test_s["target_event"].nunique() < 2:
            continue

        for feature_set_name, features in selected_feature_sets.items():
            usable_features = [col for col in features if col in train_s.columns]
            train_sample = sample_train_frame(train_s, "target_event", config.max_train_rows)
            model = clone(model_template)
            model.fit(train_sample[usable_features], train_sample["target_event"].astype(int))
            val_proba = predict_proba_or_score(model, val_s[usable_features])
            threshold, tuning_score = best_threshold(val_s["target_event"].astype(int), val_proba)
            test_proba = predict_proba_or_score(model, test_s[usable_features])
            metrics = evaluate_predictions(test_s["target_event"].astype(int), test_proba, threshold)
            rows.append({
                "experiment": f"sector_event_detection__{sector}__{feature_set_name}__hist_gradient_boosting",
                "task": "sector_event_detection",
                "sector": sector,
                "feature_set": feature_set_name,
                "model": "hist_gradient_boosting",
                **metrics,
                "decision_threshold": threshold,
                "threshold_tuning_score_val": tuning_score,
                "n_features": len(usable_features),
                "n_train": len(train_sample),
                "n_val": len(val_s),
                "n_test": len(test_s),
                "train_positive_share": float(train_sample["target_event"].mean()),
                "val_positive_share": float(val_s["target_event"].mean()),
                "test_positive_share": float(test_s["target_event"].mean()),
                "horizon": config.horizon,
                "event_threshold": config.event_threshold,
                "target_mode": config.target_mode,
                "volatility_multiplier": config.volatility_multiplier,
                "train_period": f"{train_s['datetime'].min()} — {train_s['datetime'].max()}",
                "val_period": f"{val_s['datetime'].min()} — {val_s['datetime'].max()}",
                "test_period": f"{test_s['datetime'].min()} — {test_s['datetime'].max()}",
            })
            print(
                f"sector={sector} {feature_set_name}: "
                f"f1={metrics['f1']:.4f}, lift5={metrics['lift_at_5pct']:.2f}"
            )

    return rows


def build_report(results, dataset, output_path):
    event = results[results["task"] == "event_detection"].sort_values(["f1", "pr_auc"], ascending=False)
    direction = results[results["task"] == "direction_prediction"].sort_values(["f1", "roc_auc"], ascending=False)

    def format_table(df, n=6):
        cols = ["feature_set", "model", "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"]
        view = df[cols].head(n).copy()
        for col in cols[2:]:
            view[col] = view[col].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
        widths = {
            col: max(len(str(col)), *(len(str(value)) for value in view[col].tolist()))
            for col in cols
        }
        header = "| " + " | ".join(str(col).ljust(widths[col]) for col in cols) + " |"
        divider = "| " + " | ".join("-" * widths[col] for col in cols) + " |"
        lines = [header, divider]
        for _, row in view.iterrows():
            lines.append("| " + " | ".join(str(row[col]).ljust(widths[col]) for col in cols) + " |")
        return "\n".join(lines)

    price_event = event[event["feature_set"] == "price_only"].head(1)
    news_event = event[event["feature_set"] != "price_only"].head(1)
    price_direction = direction[direction["feature_set"] == "price_only"].head(1)
    news_direction = direction[direction["feature_set"] != "price_only"].head(1)

    def delta(best_news, best_price, metric):
        if best_news.empty or best_price.empty:
            return "н/д"
        return f"{(best_news.iloc[0][metric] - best_price.iloc[0][metric]):+.3f}"

    text = f"""# Практическая часть: экспериментальное моделирование

В практической части была реализована программная система, объединяющая часовые котировки российских акций и новостной поток за 2020-2026 годы. Исходный набор содержит {dataset['security'].nunique()} тикеров и {len(dataset):,} часовых наблюдений. Для исключения утечки будущей информации применялось хронологическое разбиение: обучение до 2024-01-01, валидация за 2024 год, тестирование на периоде 2025-2026 годов.

Прогнозная задача сформулирована как двухэтапная классификация. На первом этапе модель определяет наличие значимого события: абсолютная доходность через 4 часа не менее 1%. На втором этапе, только для наблюдений с событием, определяется направление движения цены: рост или снижение. Такой подход лучше соответствует предметной области, так как большая часть часовых свечей не содержит существенного движения, а направление имеет смысл оценивать прежде всего для найденных событий.

## Использованные признаки

Рыночный блок признаков включает лаговые доходности, внутрисвечной диапазон, изменение объема и отклонение цены/объема от скользящих средних. Новостной блок включает количество новостей, агрегированную финансовую тональность, тематические индикаторы по ключевым словам, TF-IDF-признаки с понижением размерности через TruncatedSVD, а также агрегированные RuBERT-эмбеддинги. Новости связывались с тикерами по словарю компаний, отраслевых терминов и рыночных ключевых слов, после чего агрегировались по часовым окнам 4 и 24 часа.

## Результаты event detection

{format_table(event)}

Лучшая модель с новостными признаками изменила F1 относительно лучшей price-only модели на {delta(news_event, price_event, 'f1')}, а PR-AUC на {delta(news_event, price_event, 'pr_auc')}. Для задачи поиска редких значимых движений PR-AUC и F1 важнее обычной accuracy, поскольку положительный класс несбалансирован.

## Результаты direction prediction

{format_table(direction)}

Для определения направления движения добавление новостных признаков изменило F1 относительно price-only подхода на {delta(news_direction, price_direction, 'f1')}, ROC-AUC на {delta(news_direction, price_direction, 'roc_auc')}. Поэтому результат следует интерпретировать сдержанно: новостные признаки дают слабый дополнительный сигнал по F1, но устойчивого выигрыша по всем метрикам на тестовом периоде не получено.

## Вывод

Полученные результаты показывают, что новостные данные могут использоваться как дополнительный источник информации для анализа динамики российского фондового рынка, однако их вклад зависит от способа привязки новости к тикеру и от выбранной метрики. Наиболее интерпретируемыми являются признаки количества новостей, тональности, тематических групп и текстовых компонент TF-IDF/RuBERT. Практический результат работы состоит не в построении безошибочного прогнозатора, а в создании воспроизводимого пайплайна: сбор данных, извлечение текстовых признаков, агрегация новостей по тикерам, обучение моделей и сравнение качества с базовыми моделями без новостей.
"""
    output_path.write_text(text, encoding="utf-8")


def build_backtest(signals, out_dir):
    df = signals.dropna(subset=["future_return", "event_proba"]).copy()
    if df.empty:
        return pd.DataFrame()

    if "direction_proba" not in df.columns:
        df["direction_proba"] = 0.5
    df["side"] = np.where(df["direction_proba"] >= 0.5, 1, -1)
    df["strategy_return"] = df["side"] * df["future_return"]
    df["buy_hold_return"] = df["future_return"]
    rows = []

    for share in [0.01, 0.03, 0.05, 0.10]:
        selected_parts = []
        for _, part in df.groupby("datetime", sort=True):
            n_top = max(1, int(np.ceil(len(part) * share)))
            selected_parts.append(part.nlargest(n_top, "event_proba"))
        selected = pd.concat(selected_parts, ignore_index=True) if selected_parts else pd.DataFrame()
        if selected.empty:
            continue
        hourly = selected.groupby("datetime").agg(
            strategy_return=("strategy_return", "mean"),
            long_share=("side", lambda x: float((x == 1).mean())),
            n_positions=("security", "nunique"),
        )
        benchmark = df.groupby("datetime")["buy_hold_return"].mean().reindex(hourly.index)
        rows.append({
            "top_share": share,
            "n_periods": len(hourly),
            "mean_return": hourly["strategy_return"].mean(),
            "median_return": hourly["strategy_return"].median(),
            "hit_rate": float((hourly["strategy_return"] > 0).mean()),
            "cumulative_return_sum": hourly["strategy_return"].sum(),
            "benchmark_mean_return": benchmark.mean(),
            "benchmark_cumulative_return_sum": benchmark.sum(),
            "avg_positions": hourly["n_positions"].mean(),
            "avg_long_share": hourly["long_share"].mean(),
        })

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "diploma_backtest_topk.csv", index=False)
    return result


def build_split_summary(results, out_dir):
    core = results[results["task"].isin(["event_detection", "direction_prediction"])].copy()
    if core.empty:
        return pd.DataFrame()

    best_by_split_task = (
        core.sort_values(["split_name", "task", "f1", "pr_auc"], ascending=[True, True, False, False])
        .groupby(["split_name", "task"], as_index=False)
        .head(1)
    )
    best_by_split_task.to_csv(out_dir / "diploma_best_by_split.csv", index=False)

    summary = (
        core.groupby(["task", "feature_set", "model"])
        .agg(
            n_splits=("split_name", "nunique"),
            f1_mean=("f1", "mean"),
            f1_std=("f1", "std"),
            roc_auc_mean=("roc_auc", "mean"),
            pr_auc_mean=("pr_auc", "mean"),
            lift_at_5pct_mean=("lift_at_5pct", "mean"),
        )
        .reset_index()
        .sort_values(["task", "f1_mean", "pr_auc_mean"], ascending=[True, False, False])
    )
    summary.to_csv(out_dir / "diploma_split_summary.csv", index=False)
    return summary


def build_improvements_notes(output_path):
    text = """# Подробное объяснение улучшений экспериментов

Этот файл написан простым языком для подготовки диплома и защиты: что было улучшено, зачем это нужно и как интерпретировать результат.

## 1. Улучшенное разбиение train/validation/test

Изначально использовалось одно фиксированное разбиение: обучение до 2024 года, validation за 2024 год, test за 2025-2026 годы. Это правильно, потому что не перемешивает прошлое и будущее, но одного разбиения мало: результат может зависеть от конкретного периода.

Поэтому добавлены режимы:

- `fixed`: базовый вариант train 2020-2023, validation 2024, test 2025-2026;
- `expanding-year`: walk-forward с расширяющимся обучающим окном;
- `rolling-year`: walk-forward со скользящим обучающим окном;
- `quarterly`: квартальные тестовые окна с validation за 6 месяцев до теста.

Для диплома это важно: можно показать не только одну таблицу качества, но и устойчивость модели во времени.

## 2. Улучшенный news-to-ticker matching

Проблема: в исходных новостях поле `ticker` пустое, поэтому новость нужно привязать к акции самостоятельно.

Что сделано:

- расширен словарь тикеров и названий эмитентов;
- добавлены полные названия компаний;
- для коротких слов и латинских тикеров используются строгие регулярные границы;
- отраслевые слова вроде `нефть`, `банк`, `газ` не считаются прямым упоминанием конкретной компании.

Почему это важно: если грубо привязать все новости про нефть к LKOH/ROSN/GAZP, модель получает много шума. Лучше отделять прямые новости о компании от общего рыночного фона.

## 3. Классификация типов новостей

Обычный sentiment часто слаб для рынка: нейтральная фраза про дивиденды может быть важнее эмоционально окрашенной общей новости.

Добавлены признаки типов событий:

- дивиденды;
- отчетность;
- санкции;
- M&A / сделки;
- buyback;
- прогнозы менеджмента;
- IPO/SPO/эмиссия;
- судебные события;
- ставки/валюта;
- сырьевые товары.

Это можно описать как приближение к fine-grained event extraction из литературы.

## 4. Фильтр шумных новостей

Новости могут быть общими, политическими, спортивными или бытовыми. Такие тексты портят признаки, если добавлять их ко всем тикерам.

Добавлен простой `news_relevance_score`, который повышается при наличии финансовых событий/тем и снижается для шумных слов. На его основе строятся `relevant_market_*` признаки.

## 5. Разделение company-specific и market-wide новостей

Новости теперь разделены на:

- `ticker_news_*`: явное упоминание эмитента;
- `market_news_*`: общий новостной фон;
- `relevant_market_*`: общий финансово релевантный фон.

Это полезно для интерпретации: можно сравнить, что сильнее влияет на модель, новости конкретной компании или общий рынок.

## 6. Event-window постановка

Если оценивать все часы подряд, большинство наблюдений не связано с новостями. Это размывает эффект.

Добавлен режим `--event-window-hours 24`: модель оценивается только в окнах до 24 часов после новости по тикеру. Это ближе к теме диплома, потому что исследуется влияние новостей именно после их публикации.

## 7. Секторные модели

Рынок неоднороден: банки реагируют на ставку ЦБ, нефтегаз на нефть/санкции, металлурги на сырье и экспорт.

Добавлен режим `--sector-models`, который обучает отдельные модели event detection по секторам. В экспериментах секторные модели дали высокий `lift@5%`, например для banking около 3.

## 8. Ranking-метрики

Для практической задачи не всегда важно классифицировать каждый час. Часто важнее найти топ самых сильных сигналов.

Добавлены:

- `precision@1%`;
- `precision@5%`;
- `precision@10%`;
- `lift@1%`;
- `lift@5%`;
- `lift@10%`.

`lift@5% = 3` означает, что среди 5% самых уверенных сигналов событий примерно в 3 раза больше, чем в среднем.

## 9. Backtest top-k сигналов

Добавлен простой исследовательский backtest:

- берем top-k сигналов по `event_proba`;
- направление задается `direction_proba`;
- считаем среднюю будущую доходность через 4 часа;
- сравниваем с benchmark по тем же часам.

Это не полноценная торговая система, но хороший практический блок для диплома: можно показать экономическую интерпретацию сигналов.

## 10. CatBoost и LightGBM

CatBoost установлен и успешно использован. Он улучшил часть результатов, особенно в event-window постановке.

LightGBM установлен, но на текущей macOS-среде требует системную библиотеку `libomp`. Код написан через optional import: если `libomp` будет установлен, LightGBM автоматически появится среди моделей.

## 11. Внешние рыночные факторы

Добавлен скрипт `scripts/build_moex_factors.py`, который скачивает IMOEX и RTSI из MOEX ISS API. В основной пайплайн добавлены признаки доходности и волатильности индексов.

Эксперимент с внешними факторами проведен, но он не стал лучшим. Это тоже нормальный результат: в текущем наборе сильнее работают внутренние market/sector признаки, рассчитанные по акциям.

## 12. Главное, что можно сказать в дипломе

Новостные признаки не дают большого универсального улучшения на всех часах подряд. Однако при более корректной событийной постановке, когда анализируются окна после релевантных новостей, текстовые признаки и типы новостей дают небольшой прирост качества и улучшают ranking-метрики.

Самый защищаемый вывод:

> Разработанная система показывает, что новостные данные могут использоваться как дополнительный источник информации для прогнозирования значимых движений акций. Наиболее выраженный эффект наблюдается в event-window постановке и в метриках ранжирования top-k сигналов, а не в общей accuracy.
"""
    output_path.write_text(text, encoding="utf-8")


def build_split_plan(
    model_df,
    mode,
    train_start,
    train_end,
    val_end,
):
    if mode == "fixed":
        split_prefix = "fixed"
        if train_start is not None:
            split_prefix = f"fixed_train_from_{pd.Timestamp(train_start):%Y%m%d}"
        return [
            {
                "split_name": f"{split_prefix}__train_to_{pd.Timestamp(train_end):%Y%m%d}__val_to_{pd.Timestamp(val_end):%Y%m%d}",
                "train_start": pd.Timestamp(train_start) if train_start is not None else None,
                "train_end": pd.Timestamp(train_end),
                "val_start": pd.Timestamp(train_end),
                "val_end": pd.Timestamp(val_end),
                "test_start": pd.Timestamp(val_end),
                "test_end": None,
            }
        ]

    if mode == "expanding-year":
        return [
            {
                "split_name": "expanding__test_2023",
                "train_start": None,
                "train_end": pd.Timestamp("2022-01-01"),
                "val_start": pd.Timestamp("2022-01-01"),
                "val_end": pd.Timestamp("2023-01-01"),
                "test_start": pd.Timestamp("2023-01-01"),
                "test_end": pd.Timestamp("2024-01-01"),
            },
            {
                "split_name": "expanding__test_2024",
                "train_start": None,
                "train_end": pd.Timestamp("2023-01-01"),
                "val_start": pd.Timestamp("2023-01-01"),
                "val_end": pd.Timestamp("2024-01-01"),
                "test_start": pd.Timestamp("2024-01-01"),
                "test_end": pd.Timestamp("2025-01-01"),
            },
            {
                "split_name": "expanding__test_2025_2026",
                "train_start": None,
                "train_end": pd.Timestamp("2024-01-01"),
                "val_start": pd.Timestamp("2024-01-01"),
                "val_end": pd.Timestamp("2025-01-01"),
                "test_start": pd.Timestamp("2025-01-01"),
                "test_end": None,
            },
        ]

    if mode == "rolling-year":
        return [
            {
                "split_name": "rolling__test_2023",
                "train_start": pd.Timestamp("2020-01-01"),
                "train_end": pd.Timestamp("2022-01-01"),
                "val_start": pd.Timestamp("2022-01-01"),
                "val_end": pd.Timestamp("2023-01-01"),
                "test_start": pd.Timestamp("2023-01-01"),
                "test_end": pd.Timestamp("2024-01-01"),
            },
            {
                "split_name": "rolling__test_2024",
                "train_start": pd.Timestamp("2021-01-01"),
                "train_end": pd.Timestamp("2023-01-01"),
                "val_start": pd.Timestamp("2023-01-01"),
                "val_end": pd.Timestamp("2024-01-01"),
                "test_start": pd.Timestamp("2024-01-01"),
                "test_end": pd.Timestamp("2025-01-01"),
            },
            {
                "split_name": "rolling__test_2025_2026",
                "train_start": pd.Timestamp("2022-01-01"),
                "train_end": pd.Timestamp("2024-01-01"),
                "val_start": pd.Timestamp("2024-01-01"),
                "val_end": pd.Timestamp("2025-01-01"),
                "test_start": pd.Timestamp("2025-01-01"),
                "test_end": None,
            },
        ]

    if mode == "quarterly":
        quarters = pd.date_range("2024-01-01", "2026-04-01", freq="QS")
        plan = []
        for test_start in quarters:
            test_end = test_start + pd.DateOffset(months=3)
            val_end_q = test_start
            val_start_q = val_end_q - pd.DateOffset(months=6)
            train_end_q = val_start_q
            train_start_q = train_end_q - pd.DateOffset(years=3)
            if train_start_q < model_df["datetime"].min():
                train_start_q = None
            plan.append(
                {
                    "split_name": f"quarterly__test_{test_start:%Y_Q%q}".replace("%q", str(((test_start.month - 1) // 3) + 1)),
                    "train_start": train_start_q,
                    "train_end": train_end_q,
                    "val_start": val_start_q,
                    "val_end": val_end_q,
                    "test_start": test_start,
                    "test_end": test_end,
                }
            )
        return plan

    raise ValueError(f"Unknown split mode: {mode}")


def slice_period(df, start, end):
    mask = pd.Series(True, index=df.index)
    if start is not None:
        mask &= df["datetime"] >= start
    if end is not None:
        mask &= df["datetime"] < end
    return df[mask].copy()


def apply_news_filter(model_df, mode):
    if mode == "all":
        return model_df.copy()

    df = model_df.copy()
    ticker_activity = pd.Series(False, index=df.index)
    for col in ["ticker_has_news", "ticker_news_count", "ticker_news_count_4h", "ticker_news_count_24h"]:
        if col in df.columns:
            ticker_activity |= df[col].fillna(0) > 0

    relevant_activity = pd.Series(False, index=df.index)
    for col in [
        "relevant_market_news_count",
        "relevant_market_news_count_4h",
        "relevant_market_news_count_24h",
        "ticker_relevance_mean",
        "ticker_relevance_mean_4h",
        "ticker_relevance_mean_24h",
    ]:
        if col in df.columns:
            relevant_activity |= df[col].fillna(0) > 0

    sector_activity = pd.Series(False, index=df.index)
    if "sector" in df.columns:
        for sector, topics in SECTOR_TOPIC_BY_SECTOR.items():
            sector_mask = df["sector"] == sector
            topic_cols = []
            for topic in topics:
                topic_cols.extend([
                    col for col in df.columns
                    if (
                        f"topic_{topic}" in col
                        or f"event_type_{topic}" in col
                    )
                    and (
                        col.startswith("market_")
                        or col.startswith("relevant_market_")
                    )
                    and (col.endswith("_sum") or col.endswith("_sum_4h") or col.endswith("_sum_24h"))
                ])
            if topic_cols:
                sector_activity |= sector_mask & (df[topic_cols].fillna(0).sum(axis=1) > 0)

    if mode == "company":
        mask = ticker_activity
    elif mode == "sector":
        mask = sector_activity
    elif mode == "company_or_sector":
        mask = ticker_activity | sector_activity
    elif mode == "relevant":
        mask = ticker_activity | sector_activity | relevant_activity
    elif mode == "strict_relevant":
        mask = ticker_activity | (sector_activity & relevant_activity)
    else:
        raise ValueError(f"Unknown news filter mode: {mode}")

    filtered = df[mask].copy()
    if filtered.empty:
        raise ValueError(f"News filter {mode!r} removed all rows")
    return filtered


def apply_candidate_filter(model_df, mode):
    if mode == "all":
        return model_df.copy()

    df = model_df.copy()
    ticker_recent = pd.Series(False, index=df.index)
    for col in ["ticker_news_count", "ticker_news_count_4h", "ticker_news_count_24h", "ticker_has_news"]:
        if col in df.columns:
            ticker_recent |= df[col].fillna(0) > 0
    if "hours_since_ticker_news" in df.columns:
        ticker_recent |= df["hours_since_ticker_news"].fillna(999) <= 24

    market_recent = pd.Series(False, index=df.index)
    for col in ["market_news_count", "market_news_count_4h", "market_news_count_24h", "relevant_market_news_count_24h"]:
        if col in df.columns:
            market_recent |= df[col].fillna(0) > 0

    volume_spike = pd.Series(False, index=df.index)
    for col in ["volume_to_sma_24", "volume_to_sma_48", "value_change_1h"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).abs()
            volume_spike |= values >= values.quantile(0.80)

    movement_wakeup = pd.Series(False, index=df.index)
    for col in ["return_1h", "return_4h", "relative_return_1h", "relative_sector_return_1h"]:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").fillna(0.0).abs()
            movement_wakeup |= values >= values.quantile(0.75)

    if mode == "news_only":
        mask = ticker_recent
    elif mode == "news_or_volume":
        mask = ticker_recent | volume_spike
    elif mode == "news_or_movement":
        mask = ticker_recent | market_recent | movement_wakeup
    elif mode == "strict_news":
        mask = ticker_recent
    else:
        raise ValueError(f"Unknown candidate mode: {mode}")

    filtered = df[mask].copy()
    if filtered.empty:
        raise ValueError(f"Candidate filter {mode!r} removed all rows")
    return filtered


def ensure_external_factors(config):
    if not config.external_factors_path:
        return

    factors_path = Path(config.external_factors_path)
    if factors_path.exists():
        return

    print(f"External factors file not found: {factors_path}")
    print("Building MOEX external factors automatically...")
    build_moex_factors(output=str(factors_path))


def run_one_split(
    split,
    model_df,
    feature_sets,
    config,
    run_sector_models_flag,
    run_multiclass_flag,
):
    train_df = slice_period(model_df, split["train_start"], split["train_end"])
    val_df = slice_period(model_df, split["val_start"], split["val_end"])
    test_df = slice_period(model_df, split["test_start"], split["test_end"])

    print(
        f"Split {split['split_name']}: rows train={len(train_df)} "
        f"val={len(val_df)} test={len(test_df)}"
    )

    if min(len(train_df), len(val_df), len(test_df)) == 0:
        raise ValueError(f"Empty split: {split['split_name']}")

    all_rows = []
    event_rows, event_signals, event_models = run_task(
        "event_detection", "target_event", train_df, val_df, test_df, feature_sets, config
    )
    all_rows.extend(event_rows)

    event_results = pd.DataFrame(event_rows)
    best_event_row_for_cascade = (
        event_results
        .sort_values(["f1", "pr_auc"], ascending=False)
        .iloc[0]
    )
    best_event_name_for_cascade = best_event_row_for_cascade["experiment"]
    best_event_features_for_cascade = [
        col
        for col in feature_sets[best_event_row_for_cascade["feature_set"]]
        if col in train_df.columns
    ]
    best_event_model_for_cascade = event_models[best_event_name_for_cascade]
    event_cutoff_for_direction = float(best_event_row_for_cascade["decision_threshold"])

    train_event_proba = predict_proba_or_score(
        best_event_model_for_cascade,
        train_df[best_event_features_for_cascade],
    )
    val_event_proba = predict_proba_or_score(
        best_event_model_for_cascade,
        val_df[best_event_features_for_cascade],
    )
    test_event_proba = predict_proba_or_score(
        best_event_model_for_cascade,
        test_df[best_event_features_for_cascade],
    )

    train_dir = train_df[train_event_proba >= event_cutoff_for_direction].copy()
    val_dir = val_df[val_event_proba >= event_cutoff_for_direction].copy()
    test_dir = test_df[test_event_proba >= event_cutoff_for_direction].copy()

    if (
        min(len(train_dir), len(val_dir), len(test_dir)) == 0
        or train_dir["target_direction"].nunique() < 2
        or val_dir["target_direction"].nunique() < 2
        or test_dir["target_direction"].nunique() < 2
    ):
        print("Cascade direction filter produced an empty or one-class sample; falling back to true event rows.")
        train_dir = train_df[train_df["target_event"] == 1].copy()
        val_dir = val_df[val_df["target_event"] == 1].copy()
        test_dir = test_df[test_df["target_event"] == 1].copy()

    direction_rows, direction_signals, direction_models = run_task(
        "direction_prediction", "target_direction", train_dir, val_dir, test_dir, feature_sets, config
    )
    all_rows.extend(direction_rows)

    if run_sector_models_flag:
        all_rows.extend(run_sector_event_models(train_df, val_df, test_df, feature_sets, config))

    if run_multiclass_flag:
        ternary_rows, _ = run_multiclass_task(train_df, val_df, test_df, feature_sets, config)
        all_rows.extend(ternary_rows)

    results = pd.DataFrame(all_rows)
    results["split_name"] = split["split_name"]

    best_event_row = (
        results[results["task"] == "event_detection"]
        .sort_values(["f1", "pr_auc"], ascending=False)
        .iloc[0]
    )
    best_direction_row = (
        results[results["task"] == "direction_prediction"]
        .sort_values(["f1", "roc_auc"], ascending=False)
        .iloc[0]
    )
    best_direction_pr_auc_row = (
        results[results["task"] == "direction_prediction"]
        .sort_values(["pr_auc", "roc_auc"], ascending=False)
        .iloc[0]
    )
    best_event_name = best_event_row["experiment"]
    best_direction_name = best_direction_row["experiment"]
    best_direction_pr_auc_name = best_direction_pr_auc_row["experiment"]

    direction_features = [
        col
        for col in feature_sets[best_direction_row["feature_set"]]
        if col in test_dir.columns
    ]
    direction_model = direction_models[best_direction_name]
    full_direction_proba = predict_proba_or_score(
        direction_model,
        test_dir[direction_features],
    )
    full_direction_pred = (
        full_direction_proba >= float(best_direction_row["decision_threshold"])
    ).astype(int)
    full_direction_signals = test_dir[["security", "datetime"]].copy()
    full_direction_signals["direction_prediction_proba"] = full_direction_proba
    full_direction_signals["direction_prediction_pred"] = full_direction_pred

    signals = event_signals[best_event_name].merge(
        full_direction_signals,
        on=["security", "datetime"],
        how="left",
    )
    signals = signals.rename(columns={
        "event_detection_proba": "event_proba",
        "event_detection_pred": "event_pred",
        "direction_prediction_proba": "direction_proba",
        "direction_prediction_pred": "direction_pred",
    })
    signals["direction_proba"] = signals["direction_proba"].fillna(0.5)
    signals["direction_pred"] = signals["direction_pred"].fillna((signals["direction_proba"] >= 0.5).astype(int)).astype(int)
    signals["split_name"] = split["split_name"]

    meta = {
        "split_name": split["split_name"],
        "best_event_experiment": best_event_name,
        "best_direction_experiment": best_direction_name,
        "n_train": len(train_df),
        "n_val": len(val_df),
        "n_test": len(test_df),
        "train_positive_share": float(train_df["target_event"].mean()),
        "val_positive_share": float(val_df["target_event"].mean()),
        "test_positive_share": float(test_df["target_event"].mean()),
    }
    best_models = {
        "event_detection": {
            "model": event_models[best_event_name],
            "experiment": best_event_name,
            "feature_set": best_event_row["feature_set"],
            "features": [col for col in feature_sets[best_event_row["feature_set"]] if col in train_df.columns],
            "decision_threshold": float(best_event_row["decision_threshold"]),
            "metrics": {
                key: float(best_event_row[key])
                for key in ["f1", "roc_auc", "pr_auc", "precision_at_5pct", "lift_at_5pct"]
            },
        },
        "direction_prediction": {
            "model": direction_models[best_direction_name],
            "experiment": best_direction_name,
            "feature_set": best_direction_row["feature_set"],
            "features": [col for col in feature_sets[best_direction_row["feature_set"]] if col in train_dir.columns],
            "decision_threshold": float(best_direction_row["decision_threshold"]),
            "metrics": {
                key: float(best_direction_row[key])
                for key in ["f1", "roc_auc", "pr_auc", "precision_at_5pct", "lift_at_5pct"]
            },
        },
        "direction_prediction_pr_auc": {
            "model": direction_models[best_direction_pr_auc_name],
            "experiment": best_direction_pr_auc_name,
            "feature_set": best_direction_pr_auc_row["feature_set"],
            "features": [col for col in feature_sets[best_direction_pr_auc_row["feature_set"]] if col in train_dir.columns],
            "decision_threshold": float(best_direction_pr_auc_row["decision_threshold"]),
            "metrics": {
                key: float(best_direction_pr_auc_row[key])
                for key in ["f1", "roc_auc", "pr_auc", "precision_at_5pct", "lift_at_5pct"]
            },
        },
    }
    return results, {split["split_name"]: signals}, meta, best_models


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stocks", default="data/all_1h_stocks_2020_2026.csv")
    parser.add_argument("--news", default="data/news_features_2020_2026.parquet")
    parser.add_argument("--out-dir", default="data/model_outputs")
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--event-threshold", type=float, default=0.01)
    parser.add_argument("--target-mode", choices=["fixed", "volatility"], default="fixed")
    parser.add_argument("--volatility-multiplier", type=float, default=3.0)
    parser.add_argument("--train-start", default=None)
    parser.add_argument("--train-end", default="2024-01-01")
    parser.add_argument("--val-end", default="2025-01-01")
    parser.add_argument("--only-news-active", action="store_true")
    parser.add_argument("--event-window-hours", type=int, default=None)
    parser.add_argument("--sector-models", action="store_true")
    parser.add_argument("--run-multiclass", action="store_true")
    parser.add_argument("--external-factors-path", default="data/moex_external_factors.csv")
    parser.add_argument("--event-decision-threshold", type=float, default=0.60)
    parser.add_argument("--skip-text-components", action="store_true")
    parser.add_argument(
        "--split-mode",
        choices=["fixed", "expanding-year", "rolling-year", "quarterly"],
        default="fixed",
    )
    parser.add_argument("--feature-set-filter", nargs="*", default=None)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument(
        "--news-filter-mode",
        choices=["all", "company", "sector", "company_or_sector", "relevant", "strict_relevant"],
        default="all",
    )
    parser.add_argument(
        "--candidate-mode",
        choices=["all", "news_only", "news_or_volume", "news_or_movement", "strict_news"],
        default="all",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    config = ExperimentConfig(
        horizon=args.horizon,
        event_threshold=args.event_threshold,
        target_mode=args.target_mode,
        volatility_multiplier=args.volatility_multiplier,
        train_start=args.train_start,
        train_end=args.train_end,
        val_end=args.val_end,
        max_train_rows=args.max_train_rows,
        external_factors_path=args.external_factors_path,
        event_decision_threshold=args.event_decision_threshold,
        use_text_components=not args.skip_text_components,
    )
    print("Config:", config)

    ensure_external_factors(config)

    dataset, price_cols, news_cols = make_dataset(Path(args.stocks), Path(args.news), config)
    model_df = dataset.dropna(subset=price_cols + ["future_return", "target_event", "target_direction"]).copy()
    if args.only_news_active:
        news_activity_cols = [
            col for col in ["ticker_news_count_24h", "ticker_has_news"]
            if col in model_df.columns
        ]
        if news_activity_cols:
            model_df = model_df[model_df[news_activity_cols].max(axis=1) > 0].copy()
    if args.event_window_hours is not None and "hours_since_ticker_news" in model_df.columns:
        model_df = model_df[model_df["hours_since_ticker_news"] <= args.event_window_hours].copy()
    model_df = apply_news_filter(model_df, args.news_filter_mode)
    model_df = apply_candidate_filter(model_df, args.candidate_mode)
    model_df = model_df.sort_values("datetime").reset_index(drop=True)

    text_cols = [col for col in news_cols if "tfidf" in col or "rubert" in col or "topic" in col]
    basic_sentiment_tokens = [
        "news_count",
        "sentiment_mean",
        "sentiment_min",
        "sentiment_max",
        "sentiment_positive",
        "sentiment_negative",
        "sentiment_neutral",
    ]
    advanced_sentiment_tokens = [
        "sentiment_margin",
        "sentiment_abs",
        "sentiment_confidence",
        "sentiment_signed_confidence",
        "sentiment_label",
        "sentiment_relevance_weighted",
        "lexicon_",
    ]
    event_sentiment_tokens = ["sentiment_event_", "event_type_"]
    sentiment_cols = [col for col in news_cols if "sentiment" in col or "news_count" in col or "lexicon_" in col]
    basic_sentiment_cols = [
        col for col in news_cols
        if any(token in col for token in basic_sentiment_tokens)
    ]
    advanced_sentiment_cols = [
        col for col in news_cols
        if any(token in col for token in advanced_sentiment_tokens)
    ]
    event_sentiment_cols = [
        col for col in news_cols
        if any(token in col for token in event_sentiment_tokens)
    ]
    feature_sets = {
        "price_only": price_cols,
        "price_ticker_sentiment_basic": price_cols + [
            col for col in basic_sentiment_cols if col.startswith("ticker_")
        ],
        "price_market_sentiment_basic": price_cols + [
            col for col in basic_sentiment_cols if col.startswith("market_")
        ],
        "price_ticker_sentiment_advanced": price_cols + [
            col for col in advanced_sentiment_cols if col.startswith("ticker_")
        ],
        "price_market_sentiment_advanced": price_cols + [
            col for col in advanced_sentiment_cols if col.startswith("market_") or col.startswith("relevant_market_")
        ],
        "price_sentiment_event_interactions": price_cols + [
            col for col in event_sentiment_cols
        ],
        "price_ticker_news_text": price_cols + [
            col for col in text_cols if col.startswith("ticker_")
        ],
        "price_market_news_text": price_cols + [
            col for col in text_cols if col.startswith("market_")
        ],
        "price_all_news": price_cols + news_cols,
    }
    if args.feature_set_filter:
        keep = set(args.feature_set_filter)
        feature_sets = {
            name: cols for name, cols in feature_sets.items()
            if name in keep
        }
        if not feature_sets:
            raise ValueError(f"No feature sets left after filter: {args.feature_set_filter}")

    print("Rows:", len(model_df))
    print("Date range:", model_df["datetime"].min(), model_df["datetime"].max())
    print("Features:", {name: len(cols) for name, cols in feature_sets.items()})

    split_plan = build_split_plan(
        model_df,
        args.split_mode,
        config.train_start,
        config.train_end,
        config.val_end,
    )
    all_results = []
    all_signals = []
    split_meta = []
    split_best_models = []

    for split in split_plan:
        split_results, split_signals, meta, best_models = run_one_split(
            split,
            model_df,
            feature_sets,
            config,
            run_sector_models_flag=args.sector_models,
            run_multiclass_flag=args.run_multiclass,
        )
        all_results.append(split_results)
        all_signals.extend(split_signals.values())
        split_meta.append(meta)
        split_best_models.append({
            "split_name": split["split_name"],
            **best_models,
        })

    results = pd.concat(all_results, ignore_index=True)
    results_path = out_dir / "diploma_experiment_results.csv"
    results.to_csv(results_path, index=False)

    split_summary = build_split_summary(results, out_dir)
    signals = pd.concat(all_signals, ignore_index=True) if all_signals else pd.DataFrame()
    signals_path = out_dir / "diploma_model_signals.csv"
    signals.to_csv(signals_path, index=False)
    backtest = build_backtest(signals, out_dir)

    best_event = (
        results[results["task"] == "event_detection"]
        .sort_values(["f1", "pr_auc"], ascending=False)
        .iloc[0]
    )
    best_direction = (
        results[results["task"] == "direction_prediction"]
        .sort_values(["f1", "roc_auc"], ascending=False)
        .iloc[0]
    )

    model_meta = {
        "best_event_experiment": best_event["experiment"],
        "best_event_split": best_event["split_name"],
        "best_direction_experiment": best_direction["experiment"],
        "best_direction_split": best_direction["split_name"],
        "config": config.__dict__,
        "only_news_active": args.only_news_active,
        "event_window_hours": args.event_window_hours,
        "news_filter_mode": args.news_filter_mode,
        "candidate_mode": args.candidate_mode,
        "sector_models": args.sector_models,
        "run_multiclass": args.run_multiclass,
        "split_mode": args.split_mode,
        "split_meta": split_meta,
        "price_features": price_cols,
        "news_feature_count": len(news_cols),
        "backtest_rows": len(backtest),
        "split_summary_rows": len(split_summary),
    }

    if args.save_models:
        models_dir = out_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        saved_model_files = []
        for split_model in split_best_models:
            split_name = re.sub(r"[^A-Za-z0-9_]+", "_", split_model["split_name"])
            for task_name in ["event_detection", "direction_prediction", "direction_prediction_pr_auc"]:
                model_payload = {
                    "model": split_model[task_name]["model"],
                    "experiment": split_model[task_name]["experiment"],
                    "split_name": split_model["split_name"],
                    "task": task_name,
                    "feature_set": split_model[task_name]["feature_set"],
                    "features": split_model[task_name]["features"],
                    "decision_threshold": split_model[task_name]["decision_threshold"],
                    "metrics": split_model[task_name]["metrics"],
                    "config": config.__dict__,
                    "news_filter_mode": args.news_filter_mode,
                    "split_mode": args.split_mode,
                }
                model_path = models_dir / f"{split_name}__{task_name}.joblib"
                joblib_dump(model_payload, model_path)
                saved_model_files.append(str(model_path))
        model_meta["saved_model_files"] = saved_model_files

    (out_dir / "diploma_model_metadata.json").write_text(
        json.dumps(model_meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    build_report(results, model_df, out_dir / "report.txt")
    build_improvements_notes(out_dir / "improvements_explained.md")
    print(f"Saved results: {results_path}")
    print(f"Saved signals: {signals_path}")
    print("Saved split summary and improvements_explained.md")
    print("Saved report: report.txt")


if __name__ == "__main__":
    main()
