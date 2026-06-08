import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import requests


MOEX_BASE_URL = "https://iss.moex.com/iss"

TIMEFRAME_TO_INTERVAL = {
    "1m": 1,
    "10m": 10,
    "1h": 60,
    "1d": 24,
    "1w": 7,
    "1mo": 31,
}

MOEX_CANDLE_COLUMNS = [
    "open",
    "close",
    "high",
    "low",
    "value",
    "volume",
    "begin",
    "end",
]


class MarketDataLoader:
    base_url = MOEX_BASE_URL
    candle_columns = MOEX_CANDLE_COLUMNS
    timeframe_to_interval = TIMEFRAME_TO_INTERVAL

    def __init__(
        self,
        board="tqbr",
        market="shares",
        engine="stock",
        timeout=10,
        sleep_seconds=0.2,
        chunk_size=500,
    ):
        self.board = board
        self.market = market
        self.engine = engine
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.chunk_size = chunk_size
        self.session = requests.Session()

    def validate_timeframe(self, timeframe):
        if timeframe not in self.timeframe_to_interval:
            available = ", ".join(self.timeframe_to_interval.keys())
            raise ValueError(f"Неизвестный timeframe: {timeframe}. Доступно: {available}")

        return self.timeframe_to_interval[timeframe]

    def build_candles_url(self, security):
        return (
            f"{self.base_url}/engines/{self.engine}/markets/{self.market}/"
            f"boards/{self.board}/securities/{security.upper()}/candles.json"
        )

    def fetch_candles_page(
        self,
        url,
        interval,
        date_from,
        date_till,
        start,
    ):
        params = {
            "interval": interval,
            "from": date_from,
            "till": date_till,
            "start": start,
            "iss.meta": "off",
        }

        response = self.session.get(url, params=params, timeout=self.timeout)
        response.raise_for_status()

        data = response.json()
        return data.get("candles", {}).get("data", [])

    def load_candles(
        self,
        security,
        date_from,
        date_till,
        timeframe="1h",
    ):
        interval = self.validate_timeframe(timeframe)
        url = self.build_candles_url(security=security)

        all_rows = []
        start = 0

        while True:
            rows = self.fetch_candles_page(
                url=url,
                interval=interval,
                date_from=date_from,
                date_till=date_till,
                start=start,
            )

            if not rows:
                break

            all_rows.extend(rows)
            start += self.chunk_size
            time.sleep(self.sleep_seconds)

        df = pd.DataFrame(all_rows, columns=self.candle_columns)

        if df.empty:
            return self._empty_candles_frame()

        df["security"] = security.upper()
        df["timeframe"] = timeframe
        df["datetime"] = pd.to_datetime(df["begin"], errors="coerce")

        df = df[
            [
                "security",
                "timeframe",
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "value",
            ]
        ]

        df = df.dropna(subset=["datetime"])
        df = df.sort_values("datetime").reset_index(drop=True)

        return df

    def update_stocks_file(
        self,
        tickers,
        output_path,
        timeframe="1h",
        date_from=None,
        date_till=None,
    ):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        if date_from is None:
            date_from = get_next_date_from_existing_file(output_path)

        if date_till is None:
            date_till = get_today()

        frames = []

        for ticker in tickers:
            try:
                df = self.load_candles(
                    security=ticker,
                    date_from=date_from,
                    date_till=date_till,
                    timeframe=timeframe,
                )

                if not df.empty:
                    frames.append(df)

            except Exception as error:
                print(f"[ERROR] {ticker}: {error}")

        if frames:
            new_df = pd.concat(frames, ignore_index=True)
        else:
            new_df = self._empty_candles_frame()

        if output_file.exists():
            old_df = pd.read_csv(output_file)
            full_df = pd.concat([old_df, new_df], ignore_index=True)
        else:
            full_df = new_df

        if not full_df.empty:
            full_df["datetime"] = pd.to_datetime(full_df["datetime"], errors="coerce")
            full_df = full_df.dropna(subset=["datetime"])
            full_df = full_df.drop_duplicates(subset=["security", "datetime"])
            full_df = full_df.sort_values(["security", "datetime"]).reset_index(drop=True)

        full_df.to_csv(output_file, index=False, encoding="utf-8-sig")

        return full_df

    @staticmethod
    def _empty_candles_frame():
        return pd.DataFrame(
            columns=[
                "security",
                "timeframe",
                "datetime",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "value",
            ]
        )


def get_next_date_from_existing_file(path):
    file_path = Path(path)

    if not file_path.exists():
        return "2020-01-01"

    df = pd.read_csv(file_path)

    if df.empty or "datetime" not in df.columns:
        return "2020-01-01"

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    max_date = df["datetime"].max()

    if pd.isna(max_date):
        return "2020-01-01"

    return (max_date.date() + timedelta(days=1)).strftime("%Y-%m-%d")


def get_today():
    return datetime.today().strftime("%Y-%m-%d")


def validate_timeframe(timeframe):
    return MarketDataLoader().validate_timeframe(timeframe)


def build_moex_candles_url(
    security,
    board="tqbr",
    market="shares",
    engine="stock",
):
    parser = MarketDataLoader(board=board, market=market, engine=engine)
    return parser.build_candles_url(security)


def fetch_moex_candles_page(
    url,
    interval,
    date_from,
    date_till,
    start,
    timeout=10,
):
    parser = MarketDataLoader(timeout=timeout)
    return parser.fetch_candles_page(
        url=url,
        interval=interval,
        date_from=date_from,
        date_till=date_till,
        start=start,
    )


def load_moex_candles(
    security,
    date_from,
    date_till,
    timeframe="1h",
    board="tqbr",
    sleep_seconds=0.2,
    chunk_size=500,
):
    parser = MarketDataLoader(
        board=board,
        sleep_seconds=sleep_seconds,
        chunk_size=chunk_size,
    )
    return parser.load_candles(
        security=security,
        date_from=date_from,
        date_till=date_till,
        timeframe=timeframe,
    )


def update_stocks_file(
    tickers,
    output_path,
    timeframe="1h",
    date_from=None,
    date_till=None,
):
    return MarketDataLoader().update_stocks_file(
        tickers=tickers,
        output_path=output_path,
        timeframe=timeframe,
        date_from=date_from,
        date_till=date_till,
    )


MOEXMarketDataParser = MarketDataLoader
