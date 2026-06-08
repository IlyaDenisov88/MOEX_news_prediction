import argparse
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests


MOEX_INDEX_URL = "https://iss.moex.com/iss/engines/stock/markets/index/securities/{ticker}/candles.json"
MOEX_COLUMNS = ["open", "close", "high", "low", "value", "volume", "begin", "end"]


def fetch_index(ticker, date_from, date_till, interval=60):
    rows = []
    start = 0
    while True:
        params = {
            "from": date_from,
            "till": date_till,
            "interval": interval,
            "start": start,
            "iss.meta": "off",
        }
        for attempt in range(4):
            try:
                response = requests.get(
                    MOEX_INDEX_URL.format(ticker=ticker),
                    params=params,
                    timeout=30,
                    headers={"Accept-Encoding": "identity"},
                )
                response.raise_for_status()
                break
            except requests.RequestException:
                if attempt == 3:
                    raise
                time.sleep(1 + attempt)
        data = response.json().get("candles", {}).get("data", [])
        if not data:
            break
        rows.extend(data)
        start += 500
        time.sleep(0.1)

    df = pd.DataFrame(rows, columns=MOEX_COLUMNS)
    if df.empty:
        return df

    df["hour"] = pd.to_datetime(df["begin"], errors="coerce").dt.floor("h")
    df = df.dropna(subset=["hour"]).drop_duplicates(subset=["hour"]).sort_values("hour")
    prefix = ticker.lower()
    df[f"{prefix}_close"] = pd.to_numeric(df["close"], errors="coerce")
    df[f"{prefix}_return_1h"] = df[f"{prefix}_close"].pct_change(1)
    df[f"{prefix}_return_4h"] = df[f"{prefix}_close"].pct_change(4)
    df[f"{prefix}_return_24h"] = df[f"{prefix}_close"].pct_change(24)
    df[f"{prefix}_volatility_24h"] = df[f"{prefix}_return_1h"].rolling(24, min_periods=4).std()
    return df[
        [
            "hour",
            f"{prefix}_close",
            f"{prefix}_return_1h",
            f"{prefix}_return_4h",
            f"{prefix}_return_24h",
            f"{prefix}_volatility_24h",
        ]
    ]


def build_moex_factors(date_from="2020-01-01", date_till=None, output="data/moex_external_factors.csv", tickers=None):
    if date_till is None:
        date_till = datetime.today().strftime("%Y-%m-%d")

    if tickers is None:
        tickers = ["IMOEX", "RTSI"]

    frames = []
    for ticker in tickers:
        print(f"Fetching {ticker}...")
        df = fetch_index(ticker, date_from, date_till)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No MOEX factors were loaded")

    result = frames[0]
    for frame in frames[1:]:
        result = result.merge(frame, on="hour", how="outer")

    result = result.sort_values("hour").reset_index(drop=True)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"Saved {len(result)} rows to {output}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date-from", default="2020-01-01")
    parser.add_argument("--date-till", default=None)
    parser.add_argument("--output", default="data/moex_external_factors.csv")
    parser.add_argument("--tickers", nargs="+", default=["IMOEX", "RTSI"])
    args = parser.parse_args()

    build_moex_factors(
        date_from=args.date_from,
        date_till=args.date_till,
        output=args.output,
        tickers=args.tickers,
    )


if __name__ == "__main__":
    main()
