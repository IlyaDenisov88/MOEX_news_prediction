import random
import re
import time
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

NEWS_COLUMNS = [
    "source",
    "published_at",
    "date",
    "time",
    "title",
    "link",
    "ticker",
    "rubric",
]

RU_MONTHS = {
    "января": 1,
    "февраля": 2,
    "марта": 3,
    "апреля": 4,
    "мая": 5,
    "июня": 6,
    "июля": 7,
    "августа": 8,
    "сентября": 9,
    "октября": 10,
    "ноября": 11,
    "декабря": 12,
}


class BaseNewsParser(ABC):
    source = "unknown"

    def __init__(self, sleep_range=(0.2, 0.5), timeout=20, request_attempts=3):
        self.sleep_range = sleep_range
        self.timeout = timeout
        self.request_attempts = request_attempts
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    @abstractmethod
    def fetch_day(self, day):
        pass

    def parse_range(self, date_from, date_till):
        start = pd.to_datetime(date_from).date()
        end = pd.to_datetime(date_till).date()

        all_items = []
        current = start

        while current <= end:
            try:
                items = self.fetch_day(current)
                all_items.extend(items)
            except Exception as error:
                print(f"[{self.source}] ошибка за {current}: {error}")

            current += timedelta(days=1)

        return normalize_news(pd.DataFrame(all_items))

    def _sleep(self):
        time.sleep(random.uniform(*self.sleep_range))

    def _get_soup(self, url, method="GET"):
        last_error = None

        for attempt in range(self.request_attempts):
            try:
                self._sleep()

                response = self.session.request(method, url, timeout=self.timeout)
                response.raise_for_status()

                if not response.encoding or response.encoding.lower() == "iso-8859-1":
                    response.encoding = response.apparent_encoding or "utf-8"

                return BeautifulSoup(response.text, "html.parser")
            except requests.RequestException as error:
                last_error = error
                if attempt < self.request_attempts - 1:
                    time.sleep(1 + attempt)

        raise last_error


class SmartLabParser(BaseNewsParser):
    source = "smart-lab.ru"

    def fetch_day(self, day):
        date_str = day.strftime("%Y-%m-%d")
        results = []

        for page in range(1, 4):
            if page == 1:
                url = f"https://smart-lab.ru/news/date/{date_str}/"
            else:
                url = f"https://smart-lab.ru/news/date/{date_str}/page{page}/"

            soup = self._get_soup(url)
            items = soup.find_all("h3", class_="feed title bluid_48504")

            if not items:
                break

            for item in items:
                link_tag = item.find("a")

                if not link_tag:
                    continue

                title = link_tag.get("title", "").strip()
                href = link_tag.get("href", "").strip()

                if not title:
                    continue

                results.append(
                    {
                        "source": self.source,
                        "published_at": pd.Timestamp(day),
                        "title": title,
                        "link": "https://smart-lab.ru" + href,
                        "ticker": None,
                        "rubric": None,
                    }
                )

        return results


class KommersantParser(BaseNewsParser):
    source = "kommersant.ru"

    rubrics = {
        "Экономика": 3,
        "Бизнес": 4,
        "Финансы": 40,
        "Потребительский рынок": 41,
    }

    def fetch_day(self, day):
        results = []
        date_str = day.strftime("%Y-%m-%d")

        for rubric_name, rubric_id in self.rubrics.items():
            url = f"https://www.kommersant.ru/archive/rubric/{rubric_id}/day/{date_str}"
            soup = self._get_soup(url)
            articles = soup.select("article.rubric_lenta__item")

            for article in articles:
                raw_date = article.get("data-article-date", date_str).strip()
                tag_node = article.select_one("p.rubric_lenta__item_tag")
                raw_time = tag_node.get_text(strip=True).split(", ")[-1] if tag_node else ""

                dt = combine_date_time(raw_date, raw_time)

                title = article.get("data-article-title", "").strip()
                link = article.get("data-article-url", "").strip()

                if not title:
                    continue

                results.append(
                    {
                        "source": self.source,
                        "published_at": dt,
                        "title": title,
                        "link": link,
                        "ticker": None,
                        "rubric": rubric_name,
                    }
                )

        return results


class InterfaxParser(BaseNewsParser):
    source = "interfax.ru"

    def fetch_day(self, day):
        url = f"https://www.interfax.ru/business/news/{day:%Y/%m/%d}/"
        soup = self._get_soup(url)

        blocks = soup.select("div.an > div[data-id]")
        results = []

        for block in blocks:
            time_tag = block.find("span")
            title_tag = block.find("h3")
            link_tag = block.find("a")

            raw_time = time_tag.get_text(strip=True) if time_tag else ""
            dt = combine_day_time(day, raw_time)

            title = title_tag.get_text(strip=True) if title_tag else ""
            href = link_tag.get("href", "") if link_tag else ""

            if not title:
                continue

            results.append(
                {
                    "source": self.source,
                    "published_at": dt,
                    "title": title,
                    "link": f"https://www.interfax.ru{href}" if href else "",
                    "ticker": None,
                    "rubric": "Бизнес",
                }
            )

        return results


class InvestFundsParser(BaseNewsParser):
    source = "investfunds.ru"

    def __init__(self, max_pages=1000, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_pages = max_pages

    def fetch_day(self, day):
        return []

    def parse_range(self, date_from, date_till):
        start = pd.to_datetime(date_from).date()
        end = pd.to_datetime(date_till).date()
        results = []
        should_stop = False

        for page in range(1, self.max_pages + 1):
            url = f"https://investfunds.ru/news/?limit=50&page={page}"
            soup = self._get_soup(url, method="POST")
            news_list = soup.select("ul.news_list li")

            if not news_list:
                break

            current_date = ""
            page_has_rows_in_range = False

            for item in news_list:
                classes = item.get("class", [])

                if "date" in classes:
                    current_date = item.get_text(strip=True)
                    continue

                if "item" not in classes:
                    continue

                title_tag = item.select_one("div.lnk a.indent_right_10")
                if not title_tag:
                    continue

                time_tag = item.select_one("span.time")
                source_tag = item.select_one("div.lnk a.source")
                raw_time = time_tag.get_text(strip=True) if time_tag else ""
                published_at = combine_russian_date_time(current_date, raw_time)

                if published_at is None:
                    continue

                news_day = published_at.date()
                if news_day < start:
                    should_stop = True
                    continue

                if news_day > end:
                    continue

                page_has_rows_in_range = True
                href = title_tag.get("href", "").strip()

                results.append(
                    {
                        "source": self.source,
                        "published_at": published_at,
                        "title": title_tag.get_text(strip=True),
                        "link": f"https://investfunds.ru{href}" if href.startswith("/") else href,
                        "ticker": None,
                        "rubric": source_tag.get_text(strip=True) if source_tag else None,
                    }
                )

            if should_stop and not page_has_rows_in_range:
                break

        return normalize_news(pd.DataFrame(results))


def combine_date_time(raw_date, raw_time):
    try:
        date_part = datetime.strptime(raw_date, "%Y-%m-%d").date()
    except ValueError:
        return None

    try:
        time_part = datetime.strptime(raw_time, "%H:%M").time() if raw_time else datetime.min.time()
    except ValueError:
        time_part = datetime.min.time()

    return datetime.combine(date_part, time_part)


def combine_day_time(day, raw_time):
    try:
        time_part = datetime.strptime(raw_time, "%H:%M").time() if raw_time else datetime.min.time()
    except ValueError:
        time_part = datetime.min.time()

    return datetime.combine(day, time_part)


def combine_russian_date_time(raw_date, raw_time):
    raw_date = clean_text(raw_date).lower()
    if not raw_date:
        return None

    today = datetime.today().date()
    if raw_date == "сегодня":
        date_part = today
    elif raw_date == "вчера":
        date_part = today - timedelta(days=1)
    else:
        pieces = raw_date.split()
        if len(pieces) < 3:
            return None

        try:
            day_number = int(pieces[0])
            month_number = RU_MONTHS.get(pieces[1])
            year = int(pieces[2])
        except ValueError:
            return None

        if month_number is None:
            return None

        date_part = date(year, month_number, day_number)

    try:
        time_part = datetime.strptime(raw_time, "%H:%M").time() if raw_time else datetime.min.time()
    except ValueError:
        time_part = datetime.min.time()

    return datetime.combine(date_part, time_part)


def clean_text(value):
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def normalize_news(df):
    if df.empty:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    df = df.copy()

    for column in NEWS_COLUMNS:
        if column not in df.columns:
            df[column] = None

    df = df[NEWS_COLUMNS]
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")
    df = df.dropna(subset=["published_at"])

    df["date"] = df["published_at"].dt.date.astype(str)
    df["time"] = df["published_at"].dt.strftime("%H:%M")

    df = df.drop_duplicates(subset=["title", "link"])
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)

    return df


def get_next_news_date_from_existing_file(path):
    file_path = Path(path)

    if not file_path.exists():
        return "2024-01-01"

    df = pd.read_csv(file_path)

    if df.empty:
        return "2024-01-01"

    if "published_at" in df.columns:
        dates = pd.to_datetime(df["published_at"], errors="coerce")
    elif "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce")
    else:
        return "2024-01-01"

    max_date = dates.max()

    if pd.isna(max_date):
        return "2024-01-01"

    return (max_date.date() + timedelta(days=1)).strftime("%Y-%m-%d")


def collect_news(date_from, date_till):
    parsers = [
        SmartLabParser(),
        InvestFundsParser(),
        KommersantParser(),
        InterfaxParser(),
    ]

    frames = []

    for parser in parsers:
        try:
            df = parser.parse_range(date_from, date_till)
        except Exception as error:
            print(f"[{parser.source}] источник пропущен: {error}")
            continue

        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame(columns=NEWS_COLUMNS)

    result = pd.concat(frames, ignore_index=True)
    result = normalize_news(result)

    return result


def update_news_file(
    output_path,
    date_from=None,
    date_till=None,
):
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if date_from is None:
        date_from = get_next_news_date_from_existing_file(output_path)

    if date_till is None:
        date_till = datetime.today().strftime("%Y-%m-%d")

    new_df = collect_news(date_from, date_till)

    if output_file.exists():
        old_df = pd.read_csv(output_file)
        full_df = pd.concat([old_df, new_df], ignore_index=True)
    else:
        full_df = new_df

    full_df = normalize_news(full_df)
    full_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    return full_df
