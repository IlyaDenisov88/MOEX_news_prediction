import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.config import NEWS_RAW_PATH, NEWS_PATH
from src.news_features import build_news_features


if __name__ == "__main__":
    build_news_features(
        raw_news_path=NEWS_RAW_PATH,
        output_path=NEWS_PATH,
        use_rubert=True,
        use_sentiment=True,
        rubert_batch_size=64,
        sentiment_batch_size=64,
        sentiment_max_length=256,
    )
