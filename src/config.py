STOCKS_PATH = "data/all_1h_stocks_2020_2026.csv"
NEWS_RAW_PATH = "data/news_raw_2020_2026_all_sources.csv"
NEWS_PATH = "data/news_features_2020_2026.parquet"
RESULTS_PATH = "data/final_model_outputs/diploma_experiment_results.csv"
NEWS_24H_COMPARISON_RESULTS_PATH = "updates/model_outputs/news_24h_event_experiment_strong/news_24h_results.csv"
NEWS_24H_COMPARISON_SIGNALS_PATH = "updates/model_outputs/news_24h_event_experiment_strong/news_24h_signal_slices.csv"
NEWS_24H_COMPARISON_PREDICTIONS_PATH = "updates/model_outputs/news_24h_event_experiment_strong/news_24h_test_predictions.csv"

DEFAULT_TICKER = "SBER"

DEFAULT_TICKERS = [
    "SBER",
    "GAZP",
    "LKOH",
    "YDEX",
    "VTBR",
    "ROSN",
    "NVTK",
    "GMKN",
    "TATN",
    "MOEX",
    "AFLT",
    "MTSS",
    "ALRS",
    "MGNT",
    "OZON",
    "VKCO",
    "PLZL",
    "CHMF",
    "NLMK",
    "RUAL",
    "TRNFP",
    "AFKS",
]

TICKER_KEYWORDS = {
    "SBER": ["сбер", "сбербанк", "sber", "sberbank"],
    "GAZP": ["газпром", "gazprom", "gazp"],
    "LKOH": ["лукойл", "lukoil", "lkoh"],
    "YDEX": ["яндекс", "yandex", "ydex"],
    "VTBR": ["втб", "vtb", "vtbr"],
    "ROSN": ["роснефть", "rosneft", "rosn"],
    "NVTK": ["новатэк", "novatek", "nvtk"],
    "GMKN": ["норникель", "норильский никель", "gmkn"],
    "TATN": ["татнефть", "tatneft", "tatn"],
    "MOEX": ["мосбиржа", "московская биржа", "moex"],
    "AFLT": ["аэрофлот", "aeroflot", "aflt"],
    "MTSS": ["мтс", "mts", "mtss"],
    "ALRS": ["алроса", "alrosa", "alrs"],
    "MGNT": ["магнит", "magnit", "mgnt"],
    "OZON": ["озон", "ozon"],
    "VKCO": ["vk", "вконтакте", "vkco"],
    "PLZL": ["полюс", "polyus", "plzl"],
    "CHMF": ["северсталь", "severstal", "chmf"],
    "NLMK": ["нлмк", "nlmk"],
    "RUAL": ["русал", "rusal"],
    "TRNFP": ["транснефть", "transneft", "trnfp"],
    "AFKS": ["афк", "система", "afks"],
}

SIGNALS_PATH = "data/final_model_outputs/diploma_model_signals.csv"
NEWS_24H_RESULTS_PATH = "data/final_news_24h_model_outputs/news_24h_results.csv"
NEWS_24H_SIGNALS_PATH = "data/final_news_24h_model_outputs/news_24h_signal_slices.csv"
NEWS_24H_PREDICTIONS_PATH = "data/final_news_24h_model_outputs/news_24h_test_predictions.csv"
NEWS_MEMORY_DIRECTION_RESULTS_PATH = "updates/model_outputs/news_memory_direction_experiment/news_memory_direction_results.csv"
NEWS_MEMORY_DIRECTION_PREDICTIONS_PATH = "updates/model_outputs/news_memory_direction_experiment/news_memory_direction_predictions.csv"
NEWS_MEMORY_DIRECTION_SIGNALS_PATH = "updates/model_outputs/news_memory_direction_experiment/news_memory_direction_signal_slices.csv"
MAGNITUDE_RESULTS_PATH = "updates/model_outputs/magnitude_estimation_experiment/magnitude_results.csv"
MAGNITUDE_PREDICTIONS_PATH = "updates/model_outputs/magnitude_estimation_experiment/magnitude_predictions.csv"
MAGNITUDE_SOURCE_ANALYSIS_PATH = "updates/model_outputs/magnitude_estimation_experiment/news_source_reaction_analysis.csv"
RETURN_24H_RESULTS_PATH = "updates/model_outputs/fixed_horizon_return_experiment_fast/fixed_horizon_return_results.csv"
RETURN_24H_PREDICTIONS_PATH = "updates/model_outputs/fixed_horizon_return_experiment_fast/fixed_horizon_return_predictions.csv"
CALENDAR_RETURN_24H_RESULTS_PATH = "updates/model_outputs/calendar_24h_return_experiment/calendar_24h_return_results.csv"
CALENDAR_RETURN_24H_PREDICTIONS_PATH = "updates/model_outputs/calendar_24h_return_experiment/calendar_24h_return_predictions.csv"
ABNORMAL_RETURN_24H_RESULTS_PATH = "updates/model_outputs/abnormal_24h_return_experiment/abnormal_24h_return_results.csv"
ABNORMAL_RETURN_24H_PREDICTIONS_PATH = "updates/model_outputs/abnormal_24h_return_experiment/abnormal_24h_return_predictions.csv"
FUTURE_RETURN_24H_PREDICTIONS_PATH = "updates/model_outputs/future_24h_forecasts/future_return_24h_predictions.csv"
FUTURE_ABNORMAL_24H_PREDICTIONS_PATH = "updates/model_outputs/future_24h_forecasts/future_abnormal_24h_predictions.csv"
MARKET_NEWS_ATTRIBUTION_PATH = "updates/model_outputs/market_news_attribution/market_news_attribution.csv"
MARKET_NEWS_DRIVER_SUMMARY_PATH = "updates/model_outputs/market_news_attribution/market_news_driver_summary.csv"
MARKET_NEWS_TICKER_SUMMARY_PATH = "updates/model_outputs/market_news_attribution/market_news_ticker_summary.csv"
COMMODITY_EXTERNAL_FACTORS_PATH = "data/commodity_external_factors.csv"
