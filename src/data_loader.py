import pandas as pd
import streamlit as st


@st.cache_data
def load_stocks(path, file_mtime=None):
    df = pd.read_csv(path)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
    df["security"] = df["security"].astype(str)

    df = df.dropna(subset=["datetime"])
    df = df.sort_values(["security", "datetime"]).reset_index(drop=True)

    return df


@st.cache_data
def load_news(path, file_mtime=None):
    df = pd.read_parquet(path)
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce")

    df = df.dropna(subset=["published_at"])
    df = df.sort_values("published_at", ascending=False).reset_index(drop=True)

    return df


@st.cache_data
def load_results(path, file_mtime=None):
    df = pd.read_csv(path)
    return df


@st.cache_data
def load_signals(path, file_mtime=None):
    df = pd.read_csv(path)

    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")

    return df


@st.cache_data
def load_news_24h_predictions(path, file_mtime=None):
    df = pd.read_csv(path)

    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")

    return df


@st.cache_data
def load_magnitude_predictions(path, file_mtime=None):
    df = pd.read_csv(path, low_memory=False)

    if "start_time" in df.columns:
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")

    return df
