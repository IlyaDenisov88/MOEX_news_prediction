from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


OUT_DIR = Path("/Users/krokodile/Desktop/moex_news_app copy/outputs/vkr_plotly_charts")
OUT_DIR.mkdir(parents=True, exist_ok=True)


CLASSIFICATION_ROWS = [
    ("Только новости", "Logistic Regression", 0.388, 0.497, 0.497, 0.429, 0.453),
    ("Только новости", "ExtraTrees", 0.477, 0.501, 0.505, 0.455, 0.484),
    ("Только новости", "CatBoost", 0.483, 0.497, 0.511, 0.607, 0.569),
    ("Только новости", "HistGradientBoosting", 0.515, 0.491, 0.504, 0.500, 0.516),
    ("Рынок в момент новости", "Logistic Regression", 0.513, 0.669, 0.676, 0.839, 0.796),
    ("Рынок в момент новости", "ExtraTrees", 0.513, 0.662, 0.672, 0.821, 0.818),
    ("Рынок в момент новости", "CatBoost", 0.541, 0.680, 0.688, 0.875, 0.804),
    ("Рынок в момент новости", "HistGradientBoosting", 0.546, 0.669, 0.681, 0.866, 0.813),
    ("Новости + рынок", "Logistic Regression", 0.524, 0.648, 0.651, 0.750, 0.751),
    ("Новости + рынок", "ExtraTrees", 0.518, 0.660, 0.670, 0.830, 0.796),
    ("Новости + рынок", "CatBoost", 0.538, 0.675, 0.680, 0.866, 0.791),
    ("Новости + рынок", "HistGradientBoosting", 0.551, 0.669, 0.681, 0.866, 0.796),
]

REGRESSION_ROWS = [
    ("Только новости", "ExtraTrees", 1.585, 2.259, -0.056, 0.487),
    ("Только новости", "CatBoost", 1.563, 2.246, -0.044, 0.483),
    ("Только новости", "HistGradientBoosting", 1.545, 2.213, -0.013, 0.504),
    ("Рынок в момент новости", "ExtraTrees", 1.593, 2.272, -0.067, 0.473),
    ("Рынок в момент новости", "CatBoost", 1.560, 2.235, -0.033, 0.454),
    ("Рынок в момент новости", "HistGradientBoosting", 1.565, 2.246, -0.043, 0.490),
    ("Новости + рынок", "ExtraTrees", 1.587, 2.267, -0.063, 0.476),
    ("Новости + рынок", "CatBoost", 1.565, 2.243, -0.040, 0.455),
    ("Новости + рынок", "HistGradientBoosting", 1.575, 2.257, -0.054, 0.472),
]

FEATURE_ORDER = ["Только новости", "Рынок в момент новости", "Новости + рынок"]
MODEL_ORDER = ["Logistic Regression", "ExtraTrees", "CatBoost", "HistGradientBoosting"]
MODEL_SHORT = {
    "Logistic Regression": "LogReg",
    "ExtraTrees": "ExtraTrees",
    "CatBoost": "CatBoost",
    "HistGradientBoosting": "HGBoost",
}
COLORS = {
    "Logistic Regression": "#2563eb",
    "ExtraTrees": "#7c3aed",
    "CatBoost": "#059669",
    "HistGradientBoosting": "#f97316",
}
SYMBOLS = {
    "Только новости": "circle",
    "Рынок в момент новости": "diamond",
    "Новости + рынок": "square",
}


def base_layout(fig, title, height=720):
    fig.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=26)),
        template="plotly_white",
        height=height,
        width=1280,
        font=dict(family="Arial", size=15, color="#1f2937"),
        margin=dict(l=70, r=40, t=105, b=80),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.75)",
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor="#e5e7eb", zeroline=False)
    return fig


def save(fig, name):
    html = OUT_DIR / f"{name}.html"
    fig.write_html(
        html,
        include_plotlyjs=True,
        full_html=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "png",
                "filename": name,
                "height": 720,
                "width": 1280,
                "scale": 2,
            },
        },
    )
    return html


def build_data():
    df = pd.DataFrame(
        CLASSIFICATION_ROWS,
        columns=[
            "feature_set",
            "model",
            "f1",
            "roc_auc",
            "pr_auc",
            "precision_at_5pct",
            "precision_at_10pct",
        ],
    )
    df["model_short"] = df["model"].map(MODEL_SHORT)
    df["feature_set"] = pd.Categorical(df["feature_set"], FEATURE_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df = df.sort_values(["feature_set", "model"])
    return df


def build_regression_data():
    df = pd.DataFrame(
        REGRESSION_ROWS,
        columns=["feature_set", "model", "mae_pct", "rmse_pct", "r2", "hit_rate"],
    )
    df["model_short"] = df["model"].map(MODEL_SHORT)
    df["feature_set"] = pd.Categorical(df["feature_set"], FEATURE_ORDER, ordered=True)
    df["model"] = pd.Categorical(df["model"], MODEL_ORDER, ordered=True)
    df = df.sort_values(["feature_set", "model"])
    return df


def best_by_model_bars(df):
    best = (
        df.sort_values(["model", "precision_at_5pct", "roc_auc"], ascending=[True, False, False])
        .groupby("model", observed=True)
        .head(1)
        .sort_values("model")
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "ROC-AUC: лучшее значение по модели",
            "Precision@5%: качество самых уверенных сигналов",
        ),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Bar(
            x=best["model_short"],
            y=best["roc_auc"],
            marker_color="#3b82f6",
            text=best["roc_auc"].map(lambda v: f"{v:.3f}"),
            textposition="outside",
            hovertext=best["feature_set"].astype(str),
            hovertemplate="Модель: %{x}<br>ROC-AUC: %{y:.3f}<br>Набор: %{hovertext}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=best["model_short"],
            y=best["precision_at_5pct"],
            marker_color="#10b981",
            text=best["precision_at_5pct"].map(lambda v: f"{v:.3f}"),
            textposition="outside",
            hovertext=best["feature_set"].astype(str),
            hovertemplate="Модель: %{x}<br>Precision@5%: %{y:.3f}<br>Набор: %{hovertext}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    base_layout(fig, "Классификация: События 24ч. Сравнение моделей", height=740)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0.45, 0.92], tickformat=".2f")
    fig.add_annotation(
        text="Метрики относятся к задаче классификации: определить, будет ли значимое движение цены в следующие 24 часа.",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.12,
        showarrow=False,
        align="left",
        font=dict(size=16, color="#374151"),
        bgcolor="#f8fafc",
        bordercolor="#cbd5e1",
        borderwidth=1,
        borderpad=8,
    )
    return fig


def precision_5_10_bars(df):
    best = (
        df.sort_values(["model", "precision_at_5pct", "precision_at_10pct"], ascending=[True, False, False])
        .groupby("model", observed=True)
        .head(1)
        .sort_values("model")
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Precision@5%: самые уверенные 5% сигналов",
            "Precision@10%: самые уверенные 10% сигналов",
        ),
        horizontal_spacing=0.12,
    )
    for col, metric, color, label in [
        (1, "precision_at_5pct", "#10b981", "Precision@5%"),
        (2, "precision_at_10pct", "#14b8a6", "Precision@10%"),
    ]:
        fig.add_trace(
            go.Bar(
                x=best["model_short"],
                y=best[metric],
                marker_color=color,
                text=best[metric].map(lambda v: f"{v:.3f}"),
                textposition="outside",
                hovertext=best["feature_set"].astype(str),
                hovertemplate=f"Модель: %{{x}}<br>{label}: %{{y:.3f}}<br>Набор: %{{hovertext}}<extra></extra>",
            ),
            row=1,
            col=col,
        )
    base_layout(fig, "Классификация: События 24ч. Precision@5% и Precision@10%", height=700)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[0.40, 0.92], tickformat=".2f")
    fig.add_annotation(
        text="Для каждого алгоритма показан лучший вариант признакового пространства по Precision@5%.",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.13,
        showarrow=False,
        align="left",
        font=dict(size=16, color="#374151"),
        bgcolor="#f8fafc",
        bordercolor="#cbd5e1",
        borderwidth=1,
        borderpad=8,
    )
    return fig


def grouped_bars_by_feature_set(df, metric, label, color):
    fig = px.bar(
        df,
        x="feature_set",
        y=metric,
        color="model",
        barmode="group",
        category_orders={"feature_set": FEATURE_ORDER, "model": MODEL_ORDER},
        color_discrete_map=COLORS,
        text=df[metric].map(lambda v: f"{v:.3f}"),
        labels={"feature_set": "Набор признаков", metric: label, "model": "Алгоритм"},
    )
    base_layout(fig, f"Классификация: События 24ч. {label} по моделям и признакам", height=720)
    fig.update_traces(textposition="outside", cliponaxis=False)
    y_min = 0.40 if metric == "precision_at_5pct" else 0.45
    fig.update_yaxes(range=[y_min, 0.92], tickformat=".2f")
    return fig


def regression_metric_bars(df, metric, label, higher_is_better=False):
    fig = px.bar(
        df,
        x="feature_set",
        y=metric,
        color="model",
        barmode="group",
        category_orders={"feature_set": FEATURE_ORDER, "model": MODEL_ORDER},
        color_discrete_map=COLORS,
        text=df[metric].map(lambda v: f"{v:.3f}"),
        labels={"feature_set": "Набор признаков", metric: label, "model": "Алгоритм"},
    )
    direction = "выше лучше" if higher_is_better else "ниже лучше"
    base_layout(fig, f"Регрессия: Доходность 24ч. {label} по моделям и признакам ({direction})", height=720)
    fig.update_traces(textposition="outside", cliponaxis=False)
    if metric in {"mae_pct", "rmse_pct"}:
        fig.update_yaxes(range=[1.45, 2.35], tickformat=".2f")
    elif metric == "r2":
        fig.update_yaxes(range=[-0.08, 0.01], tickformat=".3f")
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    else:
        fig.update_yaxes(range=[0.43, 0.53], tickformat=".3f")
        fig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8")
    return fig


def regression_summary_bars(df):
    mae_best = (
        df.sort_values(["model", "mae_pct", "rmse_pct"], ascending=[True, True, True])
        .groupby("model", observed=True)
        .head(1)
        .sort_values("model")
    )
    hit_best = (
        df.sort_values(["model", "hit_rate", "mae_pct"], ascending=[True, False, True])
        .groupby("model", observed=True)
        .head(1)
        .sort_values("model")
    )
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("MAE, %: ниже лучше", "Hit-rate направления: выше лучше"),
        horizontal_spacing=0.12,
    )
    fig.add_trace(
        go.Bar(
            x=mae_best["model_short"],
            y=mae_best["mae_pct"],
            marker_color="#3b82f6",
            text=mae_best["mae_pct"].map(lambda v: f"{v:.3f}"),
            textposition="outside",
            hovertext=mae_best["feature_set"].astype(str),
            hovertemplate="Модель: %{x}<br>MAE: %{y:.3f}%<br>Набор: %{hovertext}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=hit_best["model_short"],
            y=hit_best["hit_rate"],
            marker_color="#10b981",
            text=hit_best["hit_rate"].map(lambda v: f"{v:.3f}"),
            textposition="outside",
            hovertext=hit_best["feature_set"].astype(str),
            hovertemplate="Модель: %{x}<br>Hit-rate: %{y:.3f}<br>Набор: %{hovertext}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    base_layout(fig, "Регрессия: Доходность 24ч. Сравнение моделей", height=700)
    fig.update_layout(showlegend=False)
    fig.update_yaxes(range=[1.45, 1.62], tickformat=".3f", row=1, col=1)
    fig.update_yaxes(range=[0.43, 0.53], tickformat=".3f", row=1, col=2)
    fig.add_annotation(
        text="Метрики относятся к задаче регрессии: оценить будущую доходность за 24 часа. MAE/RMSE измеряются в процентных пунктах.",
        xref="paper",
        yref="paper",
        x=0,
        y=-0.13,
        showarrow=False,
        align="left",
        font=dict(size=16, color="#374151"),
        bgcolor="#f8fafc",
        bordercolor="#cbd5e1",
        borderwidth=1,
        borderpad=8,
    )
    return fig


def scatter_roc_precision(df):
    plot_df = df.copy()
    fig = go.Figure()
    for _, row in plot_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["roc_auc"]],
                y=[row["precision_at_5pct"]],
                mode="markers",
                marker=dict(
                    size=14 + 18 * row["f1"],
                    color=COLORS[str(row["model"])],
                    symbol=SYMBOLS[str(row["feature_set"])],
                    opacity=0.88,
                    line=dict(width=1.5, color="white"),
                ),
                showlegend=False,
                customdata=[[
                    str(row["model"]),
                    str(row["feature_set"]),
                    row["pr_auc"],
                    row["f1"],
                    row["precision_at_10pct"],
                ]],
                hovertemplate=(
                    "Алгоритм: %{customdata[0]}<br>"
                    "Набор признаков: %{customdata[1]}<br>"
                    "ROC-AUC: %{x:.3f}<br>"
                    "Precision@5%: %{y:.3f}<br>"
                    "PR-AUC: %{customdata[2]:.3f}<br>"
                    "F1: %{customdata[3]:.3f}<br>"
                    "Precision@10%: %{customdata[4]:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    # Compact custom legend: colors for algorithms, symbols for feature spaces.
    for model in MODEL_ORDER:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=13, color=COLORS[model], symbol="circle"),
                name=MODEL_SHORT[model],
                legendgroup="Алгоритм",
                legendgrouptitle_text="Алгоритм",
            )
        )
    for feature in FEATURE_ORDER:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=13, color="#475569", symbol=SYMBOLS[feature]),
                name=str(feature),
                legendgroup="Набор признаков",
                legendgrouptitle_text="Набор признаков",
            )
        )

    base_layout(fig, "Модели классификации", height=760)
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        margin=dict(l=70, r=270, t=105, b=80),
    )
    fig.update_xaxes(
        title="ROC-AUC",
        range=[0.47, 0.70],
        tickformat=".3f",
    )
    fig.update_yaxes(
        title="Precision@5%",
        range=[0.40, 0.91],
        tickformat=".3f",
    )
    fig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8")
    fig.add_vline(x=0.5, line_dash="dash", line_color="#94a3b8")
    fig.add_annotation(
        x=0.681,
        y=0.875,
        text="Лучший результат:<br>CatBoost + рынок",
        showarrow=True,
        arrowhead=2,
        ax=-95,
        ay=-35,
        bgcolor="white",
        bordercolor="#cbd5e1",
        borderwidth=1,
        font=dict(size=14),
    )
    return fig


def scatter_return_mae_metric(df, y_metric, y_title, output_note=None, annotation=None):
    plot_df = df.copy()
    fig = go.Figure()
    for _, row in plot_df.iterrows():
        fig.add_trace(
            go.Scatter(
                x=[row["mae_pct"]],
                y=[row[y_metric]],
                mode="markers",
                marker=dict(
                    size=17,
                    color=COLORS[str(row["model"])],
                    symbol=SYMBOLS[str(row["feature_set"])],
                    opacity=0.88,
                    line=dict(width=1.5, color="white"),
                ),
                showlegend=False,
                customdata=[[
                    str(row["model"]),
                    str(row["feature_set"]),
                    row["rmse_pct"],
                    row["r2"],
                    row["hit_rate"],
                ]],
                hovertemplate=(
                    "Алгоритм: %{customdata[0]}<br>"
                    "Набор признаков: %{customdata[1]}<br>"
                    "MAE: %{x:.3f}%<br>"
                    f"{y_title}: %{{y:.3f}}<br>"
                    "RMSE: %{customdata[2]:.3f}%<br>"
                    "R²: %{customdata[3]:.3f}<br>"
                    "Hit-rate: %{customdata[4]:.3f}"
                    "<extra></extra>"
                ),
            )
        )

    for model in ["ExtraTrees", "CatBoost", "HistGradientBoosting"]:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=13, color=COLORS[model], symbol="circle"),
                name=MODEL_SHORT[model],
                legendgroup="Алгоритм",
                legendgrouptitle_text="Алгоритм",
            )
        )
    for feature in FEATURE_ORDER:
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(size=13, color="#475569", symbol=SYMBOLS[feature]),
                name=str(feature),
                legendgroup="Набор признаков",
                legendgrouptitle_text="Набор признаков",
            )
        )

    base_layout(fig, "Модели доходности", height=780)
    fig.update_layout(
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="left",
            x=1.01,
            bgcolor="rgba(255,255,255,0.85)",
        ),
        margin=dict(l=80, r=270, t=105, b=125),
    )
    fig.update_xaxes(
        title="MAE, %",
        range=[1.535, 1.605],
        tickformat=".3f",
        gridcolor="#e5e7eb",
    )
    if y_metric == "hit_rate":
        fig.update_yaxes(title=y_title, range=[0.44, 0.515], tickformat=".3f", gridcolor="#e5e7eb")
        fig.add_hline(y=0.5, line_dash="dash", line_color="#94a3b8")
    elif y_metric == "rmse_pct":
        fig.update_yaxes(title=y_title, range=[2.20, 2.28], tickformat=".3f", gridcolor="#e5e7eb")
    elif y_metric == "r2":
        fig.update_yaxes(title=y_title, range=[-0.075, 0.005], tickformat=".3f", gridcolor="#e5e7eb")
        fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")

    if annotation:
        fig.add_annotation(
            x=annotation["x"],
            y=annotation["y"],
            text=annotation["text"],
            showarrow=True,
            arrowhead=2,
            ax=annotation.get("ax", 80),
            ay=annotation.get("ay", -35),
            bgcolor="white",
            bordercolor="#cbd5e1",
            borderwidth=1,
            font=dict(size=14),
        )
    if output_note:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0,
            y=-0.15,
            text=output_note,
            showarrow=False,
            align="left",
            font=dict(size=14, color="#374151"),
            bgcolor="#f8fafc",
            bordercolor="#cbd5e1",
            borderwidth=1,
            borderpad=7,
        )
    return fig


def scatter_return_mae_hit_rate(df):
    return scatter_return_mae_metric(
        df,
        "hit_rate",
        "Hit-rate направления",
        output_note=None,
        annotation={"x": 1.545, "y": 0.504, "text": "Лучший результат:<br>HGBoost + новости"},
    )


def main():
    df = build_data()
    reg_df = build_regression_data()
    df.to_csv(OUT_DIR / "classification_event_24h_metrics.csv", index=False)
    reg_df.to_csv(OUT_DIR / "regression_return_24h_metrics.csv", index=False)

    outputs = [
        save(best_by_model_bars(df), "classification_bar_best_by_model"),
        save(precision_5_10_bars(df), "classification_bar_precision5_precision10_best_by_model"),
        save(grouped_bars_by_feature_set(df, "roc_auc", "ROC-AUC", "#3b82f6"), "classification_bar_roc_auc_by_feature_set"),
        save(grouped_bars_by_feature_set(df, "precision_at_5pct", "Precision@5%", "#10b981"), "classification_bar_precision5_by_feature_set"),
        save(grouped_bars_by_feature_set(df, "precision_at_10pct", "Precision@10%", "#14b8a6"), "classification_bar_precision10_by_feature_set"),
        save(scatter_roc_precision(df), "classification_scatter_roc_auc_precision5"),
        save(scatter_return_mae_hit_rate(reg_df), "regression_scatter_mae_hit_rate"),
        save(scatter_return_mae_metric(
            reg_df,
            "rmse_pct",
            "RMSE, процентные пункты доходности",
            output_note="MAE и RMSE дают близкую картину: ошибка прогноза находится примерно на уровне 1.5-1.6 п.п. MAE и 2.2-2.3 п.п. RMSE.",
            annotation={"x": 1.545, "y": 2.213, "text": "Минимальная ошибка:<br>HGBoost + новости", "ax": 85, "ay": -35},
        ), "regression_scatter_mae_rmse"),
        save(scatter_return_mae_metric(
            reg_df,
            "r2",
            "R²",
            output_note="R² около нуля или ниже: модели слабо объясняют точную величину будущей доходности.",
            annotation={"x": 1.545, "y": -0.013, "text": "Лучший R²:<br>HGBoost + новости", "ax": 85, "ay": -35},
        ), "regression_scatter_mae_r2"),
        save(regression_summary_bars(reg_df), "regression_bar_summary_by_model"),
        save(regression_metric_bars(reg_df, "mae_pct", "MAE, %", higher_is_better=False), "regression_bar_mae_by_feature_set"),
        save(regression_metric_bars(reg_df, "rmse_pct", "RMSE, %", higher_is_better=False), "regression_bar_rmse_by_feature_set"),
        save(regression_metric_bars(reg_df, "r2", "R²", higher_is_better=True), "regression_bar_r2_by_feature_set"),
        save(regression_metric_bars(reg_df, "hit_rate", "Hit-rate", higher_is_better=True), "regression_bar_hit_rate_by_feature_set"),
    ]
    for path in outputs:
        print(path)
    print(OUT_DIR / "classification_event_24h_metrics.csv")


if __name__ == "__main__":
    main()
