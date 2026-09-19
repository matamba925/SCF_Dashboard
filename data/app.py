# 1. Imports
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, Input, Output, dcc, html
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 2. Data Wrangling Logic
def resolve_dataset_path():
    """Prefer the provided WDI CSV, then common local fallbacks."""
    candidates = [
        os.getenv("WDI_DATA_PATH"),
        "data/WDICSV.csv",
        "WDICSV.csv",
        "data/my_local_data.csv",
        "data/local_data.csv",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(Path(candidate))
    return None


def load_indicator_data(filepath=None):
    """Load the WDI CSV and return the full long-form dataset."""
    source = filepath or resolve_dataset_path()
    if source and Path(source).exists():
        df = pd.read_csv(source)
        if {"Country Name", "Indicator Name"}.issubset(df.columns):
            year_columns = [col for col in df.columns if str(col).isdigit()]
            if year_columns:
                # Convert wide data to long format for easier interactive analysis.
                id_vars = ["Country Name", "Country Code", "Indicator Name", "Indicator Code"]
                long_df = df[id_vars + year_columns].melt(
                    id_vars=id_vars,
                    value_vars=year_columns,
                    var_name="year",
                    value_name="value",
                )
                long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
                long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
                return long_df.dropna(subset=["year", "value"]).sort_values(["Indicator Name", "year"])
    return pd.DataFrame(
        {
            "Country Name": ["Sample A", "Sample B", "Sample C", "Sample D"],
            "Country Code": ["A", "B", "C", "D"],
            "Indicator Name": ["Access to electricity (% of population)"] * 4,
            "Indicator Code": ["ELEC"] * 4,
            "year": [2020, 2020, 2020, 2020],
            "value": [80, 65, 72, 90],
        }
    )


def get_high_var_features(trimmed=True, return_feat_names=False, df=None):
    """Return the highest-variance indicators from the WDI data."""
    model_df = raw_df if df is None else df
    if model_df is None or model_df.empty:
        empty = pd.Series(dtype=float)
        return empty.index if return_feat_names else empty

    variance_by_indicator = (
        model_df.groupby("Indicator Name")["value"]
        .apply(lambda s: s.dropna().sort_values().tolist())
    )

    if trimmed:
        def trimmed_variance(values):
            vals = pd.Series(values)
            if len(vals) < 3:
                return float("nan")
            cutoff = max(1, int(len(vals) * 0.10))
            trimmed = vals.iloc[cutoff:-cutoff] if cutoff * 2 < len(vals) else vals
            return float(trimmed.var(ddof=0)) if not trimmed.empty else float("nan")

        variance_by_indicator = variance_by_indicator.apply(trimmed_variance)
    else:
        variance_by_indicator = model_df.groupby("Indicator Name")["value"].var(ddof=0)

    top_five = variance_by_indicator.dropna().sort_values(ascending=False).head(5)
    if return_feat_names:
        return list(top_five.index)
    return top_five


def get_model_metrics(df=None, trimmed=True, k=2, return_metrics=False):
    """Build ``KMeans`` model based on five highest-variance features in ``df``.

    Parameters
    ----------
    trimmed : bool, default=True
        If ``True``, calculates trimmed variance, removing bottom and top 10%
        of observations.

    k : int, default=2
        Number of clusters.

    return_metrics : bool, default=False
        If ``False`` returns ``KMeans`` model. If ``True`` returns ``dict``
        with inertia and silhouette score.
    """
    model_df = raw_df if df is None else df
    if model_df is None or model_df.empty:
        raise ValueError("df must be a non-empty DataFrame.")

    variance_by_indicator = (
        model_df.groupby("Indicator Name")["value"]
        .apply(lambda s: s.dropna().sort_values().tolist())
    )

    if trimmed:
        def trimmed_variance(values):
            vals = pd.Series(values)
            if len(vals) < 3:
                return float("nan")
            cutoff = max(1, int(len(vals) * 0.10))
            trimmed = vals.iloc[cutoff:-cutoff] if cutoff * 2 < len(vals) else vals
            return float(trimmed.var(ddof=0)) if not trimmed.empty else float("nan")

        variance_by_indicator = variance_by_indicator.apply(trimmed_variance)
    else:
        variance_by_indicator = model_df.groupby("Indicator Name")["value"].var(ddof=0)

    top_features = variance_by_indicator.dropna().sort_values(ascending=False).head(5).index.tolist()
    if not top_features:
        raise ValueError("No valid features available for clustering.")

    feature_table = (
        model_df[model_df["Indicator Name"].isin(top_features)]
        .pivot_table(index="Country Name", columns="Indicator Name", values="value", aggfunc="mean")
        .loc[:, top_features]
        .dropna()
    )

    if feature_table.empty:
        raise ValueError("No rows remain after dropping missing values.")

    if k < 2 or k >= len(feature_table):
        raise ValueError("k must be between 2 and one less than the number of rows in the feature table.")

    features = feature_table[top_features].values
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, random_state=42, n_init=10)),
        ]
    )
    model.fit(features)

    if return_metrics:
        cluster_labels = model.predict(features)
        scaled_features = model.named_steps["scaler"].transform(features)
        metrics = {
            "inertia": float(model.named_steps["kmeans"].inertia_),
            "silhouette_score": float(silhouette_score(scaled_features, cluster_labels)),
        }
        return metrics

    return model


def get_pca_labels(df=None, trimmed=True, k=2):
    """Return a two-dimensional PCA projection with KMeans cluster labels."""
    model_df = raw_df if df is None else df
    feature_names = get_high_var_features(df=model_df, trimmed=trimmed, return_feat_names=True)
    if model_df is None or model_df.empty or not feature_names:
        raise ValueError("df must contain valid features for PCA.")

    feature_table = (
        model_df[model_df["Indicator Name"].isin(feature_names)]
        .pivot_table(index="Country Name", columns="Indicator Name", values="value", aggfunc="mean")
        .reindex(columns=feature_names)
        .dropna()
    )
    if feature_table.empty:
        raise ValueError("No rows remain after dropping missing values.")

    model = get_model_metrics(df=model_df, trimmed=trimmed, k=k)
    features = feature_table.to_numpy()
    scaled_features = model.named_steps["scaler"].transform(features)
    components = PCA(n_components=2).fit_transform(scaled_features)

    return pd.DataFrame(
        {
            "PC1": components[:, 0],
            "PC2": components[:, 1],
            "labels": model.predict(features),
        },
        index=feature_table.index,
    ).reset_index(drop=True)


# Load the data immediately.
raw_df = load_indicator_data()
indicator_list = sorted(raw_df["Indicator Name"].unique().tolist())
initial_indicator = indicator_list[0] if indicator_list else "Access to electricity (% of population)"
initial_year = int(raw_df["year"].max()) if not raw_df.empty else 2020

# 3. Initialize the App
app = Dash(__name__)


@app.callback(
    Output("bar-chart", "figure"),
    Input("trim-button", "value"),
)
def serve_bar_chart(trimmed=True):
    """Returns a horizontal bar chart of the five highest-variance features."""
    top_five_features = get_high_var_features(trimmed=trimmed, return_feat_names=False)
    fig = px.bar(
        x=top_five_features.values,
        y=top_five_features.index,
        orientation="h",
        title="Top 5 Highest-Variance Indicators",
        labels={"x": "Variance", "y": "Features"},
    )
    fig.update_layout(
        xaxis_title="Variance",
        yaxis_title="Features",
        yaxis={
            "categoryorder": "array",
            "categoryarray": list(reversed(top_five_features.index.tolist())),
        },
    )
    return fig


@app.callback(
    Output("scatter-plot", "figure"),
    Input("k-slider", "value"),
    Input("trim-button", "value"),
)
def serve_scatter_plot(k=2, trimmed=True):
    """Return a PCA scatter plot colored by KMeans cluster labels."""
    pca_labels = get_pca_labels(df=raw_df, trimmed=trimmed, k=k)
    return px.scatter(
        pca_labels,
        x="PC1",
        y="PC2",
        color="labels",
        title="K-Means Clusters in Two Dimensions",
        labels={"labels": "Cluster"},
    )


@app.callback(
    Output("metrics", "children"),
    Input("k-slider", "value"),
    Input("trim-button", "value"),
)
def serve_metrics(k=3, trimmed=True):
    """Return model inertia and silhouette score as H3 headers."""
    metrics = get_model_metrics(
        df=raw_df,
        trimmed=trimmed,
        k=k,
        return_metrics=True,
    )
    return (
        html.H3(f"Inertia: {metrics['inertia']:.2f}"),
        html.H3(f"Silhouette Score: {metrics['silhouette_score']:.3f}"),
    )


# 4. Define the Layout
app.layout = html.Div(
    [
        html.H1("World Development Indicators Dashboard"),
        html.P("Explore development indicators from your local WDI dataset."),
        html.H2("K-means Clustering"),
        html.H3("Number of Clusters (k)"),
        dcc.Slider(
            id="k-slider",
            min=2,
            max=12,
            step=1,
            value=3,
            marks={i: str(i) for i in range(2, 13)},
            tooltip={"placement": "bottom", "always_visible": True},
        ),
        html.Div(
            [
                html.Label("Select Indicator:"),
                dcc.Dropdown(
                    id="indicator-dropdown",
                    options=[{"label": item, "value": item} for item in indicator_list],
                    value=initial_indicator,
                    clearable=False,
                ),
                html.Br(),
                html.Label("Select Year:"),
                dcc.Slider(
                    id="year-slider",
                    min=int(raw_df["year"].min()) if not raw_df.empty else 2020,
                    max=int(raw_df["year"].max()) if not raw_df.empty else 2020,
                    step=1,
                    value=int(raw_df["year"].max()) if not raw_df.empty else 2020,
                    marks={int(y): str(int(y)) for y in sorted(raw_df["year"].unique())[:10] if pd.notna(y)},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Br(),
                dcc.RadioItems(
                    id="trim-button",
                    options=[
                        {"label": "trimmed", "value": True},
                        {"label": "not trimmed", "value": False},
                    ],
                    value=True,
                    inline=True,
                ),
            ],
            style={"width": "60%", "marginBottom": "20px"},
        ),
        dcc.Graph(figure=serve_bar_chart(), id="bar-chart"),
        dcc.Graph(id="scatter-plot"),
        html.Div(id="metrics"),
    ]
)


# 5. Run the Server
if __name__ == '__main__':
    app.run(debug=True)
