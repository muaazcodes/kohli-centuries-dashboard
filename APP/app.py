"""
Virat Kohli Centuries Dashboard
Streamlit app: stats overview, visual analysis, and a Ridge-regression
based Strike Rate predictor.

Run with:
    streamlit run app.py

Place these files in the SAME folder as app.py:
    - Virat_Kohli_100s.csv   (or update CSV_PATH below)
    - virat.jpg              (optional, header photo)
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

CSV_PATH = "Virat_Kohli_100s.csv"
PHOTO_PATH = "virat.jpg"

# ---------------------------------------------------------------
# Page config + light styling
# ---------------------------------------------------------------
st.set_page_config(
    page_title="Virat Kohli | Centuries Dashboard",
    page_icon="🏏",
    layout="wide",
)

st.markdown(
    """
    <style>
    .metric-card {
        background-color: rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 18px 16px;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    .metric-card h2 {
        margin: 0;
        font-size: 28px;
        color: #ffffff;
    }
    .metric-card p {
        margin: 4px 0 0 0;
        font-size: 13px;
        color: #d0d0d8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .section-title {
        font-size: 20px;
        font-weight: 600;
        margin-top: 18px;
        margin-bottom: 6px;
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------
# Data loading + model training (cached)
# ---------------------------------------------------------------
@st.cache_data
def load_data(path):
    df = pd.read_csv(path)
    return df


@st.cache_resource
def train_model(df):
    y = df["Strike Rate"]

    X = df.drop(
        ["Strike Rate", "Number", "Score", "Balls",
         "Against", "Venue", "Series", "Host Nation"],
        axis=1,
    )
    if "Score Category" in X.columns:
        X = X.drop("Score Category", axis=1)

    X = pd.get_dummies(
        X, columns=["Format", "Win", "Captain", "Not Out", "MOTM", "Position"],
        drop_first=True,
    )

    numeric_cols = ["Team Total", "Wickets lost", "Year", "Inning"]
    preprocessor = ColumnTransformer(
        transformers=[("num", StandardScaler(), numeric_cols)],
        remainder="passthrough",
    )
    model = Ridge(alpha=1.0)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    coefficients = pd.Series(model.coef_, index=X.columns).sort_values(ascending=False)

    return pipeline, X.columns, mae, r2, coefficients


if not os.path.exists(CSV_PATH):
    st.error(f"Dataset not found: '{CSV_PATH}'. Place it in the same folder as app.py.")
    st.stop()

df = load_data(CSV_PATH)
pipeline, feature_columns, mae, r2, coefficients = train_model(df)


# ---------------------------------------------------------------
# Header
# ---------------------------------------------------------------
st.title("Virat Kohli — Centuries Dashboard")


st.divider()


# ---------------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------------
if os.path.exists(PHOTO_PATH):
    st.sidebar.image(PHOTO_PATH, width=110)
st.sidebar.header("Filters")
formats = st.sidebar.multiselect(
    "Format", options=sorted(df["Format"].unique()), default=sorted(df["Format"].unique())
)
year_min, year_max = int(df["Year"].min()), int(df["Year"].max())
year_range = st.sidebar.slider("Year range", year_min, year_max, (year_min, year_max))

filtered_df = df[
    df["Format"].isin(formats)
    & df["Year"].between(year_range[0], year_range[1])
]

st.sidebar.divider()
st.sidebar.caption(f"Model: Ridge Regression · MAE {mae:.1f} · R² {r2:.2f}")


# ---------------------------------------------------------------
# Metric cards
# ---------------------------------------------------------------
st.markdown('<div class="section-title">Overview</div>', unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns(5)
metrics = [
    (m1, len(filtered_df), "Centuries"),
    (m2, int(filtered_df["Score"].max()) if len(filtered_df) else 0, "Highest Score"),
    (m3, round(filtered_df["Strike Rate"].mean(), 1) if len(filtered_df) else 0, "Avg Strike Rate"),
    (m4, int(filtered_df["Team Total"].mean()) if len(filtered_df) else 0, "Avg Team Total"),
    (m5, (filtered_df["MOTM"] == "Yes").sum() if len(filtered_df) else 0, "MOTM Awards"),
]
for col, value, label in metrics:
    col.markdown(
        f'<div class="metric-card"><h2>{value}</h2><p>{label}</p></div>',
        unsafe_allow_html=True,
    )

st.write("")


# ---------------------------------------------------------------
# Charts row 1
# ---------------------------------------------------------------
st.markdown('<div class="section-title">Trends</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)

with c1:
    yearly = filtered_df.groupby(["Year", "Format"])["Strike Rate"].mean().reset_index()
    fig = px.line(
        yearly, x="Year", y="Strike Rate", color="Format", markers=True,
        title="Average Strike Rate by Year",
    )
    fig.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fmt_avg = filtered_df.groupby("Format")["Strike Rate"].mean().reset_index()
    fig = px.bar(
        fmt_avg, x="Format", y="Strike Rate", color="Format",
        title="Average Strike Rate by Format",
    )
    fig.update_layout(margin=dict(t=40, b=10), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------
# Charts row 2
# ---------------------------------------------------------------
c3, c4 = st.columns(2)

with c3:
    opp_counts = filtered_df["Against"].value_counts().reset_index()
    opp_counts.columns = ["Against", "Centuries"]
    fig = px.bar(
        opp_counts, x="Against", y="Centuries",
        title="Centuries by Opponent",
    )
    fig.update_layout(margin=dict(t=40, b=10), xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

with c4:
    fig = px.scatter(
        filtered_df, x="Balls", y="Score", color="Format", size="Strike Rate",
        hover_data=["Against", "Year"],
        title="Score vs Balls Faced",
    )
    fig.update_layout(margin=dict(t=40, b=10))
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------
# Prediction tool
# ---------------------------------------------------------------
st.divider()
st.markdown('<div class="section-title">Predict Strike Rate</div>', unsafe_allow_html=True)
st.caption("Fill in innings details and the Ridge model will estimate the Strike Rate.")

p1, p2, p3, p4 = st.columns(4)
with p1:
    in_format = st.selectbox("Format", sorted(df["Format"].unique()))
    in_inning = st.selectbox("Inning", sorted(df["Inning"].unique()))
with p2:
    in_position = st.selectbox("Position", sorted(df["Position"].astype(str).unique()))
    in_year = st.number_input("Year", min_value=2008, max_value=2030, value=2024)
with p3:
    in_team_total = st.number_input("Team Total", min_value=50, max_value=700, value=300)
    in_wickets = st.slider("Wickets Lost", 0, 10, 5)
with p4:
    in_win = st.selectbox("Result", sorted(df["Win"].astype(str).unique()))
    in_notout = st.selectbox("Not Out", ["Yes", "No"])
    in_captain = st.selectbox("Captain", ["Yes", "No"])
    in_motm = st.selectbox("Man of the Match", ["Yes", "No"])

if st.button("Predict Strike Rate", type="primary"):
    row = pd.DataFrame([{
        "Format": in_format,
        "Inning": in_inning,
        "Position": in_position,
        "Year": in_year,
        "Team Total": in_team_total,
        "Wickets lost": in_wickets,
        "Win": in_win,
        "Not Out": in_notout,
        "Captain": in_captain,
        "MOTM": in_motm,
    }])
    row_encoded = pd.get_dummies(row)
    row_encoded = row_encoded.reindex(columns=feature_columns, fill_value=0)

    prediction = pipeline.predict(row_encoded)[0]
    st.success(f"Predicted Strike Rate: **{prediction:.1f}**")


# ---------------------------------------------------------------
# Model insight
# ---------------------------------------------------------------
st.divider()
st.markdown('<div class="section-title">Model Insights</div>', unsafe_allow_html=True)
i1, i2 = st.columns([2, 1])

with i1:
    fig = px.bar(
        coefficients.reset_index().rename(columns={"index": "Feature", 0: "Coefficient"}),
        x="Coefficient", y="Feature", orientation="h",
        title="Ridge Coefficients (impact on Strike Rate)",
    )
    fig.update_layout(margin=dict(t=40, b=10), height=500)
    st.plotly_chart(fig, use_container_width=True)




# ---------------------------------------------------------------
# Data table
# ---------------------------------------------------------------
st.divider()
st.markdown('<div class="section-title">Full Dataset</div>', unsafe_allow_html=True)
st.dataframe(filtered_df, use_container_width=True, hide_index=True)