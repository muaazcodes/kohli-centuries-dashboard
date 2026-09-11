import os
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Virat Kohli - Century Ledger", layout="wide")

# ---- Load data (CSV must be in the SAME folder as this script) --------
CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Virat_Kohli_100s.csv")
df = pd.read_csv(CSV_PATH)

# ---- Sidebar filter ----------------------------------------------------
st.sidebar.header("Filter")
formats = ["All"] + sorted(df["Format"].unique().tolist())
selected_format = st.sidebar.selectbox("Format", formats)

data = df if selected_format == "All" else df[df["Format"] == selected_format]

st.title("The Century Ledger")
st.caption("Every international hundred Virat Kohli has scored, filtered by format.")

if data.empty:
    st.warning("No centuries in this format.")
    st.stop()

# ---- Stat cards ----------------------------------------------------
avg_sr = data["Strike Rate"].mean()
win_pct = (data["Win"] == "Yes").mean() * 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("Centuries", len(data))
c2.metric("Avg Strike Rate", "{:.1f}".format(avg_sr))
c3.metric("Team Won", "{:.0f}%".format(win_pct))
c4.metric("Highest Score", int(data["Score"].max()))

st.divider()

# ---- Strike rate by year ----------------------------------------------
st.subheader("Strike rate by year")
fig_sr = px.scatter(
    data,
    x="Year",
    y="Strike Rate",
    color="Format" if selected_format == "All" else None,
    hover_data=["Against", "Score", "Balls", "Venue"],
)
fig_sr.update_traces(marker=dict(size=10))
st.plotly_chart(fig_sr, use_container_width=True)

col1, col2 = st.columns(2)

# ---- Centuries per opponent --------------------------------------------
with col1:
    st.subheader("Centuries against each opponent")
    opp_counts = data["Against"].value_counts().sort_values(ascending=True)
    fig_opp = px.bar(
        x=opp_counts.values,
        y=opp_counts.index,
        orientation="h",
        labels={"x": "Centuries", "y": "Opponent"},
    )
    st.plotly_chart(fig_opp, use_container_width=True)

# ---- Score vs balls faced ----------------------------------------------
with col2:
    st.subheader("Score vs. balls faced")
    fig_scatter = px.scatter(
        data,
        x="Balls",
        y="Score",
        hover_data=["Against", "Year", "Strike Rate"],
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

with st.expander("Show raw data"):
    st.dataframe(data.reset_index(drop=True))