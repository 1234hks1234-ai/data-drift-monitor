import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Data Drift Monitor", layout="wide")

st.title("📊 Data Drift Monitor")
st.caption("Monitoring feature drift between old and new transaction data (Credit Card Fraud dataset)")

# --- load data ---
old_data = pd.read_csv("data/old_data.csv")
new_data = pd.read_csv("data/new_data.csv")
report = pd.read_csv("data/drift_report.csv")

# --- summary metrics at the top ---
col1, col2, col3, col4 = st.columns(4)
col1.metric("Old data rows", f"{len(old_data):,}")
col2.metric("New data rows", f"{len(new_data):,}")
n_moderate_plus = (report["severity"] != "negligible").sum()
col3.metric("Columns w/ real drift", f"{n_moderate_plus} / {len(report)}")
n_high = (report["severity"] == "high").sum()
col4.metric("High-severity columns", n_high)

st.divider()

# --- color-coded table ---
st.subheader("Drift Report — All Columns")

def color_severity(val):
    colors = {"negligible": "background-color: #1e3d2f",
              "moderate": "background-color: #4d3b12",
              "high": "background-color: #5c1a1a"}
    return colors.get(val, "")

styled = report.style.map(color_severity, subset=["severity"])
st.dataframe(styled, use_container_width=True, hide_index=True)

st.caption("Note: 'statistically_significant' (p<0.05) is often True across the board due to large "
           "sample size — 'severity' (based on ks_stat) is the more reliable signal for real drift.")

st.divider()

# --- interactive column explorer ---
st.subheader("Explore a Column")

selected_col = st.selectbox("Pick a column to inspect:", report["column"].tolist())

row = report[report["column"] == selected_col].iloc[0]
c1, c2, c3 = st.columns(3)
c1.metric("KS statistic", row["ks_stat"])
c2.metric("p-value", row["p_value"])
c3.metric("Severity", row["severity"].upper())

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(old_data[selected_col], bins=50, alpha=0.5, label="Old", density=True, color="steelblue")
ax.hist(new_data[selected_col], bins=50, alpha=0.5, label="New", density=True, color="orange")
ax.set_title(f"{selected_col}: Old vs New Distribution")
ax.legend()
st.pyplot(fig)