import pandas as pd
import matplotlib.pyplot as plt

old_data = pd.read_csv("data/old_data.csv")
new_data = pd.read_csv("data/new_data.csv")
report = pd.read_csv("data/drift_report.csv")

# take the top 4 most-drifted columns
top_columns = report.sort_values("ks_stat", ascending=False).head(4)["column"].tolist()

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()

for i, col in enumerate(top_columns):
    ax = axes[i]
    ax.hist(old_data[col], bins=50, alpha=0.5, label="Old", density=True, color="steelblue")
    ax.hist(new_data[col], bins=50, alpha=0.5, label="New", density=True, color="orange")
    ks_stat = report.loc[report["column"] == col, "ks_stat"].values[0]
    ax.set_title(f"{col}  (ks_stat = {ks_stat})")
    ax.legend()

plt.tight_layout()
plt.savefig("data/top_drift_charts.png", dpi=120)
print("Saved chart to data/top_drift_charts.png")
print("Top drifted columns plotted:", top_columns)
