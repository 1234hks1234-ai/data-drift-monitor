import pandas as pd
from scipy.stats import ks_2samp

old_data = pd.read_csv("data/old_data.csv")
new_data = pd.read_csv("data/new_data.csv")

# columns to test: everything except Time (used for the split) and Class (the label)
feature_columns = [col for col in old_data.columns if col not in ["Time", "Class"]]

results = []

for col in feature_columns:
    old_values = old_data[col].dropna()
    new_values = new_data[col].dropna()

    ks_stat, p_value = ks_2samp(old_values, new_values)

    statistically_significant = p_value < 0.05  # can be misleading with huge sample sizes

    if ks_stat < 0.1:
        severity = "negligible"
    elif ks_stat < 0.2:
        severity = "moderate"
    else:
        severity = "high"

    results.append({
        "column": col,
        "ks_stat": round(ks_stat, 4),
        "p_value": round(p_value, 6),
        "statistically_significant": statistically_significant,
        "severity": severity
    })

report = pd.DataFrame(results).sort_values("ks_stat", ascending=False)
report.to_csv("data/drift_report.csv", index=False)

print(report.to_string(index=False))

n_sig = report["statistically_significant"].sum()
n_real = (report["severity"] != "negligible").sum()
print(f"\n{n_sig} out of {len(report)} columns are 'statistically significant' (p < 0.05) -- "
      f"but with this sample size, that alone is not meaningful.")
print(f"{n_real} out of {len(report)} columns show at least MODERATE practical drift (ks_stat >= 0.1) -- "
      f"this is the number that actually matters.")