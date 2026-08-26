import pandas as pd

df = pd.read_csv("data/creditcard.csv")

# sort chronologically just to be safe (should already be sorted, but don't assume)
df = df.sort_values("Time").reset_index(drop=True)

cutoff = df["Time"].quantile(0.7)

old_data = df[df["Time"] <= cutoff].copy()
new_data = df[df["Time"] > cutoff].copy()

print("Cutoff time (seconds):", cutoff)
print("Old data shape:", old_data.shape)
print("New data shape:", new_data.shape)
print("Old time range:", old_data["Time"].min(), "-", old_data["Time"].max())
print("New time range:", new_data["Time"].min(), "-", new_data["Time"].max())

old_data.to_csv("data/old_data.csv", index=False)
new_data.to_csv("data/new_data.csv", index=False)

print("\nSaved: data/old_data.csv and data/new_data.csv")