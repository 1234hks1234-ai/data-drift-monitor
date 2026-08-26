📊 Data Drift Monitor

A statistical drift detection system that flags when incoming data starts to look meaningfully different from the data a model was trained on — the kind of early-warning system most ML models in production don't have.

Why this matters

ML models are trained once on historical data, then deployed and often left running for months. Meanwhile the real world keeps changing — behavior patterns shift, and the model quietly keeps making decisions based on an outdated picture of reality. There's no crash, no error message — just a slow decline in prediction quality that's easy to miss until it costs real money or trust. This project is a simplified version of the "drift monitoring" systems used in production MLOps (similar in spirit to tools like Evidently AI or WhyLabs), built from scratch to understand how they actually work under the hood.

What it does
Splits a transaction dataset into "old" and "new" periods based on time
Runs a Kolmogorov-Smirnov (KS) test on every numeric feature, comparing the old vs. new distributions
Flags which features have drifted — and how severely
Displays everything in an interactive Streamlit dashboard
Dataset

Credit Card Fraud Detection dataset (mlg-ulb, via Kaggle) — 284,807 anonymized transactions over a 48-hour window, with 28 PCA-transformed features (V1–V28), Time, Amount, and a fraud label (Class).

Methodology

Splitting: Rows are sorted chronologically by Time and split at the 70th percentile — the first 70% of transactions become the "old" baseline, the last 30% become "new." A random shuffle-split was deliberately avoided: randomly shuffled rows from the same time period look statistically identical, which would defeat the purpose of the analysis.

Testing: For each of the 29 feature columns, a two-sample KS-test (scipy.stats.ks_2samp) compares the old and new distributions, producing:

ks_stat — the size of the gap between the two distributions (0 = identical)
p_value — the probability this gap could occur by chance if nothing had really changed

A key finding — and a correction along the way: with ~285K rows, the p-value flagged 29 out of 29 columns as "statistically significant" (p < 0.05). Taken at face value, that would suggest the entire dataset drifted — which turned out to be misleading. Large sample sizes make the KS-test extremely sensitive, so even trivial, practically meaningless differences become "significant." The fix: report ks_stat (effect size) alongside the p-value, and classify drift by practical severity instead of p-value alone:

ks_stat	Severity
< 0.10	Negligible
0.10 – 0.20	Moderate
> 0.20	High

Using this corrected approach, 14 of 29 columns show at least moderate drift, and 4 (V3, V1, V28, V25) show high drift — a far more honest and useful result than "everything changed."

Results
Old data: 199,368 transactions (hours 0–37)
New data: 85,439 transactions (hours 37–48)
14/29 columns show real (moderate+) drift
4 columns show high-severity drift: V3, V1, V28, V25
Likely driver: time-of-day / day-of-week effects on transaction patterns, rather than any single dramatic event
Tech stack
Python — pandas, scipy (KS-test), matplotlib
SQLite — structured storage and querying of old/new data
Streamlit — interactive dashboard
Project structure
drift-project/
├── data/                  # raw + split + report CSVs
├── db/                    # SQLite database
├── src/
│   ├── split_data.py      # chronological old/new split
│   ├── drift_metrics.py   # KS-test across all features
│   └── visualize_drift.py # static histogram comparisons
├── dashboard/
│   └── app.py              # Streamlit dashboard
└── README.md
Running it locally
bash
pip install pandas scipy matplotlib streamlit

python src/split_data.py
python src/drift_metrics.py
streamlit run dashboard/app.py
What I'd build next
Turn the drift check into a small API endpoint (POST /check-drift) that accepts two datasets and returns a JSON drift report
Add multivariate drift detection (PCA-based) to catch correlated shifts across multiple features at once, not just one column at a time
Automate it to run on a schedule and send an alert when drift is detected
What I learned

The hardest part of this project wasn't the code — it was catching that a "statistically significant" result isn't automatically a meaningful one. Reporting the p-value alone would have produced a technically-true but practically-wrong conclusion ("everything drifted"). Separating statistical significance from practical significance is the real lesson this project is built around.