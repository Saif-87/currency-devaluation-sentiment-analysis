# -------------------------------------------------------------------------------- NOTEBOOK-CELL: IMPORTS
import dataiku
import pandas as pd

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: LOAD DATA
translation_summary_ds = dataiku.Dataset("translation_summary")
df = translation_summary_ds.get_dataframe()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: CLEANUP / VALIDATION
required_cols = ["Date", "Country", "Model Sentiment Label", "Model Sentiment Score"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"❌ Missing required column(s): {missing}")

CONF_THRESHOLD = 0.70

# Normalise label casing — the sentiment model emits e.g. "Bullish"/"Bearish"/"Neutral"
df["Model Sentiment Label"] = df["Model Sentiment Label"].astype(str).str.strip().str.lower()
print("🏷️ Label distribution:")
print(df["Model Sentiment Label"].value_counts())

before = len(df)
df = df.dropna(subset=["Model Sentiment Score"])
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df = df.dropna(subset=["Date"])
if len(df) != before:
    print(f"🧹 Dropped {before - len(df)} rows with missing score or unparseable date")

if "URL" in df.columns:
    before = len(df)
    df = df.drop_duplicates(subset=["URL"])
    if len(df) != before:
        print(f"🧹 Dropped {before - len(df)} duplicate URLs")

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: BUILD A SIGNED SCORE
# bullish -> +score, bearish -> -score, anything else -> 0.0
sign = df["Model Sentiment Label"].map({"bullish": 1, "bearish": -1}).fillna(0)
df["signed_score"] = sign * df["Model Sentiment Score"]

# -------------------------------------------------------------------------------- KEEP ONLY HIGH-CONFIDENCE ROWS
# The score is a softmax confidence (always >= 0) and signed_score.abs() == score
# for non-neutral rows, so the whole high-confidence rule is just a score threshold.
df = df[df["Model Sentiment Score"] >= CONF_THRESHOLD].copy()

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: GROUP BY COUNTRY + MONTH
df["year_month"] = df["Date"].dt.to_period("M")             # e.g. 2025-08

def majority_vote(labels):
    counts = labels.value_counts()
    return "neutral" if counts.empty else counts.idxmax()   # ties: first-seen label wins

agg_df = (
    df
    .groupby(["Country", "year_month"])
    .agg(
        avg_sentiment      = ("signed_score", "mean"),
        article_count      = ("Model Sentiment Label", "size"),
        pos_count          = ("Model Sentiment Label", lambda x: (x == "bullish").sum()),
        neg_count          = ("Model Sentiment Label", lambda x: (x == "bearish").sum()),
        neu_count          = ("Model Sentiment Label", lambda x: (x == "neutral").sum()),
        majority_sentiment = ("Model Sentiment Label", majority_vote)
    )
    .reset_index()
)

# -------------------------------------------------------------------------------- HUMAN-READABLE AVG LABEL
agg_df["avg_sentiment_label"] = agg_df["avg_sentiment"].apply(
    lambda x: "bullish" if x > 0 else ("bearish" if x < 0 else "neutral")
)

# -------------------------------------------------------------------------------- FORMAT DATE AS START-OF-MONTH ISO
agg_df["Date"] = (
    agg_df["year_month"]
    .dt.to_timestamp()                      # first day of month 00:00
    .dt.strftime("%Y-%m-%dT00:00:00.000Z")
)

agg_df = agg_df.drop(columns="year_month")

#RENAME COLUMNS
agg_df = agg_df.rename(columns={
    "Country": "COUNTRY",
    "Date": "Year-Month"
})


desired_order = [
    "Year-Month", "COUNTRY",
    "article_count", "pos_count", "neg_count", "neu_count",
    "majority_sentiment",
    "avg_sentiment_label",      # readable
    "avg_sentiment"             # numeric mean
]
agg_df = agg_df[desired_order]

# -------------------------------------------------------------------------------- NOTEBOOK-CELL: SAVE OUTPUT
score_groupby_ds = dataiku.Dataset("score_groupby")
score_groupby_ds.write_with_schema(agg_df)

if agg_df.empty:
    print("⚠️ 'score_groupby' written EMPTY — check the label distribution and CONF_THRESHOLD above")
else:
    print(
        f"✅ Saved {len(agg_df)} country-month rows "
        f"(high-confidence score ≥ {CONF_THRESHOLD}) into 'score_groupby'"
    )
