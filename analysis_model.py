"""Exploratory data analysis for AIML Dataset.csv.

Run from any directory: python analysis_model.py
Dependencies: pandas, matplotlib
"""
from pathlib import Path
import math
import pandas as pd
import matplotlib.pyplot as plt

DATA_FILENAME = "AIML Dataset.csv"

def find_data_file():
    roots = [Path.cwd(), Path(__file__).resolve().parent]
    seen = set()
    for root in roots:
        for folder in (root, *root.parents):
            candidate = folder / DATA_FILENAME
            if candidate not in seen:
                seen.add(candidate)
                if candidate.is_file():
                    return candidate
    raise FileNotFoundError(f"Could not find {DATA_FILENAME!r}. Put it beside this script or set DATA_PATH.")

DATA_PATH = find_data_file()
print(f"Loading {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df):,} rows and {df.shape[1]:,} columns.")

print("Preview:")
print(df.head().to_string(index=False))
print("Shape:", df.shape)
print("Columns and types:")
print(df.dtypes.to_string())
print("DataFrame info:")
df.info()

missing = pd.DataFrame({"missing_count": df.isna().sum(), "missing_percent": (df.isna().mean() * 100).round(2)})
print("Missing values:")
print(missing.sort_values("missing_count", ascending=False).to_string())
print(f"Exact duplicate rows: {df.duplicated().sum():,}")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
categorical_cols = df.select_dtypes(exclude="number").columns.tolist()
print("Numeric summary:")
if numeric_cols:
    print(df[numeric_cols].describe().T.to_string())
else:
    print("No numeric columns found.")
print("Categorical summary:")
if categorical_cols:
    for col in categorical_cols:
        print(f"\n{col}: {df[col].nunique(dropna=False):,} unique values")
        print(df[col].value_counts(dropna=False).head(12).to_string())
else:
    print("No categorical columns found.")

# Visualizations are selected dynamically from columns present in the dataset.
if numeric_cols:
    plot_cols = numeric_cols[:12]
    ncols = 3
    nrows = math.ceil(len(plot_cols) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows), squeeze=False)
    for ax, col in zip(axes.flat, plot_cols):
        values = df[col].dropna()
        if values.empty:
            ax.set_title(f"{col} (no non-missing values)")
        elif (values >= 0).all() and values.skew() > 2:
            clipped = values.clip(upper=values.quantile(0.99))
            ax.hist(clipped.map(math.log1p), bins=40, color="#2878B5")
            ax.set_xlabel("log1p(value), clipped at 99th percentile")
            ax.set_title(col)
        else:
            ax.hist(values, bins=40, color="#2878B5")
            ax.set_xlabel(col)
            ax.set_title(col)
    for ax in axes.flat[len(plot_cols):]:
        ax.set_visible(False)
    fig.tight_layout()
    plt.show()

if categorical_cols:
    plot_cols = categorical_cols[:6]
    fig, axes = plt.subplots(len(plot_cols), 1, figsize=(11, 3.5 * len(plot_cols)), squeeze=False)
    for ax, col in zip(axes[:, 0], plot_cols):
        counts = df[col].astype("string").fillna("<missing>").value_counts().head(12).sort_values()
        counts.plot(kind="barh", ax=ax, color="#41A36A")
        ax.set_title(f"{col}: most common values")
        ax.set_xlabel("Rows")
    fig.tight_layout()
    plt.show()

if len(numeric_cols) >= 2:
    corr_cols = numeric_cols[:20]
    corr = df[corr_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(corr_cols)), max(6, 0.5 * len(corr_cols))))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_cols)), corr_cols, rotation=90)
    ax.set_yticks(range(len(corr_cols)), corr_cols)
    ax.set_title("Numeric feature correlations")
    fig.colorbar(image, ax=ax, label="Correlation")
    fig.tight_layout()
    plt.show()

fraud_cols = [c for c in df.columns if str(c).strip().lower() == "isfraud"]
if fraud_cols:
    target = fraud_cols[0]
    counts = df[target].value_counts(dropna=False).sort_index()
    print("Fraud-label counts and percentages:")
    print(pd.DataFrame({"count": counts, "percent": (counts / len(df) * 100).round(3)}).to_string())
    counts.plot(kind="bar", figsize=(7, 4), color="#D95F02", title=f"Class counts: {target}")
    plt.xlabel("Class")
    plt.ylabel("Rows")
    plt.tight_layout()
    plt.show()
else:
    print("No column named isFraud found; skipping fraud-label chart.")
