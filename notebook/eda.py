import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ── Load Data ─────────────────────────────────────────────────────────────────
df = pd.read_csv("data/application_train.csv")

print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print("Shape:", df.shape)
print("\nFirst 5 Rows:")
print(df.head())
print("\nData Types:")
print(df.dtypes.value_counts())

# ── Target Distribution ───────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TARGET DISTRIBUTION")
print("=" * 60)
print(df["TARGET"].value_counts())
print("\nTarget Percentage:")
print((df["TARGET"].value_counts(normalize=True) * 100).round(2).astype(str) + "%")

plt.figure(figsize=(6, 4))
df["TARGET"].value_counts().plot(kind="bar", color=["steelblue", "tomato"])
plt.title("Target Distribution (0 = No Default, 1 = Default)")
plt.xlabel("Target")
plt.ylabel("Count")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("notebook/target_distribution.png")
plt.close()
print("Saved: notebook/target_distribution.png")

# ── Missing Values Analysis ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("MISSING VALUES ANALYSIS")
print("=" * 60)

missing_count = df.isnull().sum()
missing_count = missing_count[missing_count > 0].sort_values(ascending=False)
missing_percent = (missing_count / len(df) * 100).round(2)

missing_df = pd.DataFrame({
    "Missing Count": missing_count,
    "Missing %": missing_percent
})

print(f"Total columns with missing values: {len(missing_df)}")
print("\nTop 20 Columns with Most Missing Values:")
print(missing_df.head(20))

high_missing = missing_df[missing_df["Missing %"] > 60]
print(f"\nColumns with >60% missing (dropped during training): {len(high_missing)}")
print(high_missing)

# ── Numerical Features Summary ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("NUMERICAL FEATURES SUMMARY")
print("=" * 60)

numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
if "TARGET" in numerical_cols:
    numerical_cols.remove("TARGET")

print(f"Total numerical features: {len(numerical_cols)}")
print(df[numerical_cols].describe().T)

# ── Categorical Features Summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("CATEGORICAL FEATURES SUMMARY")
print("=" * 60)

categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
print(f"Total categorical features: {len(categorical_cols)}")

for col in categorical_cols:
    print(f"\n{col} — {df[col].nunique()} unique values:")
    print(df[col].value_counts().head(5))

# ── Correlation with Target ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("CORRELATION WITH TARGET")
print("=" * 60)

corr = df[numerical_cols + ["TARGET"]].corr()["TARGET"].drop("TARGET")
corr_sorted = corr.abs().sort_values(ascending=False)

print("\nTop 15 Features Most Correlated with Target:")
print(corr_sorted.head(15))

plt.figure(figsize=(10, 6))
corr_sorted.head(15).plot(kind="bar", color="steelblue")
plt.title("Top 15 Features Correlated with Loan Default")
plt.ylabel("Absolute Correlation")
plt.tight_layout()
plt.savefig("notebook/top_correlations.png")
plt.close()
print("Saved: notebook/top_correlations.png")

# ── Age vs Default ────────────────────────────────────────────────────────────
if "DAYS_BIRTH" in df.columns:
    print("\n" + "=" * 60)
    print("AGE VS DEFAULT")
    print("=" * 60)

    df["AGE"] = (-df["DAYS_BIRTH"] / 365).astype(int)

    plt.figure(figsize=(8, 4))
    df[df["TARGET"] == 0]["AGE"].plot(kind="hist", alpha=0.6, label="No Default", bins=30, color="steelblue")
    df[df["TARGET"] == 1]["AGE"].plot(kind="hist", alpha=0.6, label="Default", bins=30, color="tomato")
    plt.title("Age Distribution by Default Status")
    plt.xlabel("Age")
    plt.legend()
    plt.tight_layout()
    plt.savefig("notebook/age_vs_default.png")
    plt.close()
    print("Saved: notebook/age_vs_default.png")

# ── Income vs Default ─────────────────────────────────────────────────────────
if "AMT_INCOME_TOTAL" in df.columns:
    print("\n" + "=" * 60)
    print("INCOME VS DEFAULT")
    print("=" * 60)

    print(df.groupby("TARGET")["AMT_INCOME_TOTAL"].describe())

    plt.figure(figsize=(8, 4))
    df[df["AMT_INCOME_TOTAL"] < df["AMT_INCOME_TOTAL"].quantile(0.99)].boxplot(
        column="AMT_INCOME_TOTAL", by="TARGET"
    )
    plt.title("Income by Default Status")
    plt.suptitle("")
    plt.xlabel("Target (0=No Default, 1=Default)")
    plt.ylabel("Income")
    plt.tight_layout()
    plt.savefig("notebook/income_vs_default.png")
    plt.close()
    print("Saved: notebook/income_vs_default.png")

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)