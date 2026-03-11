![logo_ironhack_blue 7](https://user-images.githubusercontent.com/23629340/40541063-a07a0a8a-601a-11e8-91b5-2f13e4e6b441.png)

# Lab | Working with Messy Real-World Data

## Overview

In this lab you will clean, analyze, and prepare a real transactional dataset that contains every kind of mess you would encounter in production: missing values, invalid records, rare and evolving categories, class imbalance, and temporal structure that must be respected when splitting. You will work with the **Online Retail Dataset** from the UCI Machine Learning Repository — a dataset of ~540,000 transactions from a UK-based online retailer between December 2010 and December 2011.

By the end of this lab, you will have transformed a raw, messy dataset into a clean, analysis-ready form, handled class imbalance for a business-critical prediction task, and demonstrated both how data leakage happens and how to prevent it.

## Dataset

**Online Retail Dataset**
UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets/Online+Retail

Download the Excel file (`Online Retail.xlsx`) from the link above and place it in the root of your project directory.

> **Disclaimer:** This dataset is publicly available from the UCI Machine Learning Repository for research and teaching. Students must not redistribute the dataset outside the university nor use it for commercial purposes.

The dataset contains the following columns:

| Column | Description |
|---|---|
| `InvoiceNo` | Invoice number (prefix "C" indicates a cancellation) |
| `StockCode` | Product code |
| `Description` | Product name |
| `Quantity` | Quantity per transaction |
| `InvoiceDate` | Date and time of the transaction |
| `UnitPrice` | Price per unit in GBP |
| `CustomerID` | Customer identifier |
| `Country` | Customer's country |

## Learning Goals

By completing this lab, you will be able to:

- Profile a raw dataset to identify missing values, outliers, and data quality issues
- Apply appropriate strategies for different types of missing data
- Handle rare and high-cardinality categorical features
- Engineer a binary classification target and address class imbalance using resampling techniques
- Intentionally introduce data leakage, detect it, and fix it with a correct temporal split
- Produce a clean, documented, analysis-ready dataset

## Prerequisites

- Python 3.9+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `openpyxl`

Install dependencies if needed:

```bash
pip install pandas numpy scikit-learn matplotlib openpyxl
```

## Requirements

- Fork this repository to your own GitHub account.
- Clone your fork to your machine.
- Create a notebook named `m2-07-real-world-data-lab.ipynb`.

## Getting Started

Load the dataset and take a first look:

```python
import pandas as pd
import numpy as np

df = pd.read_excel("Online Retail.xlsx")
print(f"Shape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nFirst rows:\n{df.head()}")
```

---

## Task 1: Data Profiling and Missing Values (~25 minutes)

### 1.1 — Profile the raw data

Compute a comprehensive profile of the dataset:

- Number of rows, columns, and memory usage
- Missing value counts and percentages for each column
- Number of unique values per column
- Basic statistics for numeric columns (min, max, mean, median, std)

### 1.2 — Classify the missing data types

The two columns with significant missing values are `CustomerID` and `Description`.

For each:
- Determine whether the missingness is MNAR, MAR, or MCAR. Justify your answer with evidence (e.g., compare rows with and without missing values — do they differ systematically in other columns?).
- Decide on a strategy: deletion, imputation, or indicator column. Justify your choice.

Guiding questions:
- Are transactions with missing `CustomerID` different from those with a `CustomerID`? Check the distribution of `Country`, `Quantity`, and `UnitPrice` for both groups.
- Do transactions with missing `Description` have valid `StockCode` values? Could you recover descriptions from other rows with the same `StockCode`?

### 1.3 — Handle the missing values

Apply your chosen strategies. Document what you did and why. After handling missing values, print the remaining missing value counts to confirm.

### Deliverable for Task 1

A profiling summary, a written classification of missing data types with evidence, and clean handling with documentation.

---

## Task 2: Cleaning Invalid and Anomalous Records (~25 minutes)

### 2.1 — Identify cancellations

Invoices starting with "C" are cancellations. These have negative quantities and represent returns, not purchases.

- Count how many cancellation transactions exist.
- Decide whether to keep, remove, or flag them. Think about the downstream task: if you later want to predict customer churn or build a recommender, how do cancellations affect the analysis?

### 2.2 — Handle invalid quantities and prices

Investigate records with:
- `Quantity <= 0` (that are not cancellations)
- `UnitPrice <= 0`
- Extreme outliers in `Quantity` or `UnitPrice`

For each category of invalid record:
- Count how many exist
- Decide what to do (remove, cap, flag) and justify

### 2.3 — Clean and validate

Apply your cleaning rules. After cleaning, verify:
- No remaining negative quantities (unless you kept cancellations with a flag)
- No zero or negative prices
- Print the shape before and after cleaning

### Deliverable for Task 2

A documented cleaning pipeline with before/after counts, justifications for each decision, and validation checks.

---

## Task 3: Categorical Data Challenges (~20 minutes)

### 3.1 — Analyze the Country column

- How many unique countries are there?
- What percentage of transactions come from the top 5 countries?
- How many countries have fewer than 50 transactions?

Create a cleaned version of the `Country` column that groups rare countries (fewer than 50 transactions) into an "Other" category. Compare the number of categories before and after.

### 3.2 — Analyze the StockCode column

`StockCode` is a high-cardinality categorical feature.

- How many unique stock codes exist?
- Are there non-product codes (e.g., postage, discounts, manual adjustments)? Identify them by looking at codes that are not purely numeric or have unusual patterns.
- Decide how to handle non-product codes for a product-level analysis.

### 3.3 — Engineer a feature from Description

The `Description` column contains free text. Create a simple feature from it — for example, the word count of the description, or a flag for descriptions containing certain keywords (e.g., "SET", "PACK", "VINTAGE"). Show that your engineered feature has a meaningful relationship with `Quantity` or `UnitPrice`.

### Deliverable for Task 3

Analysis of categorical columns, a strategy for rare and high-cardinality categories, and one engineered feature with evidence of its usefulness.

---

## Task 4: Class Imbalance — Predicting High-Value Customers (~25 minutes)

### 4.1 — Engineer a binary target

Create a customer-level dataset by aggregating transactions per `CustomerID`. Compute:
- Total revenue (`Quantity * UnitPrice`)
- Number of orders (unique `InvoiceNo` values)
- Number of distinct products purchased
- Date of first and last purchase

Define a binary target: a customer is **high-value** if their total revenue is in the top 10%. Label these as `1` and the rest as `0`.

```python
# Hint: compute total revenue per customer, then use a percentile threshold
customer_revenue = df.groupby("CustomerID").apply(
    lambda x: (x["Quantity"] * x["UnitPrice"]).sum()
).reset_index(name="total_revenue")
```

### 4.2 — Measure the imbalance

- What is the class distribution (high-value vs. regular)?
- If a model always predicted "regular," what would its accuracy be?
- Why is accuracy a poor metric here?

### 4.3 — Apply resampling

Split the customer-level dataset into train and test sets (80/20). Then apply two resampling techniques to the **training set only**:

1. **Random oversampling** of the minority class
2. **Random undersampling** of the majority class

For each:
- Print the class distribution before and after resampling
- Train a simple model (e.g., `LogisticRegression` or `DecisionTreeClassifier`) on both the original and resampled training sets
- Evaluate on the **original (not resampled) test set** using precision, recall, and F1 for the high-value class

### Deliverable for Task 4

Customer-level dataset, class distribution analysis, and resampling results with evaluation metrics for each approach.

---

## Task 5: Data Leakage — Introduce, Detect, and Fix (~25 minutes)

### 5.1 — Intentionally introduce temporal leakage

The dataset spans December 2010 through December 2011. Suppose you want to predict whether a customer will make a purchase in December 2011 based on their behavior in earlier months.

First, do it **wrong**: randomly split all customer features (computed from the full date range) into train and test sets. Train a model and record its performance.

### 5.2 — Detect the leakage

- Check whether train and test sets contain features computed from overlapping time periods.
- Look for suspiciously high model performance (a sign of leakage).
- Compute feature-target correlations and identify any that seem too good to be true.

### 5.3 — Fix with a correct temporal split

Now do it **right**:
- Use data from December 2010 through September 2011 to compute customer features (the "observation window").
- Use data from October 2011 through December 2011 to create the target variable: did the customer make at least one purchase in this "prediction window"?
- Train the same model and compare performance.

```python
# Temporal split
observation_end = pd.Timestamp("2011-09-30")
prediction_start = pd.Timestamp("2011-10-01")

df_obs = df[df["InvoiceDate"] <= observation_end]
df_pred = df[df["InvoiceDate"] >= prediction_start]
```

### Deliverable for Task 5

Both the leaked and correct model results, with identification of the leaked information and a written explanation of why the temporal split is the correct approach.

---

## Submission

### What to submit

Submit the following file:

- The notebook `m2-07-real-world-data-lab.ipynb`

Do **not** include the dataset file (`Online Retail.xlsx`) in your submission.

### Definition of done (checklist)

Before you submit, make sure:

- [ ] The notebook runs top to bottom without errors after a kernel restart.
- [ ] All five tasks are completed with code, output, and written analysis.
- [ ] Missing value handling is documented and justified (not just applied silently).
- [ ] Resampling was applied only to the training set, and evaluation used the original test set.
- [ ] The temporal leakage experiment includes both the wrong and right approach, with a clear explanation of why they differ.
- [ ] The dataset file is not included in the commit.

### How to submit (Git workflow)

When you are done, save your notebook, then run:

```bash
git add .
git commit -m "Solved m2-07 lab"
git push -u origin HEAD
```

- Create a pull request from your fork.
- Paste the link to your pull request in the Student Portal.

## Evaluation Criteria

Your work will be evaluated on completeness, correctness, and analysis quality. **Completeness** means all five tasks are present with code and written analysis. **Correctness** means your cleaning decisions are sound, resampling is applied correctly (train only), and the temporal split prevents leakage. **Analysis quality** means your written sections show understanding of *why* each technique matters, not just *how* to apply it.
