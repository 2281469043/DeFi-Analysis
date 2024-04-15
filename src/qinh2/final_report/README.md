# Final Report

## Project Overview

This semester, our project team focused on the analysis of blockchain transaction data, applying various machine learning models to predict daily transaction counts. The dataset spans several blockchain platforms, including:

- V2/Mainnet/transactions.rds
- V3/Arbitrum/transactions.rds
- V3/Avalanche/transactions.rds
- V3/Fantom/transactions.rds
- V3/Harmony/transactions.rds
- V3/Optimism/transactions.rds
- V3/Polygon/transactions.rds

## Machine Learning Models

In this project, we employed the following machine learning models for data analysis and prediction:

- Logistic Regression
- K Nearest Neighbors
- Naive Bayes
- Decision Tree
- Random Forest

## Prediction Task

Our primary task was to mine data from each `transactions.rds` file and predict `dailyTransactionCount`. This metric reflects the daily transaction activity on the platform.

## Feature Engineering

During the feature engineering phase, we extracted features from various transaction types to integrate into our model for predicting `dailyTransactionCount`. The specific features include:

- Deposits
- Repayments
- Withdrawals
- Borrowings
- Liquidations
- Mean Prices

## Model Performance

The performance of each model on different datasets is summarized in the table below:

| Dataset                    | Logistic Regression | K Nearest Neighbor | Naive Bayes | Decision Tree | Random Forest |
|----------------------------|---------------------|--------------------|-------------|---------------|---------------|
| V2/Mainnet                 | 47.70%              | 55.59%             | 48.36%      | 51.97%        | 58.55%        |
| V3/Arbitrum                | 47.06%              | 53.68%             | 50.00%      | 48.53%        | 53.68%        |
| V3/Avalanche               | 52.94%              | 47.06%             | 52.21%      | 47.79%        | 44.12%        |
| V3/Fantom                  | 55.38%              | 72.31%             | 53.08%      | 66.15%        | 71.54%        |
| V3/Harmony                 | 58.33%              | 79.17%             | 66.67%      | 72.22%        | 79.17%        |
| V3/Optimism                | 52.13%              | 51.06%             | 48.94%      | 51.60%        | 53.72%        |
| V3/Polygon                 | 49.26%              | 47.79%             | 44.85%      | 47.79%        | 50.74%        |

## Conclusion
From the results, we can observe that the Random Forest and K Nearest Neighbors models generally perform well across different datasets. Notably, the V3/Harmony dataset achieved the highest accuracies with K Nearest Neighbors and Random Forest models, both exceeding 79%. This suggests that for datasets with characteristics similar to V3/Harmony, these two models might provide the most reliable predictions. On the other hand, models like Naive Bayes and Logistic Regression showed more variability and generally lower performance across the datasets.
