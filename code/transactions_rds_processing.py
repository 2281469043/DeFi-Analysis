# Author: Hanzhen Qin

import pyreadr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
'''
[1] Histograms of Feature Distributions:

Helping visualize the distribution of individual features. Understanding the distribution is essential for identifying potential outliers, detecting skewness, and gaining insights into the overall data structure.

[2] Boxplot of Feature Distributions:

Boxplots provide a concise summary of the distribution, helping identify potential outliers and understand the spread of values for each feature.

[3] Checking for Missing Values and get the approximately information of data:

Detecting missing values is crucial for understanding data quality. If there were missing values, specific techniques (like imputation or removal) might be necessary to handle them.

[4] Correlation Heatmap:

Understanding feature correlations is important for feature selection and multicollinearity analysis. A heatmap provides an intuitive representation of the relationships between different features.
'''
transaction = pyreadr.read_r('/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/transactions.rds')
df = transaction[None]
# for test if the data is loaded
df.head()

# get information of data
df.info()

# checking for missing values in the data
print(df.isnull().sum())

# draw the corelation heatmap
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix)
plt.title("Correlation Heatmap for transactions.rds")
plt.show()

# draw the boxplot of data
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, orient="v")
plt.xticks(range(len(df.columns)), df.columns, rotation=90)
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Boxplot of Features")
plt.show()

# draw the Histograms of Feature Distributions
for feature_name in df.columns:
    plt.figure(figsize=(6, 6))
    sns.distplot(df[feature_name], kde=True)
    plt.xlabel(feature_name)
    plt.ylabel("Count")
    plt.title("Histogram of {}".format(feature_name))
    plt.show()