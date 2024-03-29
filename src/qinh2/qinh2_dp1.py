# Author: Hanzhen Qin

import pyreadr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
'''
[1] Checking for Missing Values and get the approximately information of data:

Detecting missing values is crucial for understanding data quality. If there were missing values, specific techniques (like imputation or removal) might be necessary to handle them.

[2] Correlation Heatmap:

Understanding feature correlations is important for feature selection and multicollinearity analysis. A heatmap provides an intuitive representation of the relationships between different features.

[3] Boxplot of Feature Distributions:

Boxplots provide a concise summary of the distribution, helping identify potential outliers and understand the spread of values for each feature.

[4] Histograms of Feature Distributions:

Helping visualize the distribution of individual features. Understanding the distribution is essential for identifying potential outliers, detecting skewness, and gaining insights into the overall data structure.
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
# DataFrame contains non-numeric data that cannot be converted to a float.
df = df.select_dtypes(include='number') # drop non-numeric columns
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