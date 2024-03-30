import pyreadr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

transaction = pyreadr.read_r(
    '/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/transactions.rds')
df = transaction[None]
# for test if the data is loaded
df.head()

# get information of data
df.info()

# checking for missing values in the data
print(df.isnull().sum())

df = df.select_dtypes(include='number')  # drop non-numeric columns
correlation_matrix = df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix)
plt.title("Correlation Heatmap for transactions.rds")
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, orient="v")
plt.xticks(range(len(df.columns)), df.columns, rotation=90)
plt.xlabel("Features")
plt.ylabel("Value")
plt.title("Boxplot of Features")
plt.show()
