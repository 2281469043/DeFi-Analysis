import pyreadr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

transaction = pyreadr.read_r('/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/transactions.rds')
df = transaction[None]
# for test if the data is loaded
df.head()

# get information of data
df.info()

# checking for missing values in the data
print(df.isnull().sum())
