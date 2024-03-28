import pandas
import pyreadr
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
transaction = pyreadr.read_r("/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/transactions.rds")
df = transaction[None]
df['DateTime'] = df['timestamp'].transform(lambda x: datetime.fromtimestamp(x))
df.head()
# We are going to make a super basic linear model to try and predict how the AAVE token's price changes each day.
# This script will engineer one feature to use for this predictive task: dailyTransactionCount.

# To do this, we group the data by the date portion of the DateTime object, 
# and then simply count how many transactions are in each group
dailyTransactionCount = df.groupby([df['DateTime'].dt.date]).count()
# add borrow type to the dataframe
borrows = df[df['type'] == "borrow"]
dailyBorrowedAmountsUSD = borrows.groupby([borrows['DateTime'].dt.date]).sum()
dailyBorrowedAmountsUSD['amountBorrowedUSD'] = dailyBorrowedAmountsUSD['amountUSD']
dailyBorrowedAmountsUSD = dailyBorrowedAmountsUSD.filter(items = ['DateTime', 'amountBorrowedUSD'], axis = 'columns')
print(dailyBorrowedAmountsUSD)

dailyTransactionCount = dailyTransactionCount[['id']]
dailyTransactionCount.rename(columns={"id": "transactionCount"}, inplace = True)
print(dailyTransactionCount)