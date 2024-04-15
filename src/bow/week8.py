import pandas
import pyreadr
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

transaction = pyreadr.read_r("../../data/transactions.rds")
df = transaction[None]
df['DateTime'] = df['timestamp'].transform(lambda x: datetime.fromtimestamp(x))
df.head()

dailyTransactionCount = df.groupby([df['DateTime'].dt.date]).count()

# add borrow type to the dataframe
borrows = df[df['type'] == "borrow"]
dailyBorrowsAmountsUSD = borrows.groupby([borrows['DateTime'].dt.date]).sum()
dailyBorrowsAmountsUSD['amountBorrowsUSD'] = dailyBorrowsAmountsUSD['amountUSD']
dailyBorrowsAmountsUSD = dailyBorrowsAmountsUSD.filter(
    items=['DateTime', 'amountBorrowsUSD'], axis='columns')
print(dailyBorrowsAmountsUSD)

# add deposit type to the dataframe
deposits = df[df['type'] == "deposit"]
dailyDepositsAmountsUSD = deposits.groupby(
    [deposits['DateTime'].dt.date]).sum()
dailyDepositsAmountsUSD['amountDepositsUSD'] = dailyDepositsAmountsUSD['amountUSD']
dailyDepositsAmountsUSD = dailyDepositsAmountsUSD.filter(
    items=['DateTime', 'amountDepositsUSD'], axis='columns')
print(dailyDepositsAmountsUSD)

# add withdraw type to the dataframe
withdraws = df[df['type'] == "withdraw"]
dailyWithdrawsAmountsUSD = withdraws.groupby(
    [withdraws['DateTime'].dt.date]).sum()
dailyWithdrawsAmountsUSD['amountWithdrawsUSD'] = dailyWithdrawsAmountsUSD['amountUSD']
dailyWithdrawsAmountsUSD = dailyWithdrawsAmountsUSD.filter(
    items=['DateTime', 'amountWithdrawsUSD'], axis='columns')
print(dailyWithdrawsAmountsUSD)

dailyTransactionCount = dailyTransactionCount[['id']]
dailyTransactionCount.rename(columns={"id": "transactionCount"}, inplace=True)
print(dailyTransactionCount)
