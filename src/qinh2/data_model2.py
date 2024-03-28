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

# We load the minutely Aave price data here:
aavePrices = pandas.read_csv('/data/IDEA_DeFi_Research/Data/Coin_Prices/Minutely/aavePrices.csv')
# And here, since we want to predict daily prices, we create a new features which is the mean daily price.
aavePrices['DateTime'] = aavePrices['timestamp'].transform(lambda x: datetime.fromtimestamp(x))
dailyMeanPrices = aavePrices.groupby([df['DateTime'].dt.date]).mean()
dailyMeanPrices = dailyMeanPrices[['priceUSD']]
print(dailyMeanPrices)

'''
# Stage 2
In stage 2, our focus shifts towards feature engineering. The **dailyTransactionCount** now
integrates four distinct types of data: deposit, withdraw, borrow, and repay. Meanwhile, 
we are currently deliberating on the best approach to incorporate the "liquidation" type 
data into the **dailyTransactionCount**.
'''
# feature engineering, merge dailyTransactionCount, dailyMeanPrices, dailyBorrowedAmountsUSD
dailyTransactionCount = dailyTransactionCount.merge(dailyMeanPrices, how = "left", on = "DateTime")
dailyTransactionCount = dailyTransactionCount.merge(dailyBorrowedAmountsUSD, how = "left", on = "DateTime")
print(dailyTransactionCount)

machine_learning_model_record = dict()