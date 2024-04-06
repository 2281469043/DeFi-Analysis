
import pandas
import pyreadr
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

def transaction_v2_mainnet():
    transaction = pyreadr.read_r("/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/transactions.rds")
    df = transaction[None]
    df['DateTime'] = df['timestamp'].transform(lambda x: datetime.fromtimestamp(x))
    df.head()
    # We are going to make a super basic linear model to try and predict how the AAVE token's price changes each day.
    # This script will engineer one feature to use for this predictive task: dailyTransactionCount.

    # To do this, we group the data by the date portion of the DateTime object, 
    # and then simply count how many transactions are in each group
    dailyTransactionCount = df.groupby([df['DateTime'].dt.date]).count()
    
    # Add the liquidations type to the dataframe
    liquidations = df[df['type'] == "liquidation"]
    liquidations = liquidations.filter(items = ['DateTime', 'type', 'user', 'principalAmountUSD', 'collateralAmountUSD'], axis = 'columns')
    liquidations = liquidations.groupby([liquidations['DateTime'].dt.date]).sum()
    print(liquidations)
    
    # add borrow type to the dataframe
    borrows = df[df['type'] == "borrow"]
    dailyBorrowsAmountsUSD = borrows.groupby([borrows['DateTime'].dt.date]).sum()
    dailyBorrowsAmountsUSD['amountBorrowsUSD'] = dailyBorrowsAmountsUSD['amountUSD']
    dailyBorrowsAmountsUSD = dailyBorrowsAmountsUSD.filter(items = ['DateTime', 'amountBorrowsUSD'], axis = 'columns')
    print(dailyBorrowsAmountsUSD)
    
    # add deposit type to the dataframe
    deposits = df[df['type'] == "deposit"]
    dailyDepositsAmountsUSD = deposits.groupby([deposits['DateTime'].dt.date]).sum()
    dailyDepositsAmountsUSD['amountDepositsUSD'] = dailyDepositsAmountsUSD['amountUSD']
    dailyDepositsAmountsUSD = dailyDepositsAmountsUSD.filter(items = ['DateTime', 'amountDepositsUSD'], axis = 'columns')
    print(dailyDepositsAmountsUSD)
    
    # add repay type to the dataframe
    repays = df[df['type'] == "repay"]
    dailyRepaysAmountsUSD = repays.groupby([repays['DateTime'].dt.date]).sum()
    dailyRepaysAmountsUSD['amountRepaysUSD'] = dailyRepaysAmountsUSD['amountUSD']
    dailyRepaysAmountsUSD = dailyRepaysAmountsUSD.filter(items = ['DateTime', 'amountRepaysUSD'], axis = 'columns')
    print(dailyRepaysAmountsUSD)
    
    # add withdraw type to the dataframe
    withdraws = df[df['type'] == "withdraw"]
    dailyWithdrawsAmountsUSD = withdraws.groupby([withdraws['DateTime'].dt.date]).sum()
    dailyWithdrawsAmountsUSD['amountWithdrawsUSD'] = dailyWithdrawsAmountsUSD['amountUSD']
    dailyWithdrawsAmountsUSD = dailyWithdrawsAmountsUSD.filter(items = ['DateTime', 'amountWithdrawsUSD'], axis = 'columns')
    print(dailyWithdrawsAmountsUSD)
    
    dailyTransactionCount = dailyTransactionCount[['id']]
    dailyTransactionCount.rename(columns={"id": "transactionCount"}, inplace = True)
    print(dailyTransactionCount)