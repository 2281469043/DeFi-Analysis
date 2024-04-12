import pandas
import pyreadr
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def transaction_v2_mainnet():
    transaction = pyreadr.read_r(
        "/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V2/Mainnet/transactions.rds")
    df = transaction[None]
    df['DateTime'] = df['timestamp'].transform(
        lambda x: datetime.fromtimestamp(x))
    df.head()

    dailyTransactionCount = df.groupby([df['DateTime'].dt.date]).count()

    liquidations = df[df['type'] == "liquidation"]
    liquidations = liquidations.filter(
        items=['DateTime', 'type', 'user', 'principalAmountUSD', 'collateralAmountUSD'], axis='columns')
    liquidations = liquidations.groupby(
        [liquidations['DateTime'].dt.date]).sum()
    print(liquidations)

    borrows = df[df['type'] == "borrow"]
    dailyBorrowsAmountsUSD = borrows.groupby(
        [borrows['DateTime'].dt.date]).sum()
    dailyBorrowsAmountsUSD['amountBorrowsUSD'] = dailyBorrowsAmountsUSD['amountUSD']
    dailyBorrowsAmountsUSD = dailyBorrowsAmountsUSD.filter(
        items=['DateTime', 'amountBorrowsUSD'], axis='columns')
    print(dailyBorrowsAmountsUSD)

    deposits = df[df['type'] == "deposit"]
    dailyDepositsAmountsUSD = deposits.groupby(
        [deposits['DateTime'].dt.date]).sum()
    dailyDepositsAmountsUSD['amountDepositsUSD'] = dailyDepositsAmountsUSD['amountUSD']
    dailyDepositsAmountsUSD = dailyDepositsAmountsUSD.filter(
        items=['DateTime', 'amountDepositsUSD'], axis='columns')
    print(dailyDepositsAmountsUSD)

    
