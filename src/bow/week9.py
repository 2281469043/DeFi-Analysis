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

    repays = df[df['type'] == "repay"]
    dailyRepaysAmountsUSD = repays.groupby([repays['DateTime'].dt.date]).sum()
    dailyRepaysAmountsUSD['amountRepaysUSD'] = dailyRepaysAmountsUSD['amountUSD']
    dailyRepaysAmountsUSD = dailyRepaysAmountsUSD.filter(
        items=['DateTime', 'amountRepaysUSD'], axis='columns')
    print(dailyRepaysAmountsUSD)

    withdraws = df[df['type'] == "withdraw"]
    dailyWithdrawsAmountsUSD = withdraws.groupby(
        [withdraws['DateTime'].dt.date]).sum()
    dailyWithdrawsAmountsUSD['amountWithdrawsUSD'] = dailyWithdrawsAmountsUSD['amountUSD']
    dailyWithdrawsAmountsUSD = dailyWithdrawsAmountsUSD.filter(
        items=['DateTime', 'amountWithdrawsUSD'], axis='columns')
    print(dailyWithdrawsAmountsUSD)

    dailyTransactionCount = dailyTransactionCount[['id']]
    dailyTransactionCount.rename(
        columns={"id": "transactionCount"}, inplace=True)
    print(dailyTransactionCount)

    aavePrices = pandas.read_csv(
        '/data/IDEA_DeFi_Research/Data/Coin_Prices/Minutely/aavePrices.csv')

    aavePrices['DateTime'] = aavePrices['timestamp'].transform(
        lambda x: datetime.fromtimestamp(x))
    dailyMeanPrices = aavePrices.groupby([df['DateTime'].dt.date]).mean()
    dailyMeanPrices = dailyMeanPrices[['priceUSD']]
    print(dailyMeanPrices)

    dailyTransactionCount = dailyTransactionCount.merge(
        dailyMeanPrices, how="left", on="DateTime")
    dailyTransactionCount = dailyTransactionCount.merge(
        dailyBorrowsAmountsUSD, how="left", on="DateTime")
    dailyTransactionCount = dailyTransactionCount.merge(
        dailyDepositsAmountsUSD, how="left", on="DateTime")
    dailyTransactionCount = dailyTransactionCount.merge(
        dailyRepaysAmountsUSD, how="left", on="DateTime")
    dailyTransactionCount = dailyTransactionCount.merge(
        dailyWithdrawsAmountsUSD, how="left", on="DateTime")
    dailyTransactionCount = dailyTransactionCount.merge(
        liquidations, how="left", on="DateTime")

    dailyTransactionCount.fillna(0, inplace=True)
    print(dailyTransactionCount)

    return dailyTransactionCount
