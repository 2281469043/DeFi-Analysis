import pandas
import pyreadr
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def transaction_v3_Optimism():
    transaction = pyreadr.read_r(
        "/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V3/Optimism/transactions.rds")
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

    borrows = df[df['type'] == "borrow"]
    dailyBorrowsAmountsUSD = borrows.groupby(
        [borrows['DateTime'].dt.date]).sum()
    dailyBorrowsAmountsUSD['amountBorrowsUSD'] = dailyBorrowsAmountsUSD['amountUSD']
    dailyBorrowsAmountsUSD = dailyBorrowsAmountsUSD.filter(
        items=['DateTime', 'amountBorrowsUSD'], axis='columns')

    deposits = df[df['type'] == "deposit"]
    dailyDepositsAmountsUSD = deposits.groupby(
        [deposits['DateTime'].dt.date]).sum()
    dailyDepositsAmountsUSD['amountDepositsUSD'] = dailyDepositsAmountsUSD['amountUSD']
    dailyDepositsAmountsUSD = dailyDepositsAmountsUSD.filter(
        items=['DateTime', 'amountDepositsUSD'], axis='columns')

    repays = df[df['type'] == "repay"]
    dailyRepaysAmountsUSD = repays.groupby([repays['DateTime'].dt.date]).sum()
    dailyRepaysAmountsUSD['amountRepaysUSD'] = dailyRepaysAmountsUSD['amountUSD']
    dailyRepaysAmountsUSD = dailyRepaysAmountsUSD.filter(
        items=['DateTime', 'amountRepaysUSD'], axis='columns')

    withdraws = df[df['type'] == "withdraw"]
    dailyWithdrawsAmountsUSD = withdraws.groupby(
        [withdraws['DateTime'].dt.date]).sum()
    dailyWithdrawsAmountsUSD['amountWithdrawsUSD'] = dailyWithdrawsAmountsUSD['amountUSD']
    dailyWithdrawsAmountsUSD = dailyWithdrawsAmountsUSD.filter(
        items=['DateTime', 'amountWithdrawsUSD'], axis='columns')

    dailyTransactionCount = dailyTransactionCount[['id']]
    dailyTransactionCount.rename(
        columns={"id": "transactionCount"}, inplace=True)

    aavePrices = pandas.read_csv(
        '/data/IDEA_DeFi_Research/Data/Coin_Prices/Minutely/aavePrices.csv')
    aavePrices['DateTime'] = aavePrices['timestamp'].transform(
        lambda x: datetime.fromtimestamp(x))
    dailyMeanPrices = aavePrices.groupby([df['DateTime'].dt.date]).mean()
    dailyMeanPrices = dailyMeanPrices[['priceUSD']]

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


def data_split(data_set):
    from sklearn.model_selection import TimeSeriesSplit
    data_set['priceUSD_lead_1'] = data_set['priceUSD'].shift(-1)
    data_set.dropna(inplace=True)
    data_set['dailyPercentChange'] = (
        data_set['priceUSD_lead_1'] - data_set['priceUSD']) / data_set['priceUSD']
    data_set['directionOfDailyChange'] = np.sign(
        data_set['dailyPercentChange'])
    tss = TimeSeriesSplit(n_splits=3)
    X = data_set.drop(
        labels=['priceUSD_lead_1', 'dailyPercentChange', 'directionOfDailyChange'], axis=1)
    y = data_set['directionOfDailyChange']
    for train_index, test_index in tss.split(data_set):
        feature_train, feature_test = X.iloc[train_index,
                                             :], X.iloc[test_index, :]
        target_train, target_test = y.iloc[train_index], y.iloc[test_index]
    return [feature_train, feature_test, target_train, target_test]


def plot_ground_truth(predictions, target_test_vals):
    plt.plot(target_test_vals, color="lightblue")
    plt.plot(predictions, color="lightpink")


def plot_difference(predictions, y_test_vals):
    plt.plot(y_test_vals - predictions)
