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
dailyTransactionCount = dailyTransactionCount.merge(dailyMeanPrices, left_index = True, right_index = True)
print(dailyTransactionCount)

# data_split takes in a dataset and returns a list of 4 dataframes: X_train, X_test, y_train, y_test #
def data_split(data_set):
    from sklearn.model_selection import TimeSeriesSplit
    # We want to use the transactionCount to predict the next day's price. To do this, we "lead" the priceUSD
    # column so in a given row, the transaction count is aligned with the next day's price.
    data_set['priceUSD_lead_1'] = data_set['priceUSD'].shift(-1)
    # We need to drop NA values. One NA value is introduced through this "lead" on the last day in the dataset.
    data_set.dropna(inplace=True)
    # In practice, it is better to predict daily percent price changes rather than predicting literal prices, so we compute the daily
    # percent change here by subtraction tomorrow's price from today's and dividing by today's price.
    data_set['dailyPercentChange'] = (data_set['priceUSD_lead_1'] - data_set['priceUSD']) / data_set['priceUSD']
    tss = TimeSeriesSplit(n_splits = 3)
    X = data_set.drop(labels=['priceUSD_lead_1', 'dailyPercentChange'],axis=1)
    y = data_set['dailyPercentChange']
    for train_index, test_index in tss.split(data_set):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index,:]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    return [X_train, X_test, y_train, y_test]

# linear_regression_model takes in the 4 dataframes from data_split and returns a list of predictions and the literal y_test values #
def linear_regression_model(X_train, X_test, y_train, y_test):
    from sklearn.linear_model import LinearRegression
    # We fit a linear model with the train data, where X_train is our feature matrix and y_train is our target variable
    # using LinearRegression to classify the data
    estimator = LinearRegression()
    fit = estimator.fit(X_train, y_train)
    # We compute the predictions for the X_test features:
    predictions = fit.predict(X_test)
    # The line below just computes the average accuracy of our predictions:
    np.linalg.norm(predictions - y_test) / len(y_test)
    # This for-loop is super ugly and represents my inexperience with Python. All it is intended to do is get
    # the literal y_test values without the associated datetimes, for plotting purposes.
    y_test_vals = list()
    for i in y_test:
        y_test_vals.append(i)
    return predictions, y_test_vals

# plot_ground_truth takes in the predictions and the literal y_test values #
def plot_ground_truth(predictions, y_test_vals):
    # We plot the ground-truth values in blue and the predicted values in red:
    plt.plot(y_test_vals, color = "blue")
    plt.plot(predictions, color = "red")
    
# plot_difference takes in the predictions and the literal y_test values #
def plot_difference(predictions, y_test_vals):
    # We plot the difference between our model's predictions and the actual values:
    plt.plot(y_test_vals - predictions)