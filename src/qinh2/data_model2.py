# Author: Hanzhen Qin
# Server

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
dailyTransactionCount = dailyTransactionCount.merge(dailyBorrowsAmountsUSD, how = "left", on = "DateTime")
dailyTransactionCount = dailyTransactionCount.merge(dailyDepositsAmountsUSD, how = "left", on = "DateTime")
dailyTransactionCount = dailyTransactionCount.merge(dailyRepaysAmountsUSD, how = "left", on = "DateTime")
dailyTransactionCount = dailyTransactionCount.merge(dailyWithdrawsAmountsUSD, how = "left", on = "DateTime")
print(dailyTransactionCount)

machine_learning_model_record = dict()

def data_split2(data_set):
    from sklearn.model_selection import TimeSeriesSplit
    # We want to use the transactionCount to predict the next day's price. To do this, we "lead" the priceUSD
    # column so in a given row, the transaction count is aligned with the next day's price.
    data_set['priceUSD_lead_1'] = data_set['priceUSD'].shift(-1)
    # We need to drop NA values. One NA value is introduced through this "lead" on the last day in the dataset.
    data_set.dropna(inplace=True)
    # In practice, it is better to predict daily percent price changes rather than predicting literal prices, so we compute the daily
    # percent change here by subtraction tomorrow's price from today's and dividing by today's price.
    data_set['dailyPercentChange'] = (data_set['priceUSD_lead_1'] - data_set['priceUSD']) / data_set['priceUSD']
    # We want to predict the direction of the daily percent change, so we create a new feature which is the sign of the daily percent change.
    data_set['directionOfDailyChange'] = np.sign(data_set['dailyPercentChange'])
    print(data_set)
    tss = TimeSeriesSplit(n_splits = 3)
    X = data_set.drop(labels=['priceUSD_lead_1', 'dailyPercentChange', 'directionOfDailyChange'],axis=1)
    y = data_set['directionOfDailyChange']
    for train_index, test_index in tss.split(data_set):
        feature_train, feature_test = X.iloc[train_index, :], X.iloc[test_index,:]
        target_train, target_test = y.iloc[train_index], y.iloc[test_index]
    return [feature_train, feature_test, target_train, target_test]

def plot_ground_truth(predictions, target_test_vals):
    # We plot the ground-truth values in blue and the predicted values in red:
    plt.plot(target_test_vals, color = "blue")
    plt.plot(predictions, color = "red")

def plot_difference(predictions, y_test_vals):
    # We plot the difference between our model's predictions and the actual values:
    plt.plot(y_test_vals - predictions)
    
# ----------------- Logistic Regression model ----------------- #
def logistic_regression_model(feature_train, feature_test, target_train, target_test):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    estimator = LogisticRegression(C = 1.0, penalty = "l2", solver = "liblinear", fit_intercept=True, max_iter=1000)
    fit = estimator.fit(feature_train, target_train)
    # We compute the predictions for the feature_test features:
    predictions = fit.predict(feature_test)
    # The line below just computes the average accuracy of our predictions:
    np.linalg.norm(predictions - target_test) / len(target_test)
    target_test_vals = list()
    for data in target_test:
        target_test_vals.append(data)
    # model evaluation
    target_predict = estimator.predict(feature_test)
    print("The target_predict is:\n", target_predict)
    print("Compare predicted results with actual values:\n", target_predict == target_test)
    print("Accuracy:\n{0:.2f}%".format(estimator.score(feature_test, target_test) * 100))
    # make record for the accuracy
    machine_learning_model_record["logistic_regression"] = estimator.score(feature_test, target_test) * 100
    # classification report for the logistic regression model
    report = classification_report(target_test, target_predict, labels=[2, 4], target_names=["Up", "Down"], zero_division=1)
    print(report)
    return predictions, target_test_vals

# logistic_regression run
feature_train, feature_test, target_train, target_test = data_split2(dailyTransactionCount)
predictions, target_test_vals = logistic_regression_model(feature_train, feature_test, target_train, target_test)

plot_ground_truth(predictions, target_test_vals)

plot_difference(predictions, target_test_vals)

# ----------------- KNN model ----------------- #
def knn_model(feature_train, feature_test, target_train, target_test):
    from sklearn.neighbors import KNeighborsClassifier
    estimator = KNeighborsClassifier(n_neighbors=3)
    estimator.fit(feature_train, target_train)
    # We compute the predictions for the feature_test features:
    predictions = estimator.predict(feature_test)
    # The line below just computes the average accuracy of our predictions:
    np.linalg.norm(predictions - target_test) / len(target_test)
    target_test_vals = list()
    for data in target_test:
        target_test_vals.append(data)
    # model evaluation
    target_predict = estimator.predict(feature_test)
    print("The target_predict is:\n", target_predict)
    print("Compare predicted results with actual values:\n", target_predict == target_test)
    print("Accuracy:\n{0:.2f}%".format(estimator.score(feature_test, target_test) * 100))
    # make record for the accuracy
    machine_learning_model_record["knn"] = estimator.score(feature_test, target_test) * 100
    return predictions, target_test_vals

# knn_model run
feature_train, feature_test, target_train, target_test = data_split2(dailyTransactionCount)
predictions, target_test_vals = knn_model(feature_train, feature_test, target_train, target_test)

plot_ground_truth(predictions, target_test_vals)

plot_difference(predictions, target_test_vals)

# ----------------- KNN model gridSearchCV ----------------- #
def knn_model_gridSearchCV(feature_train, feature_test, target_train, target_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    transfer = StandardScaler()
    # train data standardization
    feature_train = transfer.fit_transform(feature_train)
    feature_test = transfer.transform(feature_test)
    estimator = KNeighborsClassifier(n_neighbors=3, weights="uniform", algorithm="auto", leaf_size=30, p=2, metric="minkowski", metric_params=None, n_jobs=None)
    # KNN model optimization
    estimator = KNeighborsClassifier()
    parameters_testcase = {"n_neighbors": [3, 5, 7, 9, 11, 13]}
    estimator = GridSearchCV(estimator, parameters_testcase, cv=3)
    estimator.fit(feature_train, target_train)
    # We compute the predictions for the feature_test features:
    predictions = estimator.predict(feature_test)
    # The line below just computes the average accuracy of our predictions:
    np.linalg.norm(predictions - target_test) / len(target_test)
    target_test_vals = list()
    for data in target_test:
        target_test_vals.append(data)
    # model evaluation
    target_predict = estimator.predict(feature_test)
    print("The target_predict is:\n", target_predict)
    print("Compare predicted results with actual values:\n", target_predict == target_test)
    print("Accuracy:\n{0:.2f}%".format(estimator.score(feature_test, target_test) * 100))
    # make record for the accuracy
    machine_learning_model_record["knn_gridSearch"] = estimator.score(feature_test, target_test) * 100
    return predictions, target_test_vals

# knn_model_gridSearchCV run
feature_train, feature_test, target_train, target_test = data_split2(dailyTransactionCount)
predictions, target_test_vals = knn_model_gridSearchCV(feature_train, feature_test, target_train, target_test)

plot_ground_truth(predictions, target_test_vals)

plot_difference(predictions, target_test_vals)

# ----------------- naive bayes model ----------------- #
def multinomialNB_model(feature_train, feature_test, target_train, target_test):
    from sklearn.naive_bayes import MultinomialNB
    estimator = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    fit = estimator.fit(feature_train, target_train)
    # We compute the predictions for the feature_test features:
    predictions = fit.predict(feature_test)
    # The line below just computes the average accuracy of our predictions:
    np.linalg.norm(predictions - target_test) / len(target_test)
    target_test_vals = list()
    for data in target_test:
        target_test_vals.append(data)
    # model evaluation
    target_predict = estimator.predict(feature_test)
    print("The target_predict is:\n", target_predict)
    print("Compare predicted results with actual values:\n", target_predict == target_test)
    print("Accuracy:\n{0:.2f}%".format(estimator.score(feature_test, target_test) * 100))
    # make record for the accuracy
    machine_learning_model_record["naive_bayes"] = estimator.score(feature_test, target_test) * 100
    return predictions, target_test_vals

# naive bayes classifier
feature_train, feature_test, target_train, target_test = data_split2(dailyTransactionCount)
predictions, target_test_vals = multinomialNB_model(feature_train, feature_test, target_train, target_test)

plot_ground_truth(predictions, target_test_vals)

plot_difference(predictions, target_test_vals)

# ----------------- decision tree model ----------------- #
def decision_tree_model(feature_train, feature_test, target_train, target_test):
    from sklearn.tree import DecisionTreeClassifier
    estimator = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0)
    fit = estimator.fit(feature_train, target_train)
    # We compute the predictions for the feature_test features:
    predictions = fit.predict(feature_test)
    # The line below just computes the average accuracy of our predictions:
    np.linalg.norm(predictions - target_test) / len(target_test)
    target_test_vals = list()
    for data in target_test:
        target_test_vals.append(data)
    # model evaluation
    target_predict = estimator.predict(feature_test)
    print("The target_predict is:\n", target_predict)
    print("Compare predicted results with actual values:\n", target_predict == target_test)
    print("Accuracy:\n{0:.2f}%".format(estimator.score(feature_test, target_test) * 100))
    # make record for the accuracy
    machine_learning_model_record["decision_tree"] = estimator.score(feature_test, target_test) * 100
    return predictions, target_test_vals

# decision tree model
feature_train, feature_test, target_train, target_test = data_split2(dailyTransactionCount)
predictions, target_test_vals = decision_tree_model(feature_train, feature_test, target_train, target_test)

plot_ground_truth(predictions, target_test_vals)

plot_difference(predictions, target_test_vals)

# ----------------- random forest model ----------------- #
def random_forest_model(feature_train, feature_test, target_train, target_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    estimator = RandomForestClassifier()
    param_dict = {"n_estimators": [120,200,300,500,800,1200], "max_depth": [5,8,15,25,30]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
    estimator.fit(feature_train, target_train)
    # We compute the predictions for the feature_test features:
    predictions = estimator.predict(feature_test)
    # The line below just computes the average accuracy of our predictions:
    np.linalg.norm(predictions - target_test) / len(target_test)
    target_test_vals = list()
    for data in target_test:
        target_test_vals.append(data)
    # model evaluation
    target_predict = estimator.predict(feature_test)
    print("The target_predict is:\n", target_predict)
    print("Compare predicted results with actual values:\n", target_predict == target_test)
    print("Accuracy:\n{0:.2f}%".format(estimator.score(feature_test, target_test) * 100))
    # make record for the accuracy
    machine_learning_model_record["decision_tree"] = estimator.score(feature_test, target_test) * 100
    return predictions, target_test_vals

# random forest model
feature_train, feature_test, target_train, target_test = data_split2(dailyTransactionCount)
predictions, target_test_vals = random_forest_model(feature_train, feature_test, target_train, target_test)

plot_ground_truth(predictions, target_test_vals)

plot_difference(predictions, target_test_vals)

# ----------------- model accuracy model ----------------- #
for i in machine_learning_model_record.keys():
    print("The accuracy of model: {} is {:.2f}%\n".format(i, machine_learning_model_record.get(i)))