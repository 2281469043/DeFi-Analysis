import pandas
import pyreadr
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt


def transaction_v3_Arbitrum():
    transaction = pyreadr.read_r(
        "/data/IDEA_DeFi_Research/Data/Lending_Protocols/Aave/V3/Arbitrum/transactions.rds")
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


def logistic_regression_model(feature_train, feature_test, target_train, target_test):
    from sklearn.linear_model import LogisticRegression
    estimator = LogisticRegression(
        C=1.0, penalty="l2", solver="liblinear", fit_intercept=True, max_iter=1000)
    fit = estimator.fit(feature_train, target_train)
    predictions = fit.predict(feature_test)
    np.linalg.norm(predictions - target_test) / len(target_test)
    target_test_vals = list()
    for data in target_test:
        target_test_vals.append(data)
    target_predict = estimator.predict(feature_test)
    print("-------------------- logistic regression --------------------\n")
    print("The target_predict is:\n", target_predict)
    print("Compare predicted results with actual values:\n",
          target_predict == target_test)
    accuracy = estimator.score(feature_test, target_test) * 100
    print("Accuracy:\n{0:.2f}%".format(accuracy))
    return predictions, target_test_vals, accuracy


def knn_model_gridSearchCV(feature_train, feature_test, target_train, target_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    transfer = StandardScaler()
    feature_train = transfer.fit_transform(feature_train)
    feature_test = transfer.transform(feature_test)
    estimator = KNeighborsClassifier(n_neighbors=3, weights="uniform", algorithm="auto",
                                     leaf_size=30, p=2, metric="minkowski", metric_params=None, n_jobs=None)
    estimator = KNeighborsClassifier()
    parameters_testcase = {"n_neighbors": [3, 5, 7, 9, 11, 13]}
    estimator = GridSearchCV(estimator, parameters_testcase, cv=3)
    estimator.fit(feature_train, target_train)
    predictions = estimator.predict(feature_test)
    np.linalg.norm(predictions - target_test) / len(target_test)
    target_test_vals = list()
    for data in target_test:
        target_test_vals.append(data)
    target_predict = estimator.predict(feature_test)
    print("-------------------- knn with gridSearchCV --------------------\n")
    print("The target_predict is:\n", target_predict)
    print("Compare predicted results with actual values:\n",
          target_predict == target_test)
    accuracy = estimator.score(feature_test, target_test) * 100
    print("Accuracy:\n{0:.2f}%".format(accuracy))
    return predictions, target_test_vals, accuracy


def multinomialNB_model(feature_train, feature_test, target_train, target_test):
    from sklearn.naive_bayes import MultinomialNB
    estimator = MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
    fit = estimator.fit(feature_train, target_train)
    predictions = fit.predict(feature_test)
    np.linalg.norm(predictions - target_test) / len(target_test)
    target_test_vals = list()
    for data in target_test:
        target_test_vals.append(data)
    target_predict = estimator.predict(feature_test)
    print("-------------------- naive bayes --------------------\n")
    print("The target_predict is:\n", target_predict)
    print("Compare predicted results with actual values:\n",
          target_predict == target_test)
    accuracy = estimator.score(feature_test, target_test) * 100
    print("Accuracy:\n{0:.2f}%".format(accuracy))
    return predictions, target_test_vals, accuracy


def decision_tree_model(feature_train, feature_test, target_train, target_test):
    from sklearn.tree import DecisionTreeClassifier
    estimator = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                                       min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0)
    fit = estimator.fit(feature_train, target_train)
    predictions = fit.predict(feature_test)
    np.linalg.norm(predictions - target_test) / len(target_test)
    target_test_vals = list()
    for data in target_test:
        target_test_vals.append(data)
    target_predict = estimator.predict(feature_test)
    print("-------------------- decision tree --------------------\n")
    print("The target_predict is:\n", target_predict)
    print("Compare predicted results with actual values:\n",
          target_predict == target_test)
    accuracy = estimator.score(feature_test, target_test) * 100
    print("Accuracy:\n{0:.2f}%".format(accuracy))
    return predictions, target_test_vals, accuracy


def random_forest_model_gridSearchCV(feature_train, feature_test, target_train, target_test):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    estimator = RandomForestClassifier()
    param_dict = {"n_estimators": [
        120, 200, 300, 500, 800, 1200], "max_depth": [5, 8, 15, 25, 30]}
    estimator = GridSearchCV(estimator, param_grid=param_dict, cv=3)
    estimator.fit(feature_train, target_train)
    predictions = estimator.predict(feature_test)
    np.linalg.norm(predictions - target_test) / len(target_test)
    target_test_vals = list()
    for data in target_test:
        target_test_vals.append(data)
    target_predict = estimator.predict(feature_test)
    print("-------------------- random forest --------------------\n")
    print("The target_predict is:\n", target_predict)
    print("Compare predicted results with actual values:\n",
          target_predict == target_test)
    accuracy = estimator.score(feature_test, target_test) * 100
    print("Accuracy:\n{0:.2f}%".format(accuracy))
    return predictions, target_test_vals, accuracy


machine_learning_model_record = dict()


dailyTransactionCount_v3_Arbitrum = transaction_v3_Arbitrum()
feature_train, feature_test, target_train, target_test = data_split(
    dailyTransactionCount_v3_Arbitrum)
predictions, target_test_vals, accuracy = logistic_regression_model(
    feature_train, feature_test, target_train, target_test)
machine_learning_model_record["logistic_regression"] = accuracy
plot_ground_truth(predictions, target_test_vals)
plot_difference(predictions, target_test_vals)

dailyTransactionCount_v3_Arbitrum = transaction_v3_Arbitrum()
feature_train, feature_test, target_train, target_test = data_split(
    dailyTransactionCount_v3_Arbitrum)
predictions, target_test_vals, accuracy = knn_model_gridSearchCV(
    feature_train, feature_test, target_train, target_test)
machine_learning_model_record["knn_gridSearch"] = accuracy
plot_ground_truth(predictions, target_test_vals)
plot_difference(predictions, target_test_vals)


dailyTransactionCount_v3_Arbitrum = transaction_v3_Arbitrum()
feature_train, feature_test, target_train, target_test = data_split(
    dailyTransactionCount_v3_Arbitrum)
predictions, target_test_vals, accuracy = multinomialNB_model(
    feature_train, feature_test, target_train, target_test)
machine_learning_model_record["naive_bayes"] = accuracy
plot_ground_truth(predictions, target_test_vals)
plot_difference(predictions, target_test_vals)


dailyTransactionCount_v3_Arbitrum = transaction_v3_Arbitrum()
feature_train, feature_test, target_train, target_test = data_split(
    dailyTransactionCount_v3_Arbitrum)
predictions, target_test_vals, accuracy = decision_tree_model(
    feature_train, feature_test, target_train, target_test)
machine_learning_model_record["decision_tree"] = accuracy
plot_ground_truth(predictions, target_test_vals)
plot_difference(predictions, target_test_vals)


dailyTransactionCount_v3_Arbitrum = transaction_v3_Arbitrum()
feature_train, feature_test, target_train, target_test = data_split(
    dailyTransactionCount_v3_Arbitrum)
predictions, target_test_vals, accuracy = random_forest_model_gridSearchCV(
    feature_train, feature_test, target_train, target_test)
machine_learning_model_record["random_forest_gridSearchCV"] = accuracy
plot_ground_truth(predictions, target_test_vals)
plot_difference(predictions, target_test_vals)

for i in machine_learning_model_record.keys():
    print("-" * 60)
    print("The accuracy of model: {} | {:.2f}%\n".format(
        i, machine_learning_model_record.get(i)))
