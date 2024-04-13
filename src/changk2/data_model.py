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

# data_split1 #
def data_split1(data_set):
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
        feature_train, feature_test = X.iloc[train_index, :], X.iloc[test_index,:]
        target_train, target_test = y.iloc[train_index], y.iloc[test_index]
    return [feature_train, feature_test, target_train, target_test]

# data_split2 #
def data_split2(data_set):
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    # We want to use the transactionCount to predict the next day's price. To do this, we "lead" the priceUSD
    # column so in a given row, the transaction count is aligned with the next day's price.
    dailyTransactionCount['priceUSD_lead_1'] = dailyTransactionCount['priceUSD'].shift(-1)
    # We need to drop NA values. One NA value is introduced through this "lead" on the last day in the dataset.
    dailyTransactionCount.dropna(inplace=True)
    # In practice, it is better to predict daily percent price changes rather than predicting literal prices, so we compute the daily
    # percent change here by subtraction tomorrow's price from today's and dividing by today's price.
    dailyTransactionCount['dailyPercentChange'] = (dailyTransactionCount['priceUSD_lead_1'] - dailyTransactionCount['priceUSD']) / dailyTransactionCount['priceUSD']
    # We want to predict the direction of the daily percent change, so we create a new feature which is the sign of the daily percent change.
    dailyTransactionCount['directionOfDailyChange'] = np.sign(dailyTransactionCount['dailyPercentChange'])
    print(dailyTransactionCount)
    tss = TimeSeriesSplit(n_splits = 3)
    X = dailyTransactionCount.drop(labels=['priceUSD_lead_1', 'dailyPercentChange', 'directionOfDailyChange'],axis=1)
    y = dailyTransactionCount['directionOfDailyChange']
    for train_index, test_index in tss.split(dailyTransactionCount):
        feature_train, feature_test = X.iloc[train_index, :], X.iloc[test_index,:]
        target_train, target_test = y.iloc[train_index], y.iloc[test_index]
    return [feature_train, feature_test, target_train, target_test]

def plot_ground_truth(predictions, target_test_vals):
    # We plot the ground-truth values in blue and the predicted values in red:
    plt.plot(target_test_vals, color = "blue")
    plt.plot(predictions, color = "red")
    
def plot_difference(predictions, target_test_vals):
    # We plot the difference between our model's predictions and the actual values:
    plt.plot(target_test_vals - predictions)

def linear_regression_model(feature_train, feature_test, target_train, target_test):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import classification_report
    # We fit a linear model with the train data, where feature_train is our feature matrix and target_train is our target variable
    # Using LinearRegression to classify the data
    estimator = LinearRegression()
    fit = estimator.fit(feature_train, target_train)
    # We compute the predictions for the feature_test features:
    predictions = fit.predict(feature_test)
    # The line below just computes the average accuracy of our predictions:
    np.linalg.norm(predictions - target_test) / len(target_test)
    # All it is intended to do is get
    # the literal target_test values without the associated datetimes, for plotting purposes.
    target_test_vals = list()
    for data in target_test:
        target_test_vals.append(data)
    # evaluate the linear regression model
    # method1: compare the real result and predict result
    target_predict = estimator.predict(feature_test)
    print("target_predict:\n", target_predict)
    print("compare real result and predict result:\n", target_test == target_predict)
    
    # method2: calculate the accuracy
    accuracy = estimator.score(feature_test, target_test)
    print("accuracy: {0:.2f}%\n".format(accuracy * 100))
    return predictions, target_test_vals

# linear regression model
'''
train_set[0] = feature_train
train_set[1] = feature_test
train_set[2] = target_train
train_set[3] = target_test
'''
train_set = list()
train_set = data_split1(dailyTransactionCount) # store all 4 types of data inside
# using the linear_regression_model to make prediction
predictions, target_test_vals = linear_regression_model(train_set[0], train_set[1], train_set[2], train_set[3])
plot_ground_truth(predictions, target_test_vals)
plot_difference(predictions, target_test_vals)

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
    # classification report for the logistic regression model
    report = classification_report(target_test, target_predict, labels=[2, 4], target_names=["Up", "Down"], zero_division=1)
    print(report)
    return predictions, target_test_vals

# logistic regression model
'''
train_set[0] = feature_train
train_set[1] = feature_test
train_set[2] = target_train
train_set[3] = target_test
'''
train_set = list()
train_set = data_split2(dailyTransactionCount) # store all 4 types of data inside
# using the linear_regression_model to make prediction
predictions, target_test_vals = logistic_regression_model(train_set[0], train_set[1], train_set[2], train_set[3])
plot_ground_truth(predictions, target_test_vals)
plot_difference(predictions, target_test_vals)

def knn_model(feature_train, feature_test, target_train, target_test):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import StandardScaler
    # transfer = StandardScaler()
    # # train data standardization
    # feature_train = transfer.fit_transform(feature_train)
    # feature_test = transfer.transform(feature_test)
    # estimator = KNeighborsClassifier(n_neighbors=3, weights="uniform", algorithm="auto", leaf_size=30, p=2, metric="minkowski", metric_params=None, n_jobs=None)
    # # KNN model optimization
    # estimator = KNeighborsClassifier()
    # parameters_testcase = {"n_neighbors": [3, 5, 7, 9, 11, 13]} # 超参数
    # estimator = GridSearchCV(estimator, parameters_testcase, cv=3) # 交叉验证
    # estimator.fit(feature_train, target_train)
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
    return predictions, target_test_vals

# K_Nearest_Neighbors model
'''
train_set[0] = feature_train
train_set[1] = feature_test
train_set[2] = target_train
train_set[3] = target_test
'''
train_set = list()
train_set = data_split2(dailyTransactionCount) # store all 4 types of data inside
# using the linear_regression_model to make prediction
predictions, target_test_vals = knn_model(train_set[0], train_set[1], train_set[2], train_set[3])
plot_ground_truth(predictions, target_test_vals)
plot_difference(predictions, target_test_vals)
    
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
    return predictions, target_test_vals

# MultinomialNB model
'''
train_set[0] = feature_train
train_set[1] = feature_test
train_set[2] = target_train
train_set[3] = target_test
'''
train_set = list()
train_set = data_split2(dailyTransactionCount) # store all 4 types of data inside
# using the naive bayes classifier to make prediction
predictions, target_test_vals = multinomialNB_model(train_set[0], train_set[1], train_set[2], train_set[3])
plot_ground_truth(predictions, target_test_vals)
plot_difference(predictions, target_test_vals)

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
    return predictions, target_test_vals

# Decision Tree model
'''
train_set[0] = feature_train
train_set[1] = feature_test
train_set[2] = target_train
train_set[3] = target_test
'''
train_set = list()
train_set = data_split2(dailyTransactionCount) # store all 4 types of data inside
# using the decision tree classifier to make prediction
predictions, target_test_vals = decision_tree_model(train_set[0], train_set[1], train_set[2], train_set[3])
plot_ground_truth(predictions, target_test_vals)
plot_difference(predictions, target_test_vals)

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
    return predictions, target_test_vals

# Random Forest model
'''
train_set[0] = feature_train
train_set[1] = feature_test
train_set[2] = target_train
train_set[3] = target_test
'''
train_set = list()
train_set = data_split2(dailyTransactionCount) # store all 4 types of data inside
# using the random forest classifier to make prediction
predictions, target_test_vals = random_forest_model(train_set[0], train_set[1], train_set[2], train_set[3])
plot_ground_truth(predictions, target_test_vals)
plot_difference(predictions, target_test_vals)
    
'''
The linear_regression_model, when applied with data_split1(), yielded suboptimal predictive performance due to the
presence of multiple target variables. Consequently, in order to enhance predictive accuracy, a logistic
regression model will be employed with a new dataset, data_split2(). This subsequent analysis will primarily
concentrate on forecasting the directional movement of prices, specifically focusing on predicting whether the
price will increase or decrease in the subsequent trading day.
'''

'''
For predict the directionOfDailyChange:

logistic regression: 62.5% (Best)

k nearest neighbors: 56.18%
(I try the standardization and gridsearch, but the result has lower possibility then only use k_neighbors=3)

naive Bayes: 52.81%

decision tree: 56.23%

random forest: 56.23%
'''


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
    print("-------------------- random forest --------------------\n")
    print("The target_predict is:\n", target_predict)
    print("Compare predicted results with actual values:\n", target_predict == target_test)
    accuracy = estimator.score(feature_test, target_test) * 100
    print("Accuracy:\n{0:.2f}%".format(accuracy))
    return predictions, target_test_vals, accuracy

# decision tree run
print("-------------------- data of v2 mainnet --------------------\n")
dailyTransactionCount_v2_mainnet = transaction_v2_mainnet()
print("-------------------- data of v3 fantom --------------------\n")
dailyTransactionCount_v3_fantom = transaction_v3_fantom()
# print("-------------------- data of v2 polygon --------------------\n")
# dailyTransactionCount_v2_polygon = transaction_v2_polygon()

# v2_mainnet
feature_train, feature_test, target_train, target_test = data_split2(dailyTransactionCount_v2_mainnet)
predictions, target_test_vals, accuracy = random_forest_model(feature_train, feature_test, target_train, target_test)
# make record for the accuracy
machine_learning_model_record["random_forest_v2_mainnet"] = accuracy
plot_ground_truth(predictions, target_test_vals)
plot_difference(predictions, target_test_vals)

# v3_fantom
feature_train, feature_test, target_train, target_test = data_split2(dailyTransactionCount_v3_fantom)
predictions, target_test_vals, accuracy = random_forest_model(feature_train, feature_test, target_train, target_test)
# make record for the accuracy
machine_learning_model_record["random_forest_v3_fantom"] = accuracy
plot_ground_truth(predictions, target_test_vals)
plot_difference(predictions, target_test_vals)

# # v2_polygon
# feature_train, feature_test, target_train, target_test = data_split2(dailyTransactionCount_v2_polygon)
# predictions, target_test_vals, accuracy = random_forest_model(feature_train, feature_test, target_train, target_test)
# # make record for the accuracy
# machine_learning_model_record["random_forest_v2_polygon"] = accuracy
plot_ground_truth(predictions, target_test_vals)
plot_difference(predictions, target_test_vals)

# model accuracy record
for i in machine_learning_model_record.keys():
    print("The accuracy of model: {} is {:.2f}%\n".format(i, machine_learning_model_record.get(i)))