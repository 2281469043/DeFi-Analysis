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

def plot_ground_truth(predictions, target_test_vals):
    # We plot the ground-truth values in blue and the predicted values in red:
    plt.plot(target_test_vals, color = "blue")
    plt.plot(predictions, color = "red")

def plot_difference(predictions, target_test_vals):
    # We plot the difference between our model's predictions and the actual values:
    plt.plot(target_test_vals - predictions)

'''
The linear_regression_model, when applied with data_split1(), yielded suboptimal predictive
performance due to the presence of multiple target variables. Consequently, in order to
enhance predictive accuracy, a logistic regression model will be employed with a new
dataset, data_split2(). This subsequent analysis will primarily concentrate on
forecasting the directional movement of prices, specifically focusing on predicting
whether the price will increase or decrease in the subsequent trading day.
'''

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