from sklearn.linear_model import LinearRegression

def linearreg(dataset):
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values
    from sklearn.cross_validation import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1 / 3, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    msg = ""
    for i in range(len(y_pred)):
        msg += "Predicated Value :" + str(y_pred[i]) + "     Expected Value  :" +str(y_test[i]) +"\n"

    return msg

