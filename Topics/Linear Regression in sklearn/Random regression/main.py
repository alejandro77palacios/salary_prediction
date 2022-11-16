from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

diabetes = load_diabetes()

X = diabetes.data
y = diabetes.target
X_train = X[:-50]
y_train = y[:-50]
X_test = X[-50:]
y_test = y[-50:]

regr = LinearRegression()
regr.fit(X_train, y_train)
pred = regr.predict(X_test)

metrica = round(mean_squared_error(y_test, pred), 2)