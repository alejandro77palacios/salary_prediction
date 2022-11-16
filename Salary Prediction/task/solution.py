import itertools
import os

import pandas as pd
import requests
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_percentage_error as mape
from sklearn.model_selection import train_test_split

# checking ../Data directory presence
if not os.path.exists('../Data'):
    os.mkdir('../Data')

# download data if it is unavailable
if 'data.csv' not in os.listdir('../Data'):
    url = "https://www.dropbox.com/s/3cml50uv7zm46ly/data.csv?dl=1"
    r = requests.get(url, allow_redirects=True)
    open('../Data/data.csv', 'wb').write(r.content)

# read data
data = pd.read_csv('../Data/data.csv')
correlation = data.drop('salary', axis=1).corr()
high_correlation = correlation > 0.2
correlated_features = ['age', 'experience', 'rating']
feature = data.drop('salary', axis=1).values
target = data['salary'].values
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.3, random_state=100)
model = LinearRegression()


def train_no_correlation(features_dropped):
    clean_data = data.drop(features_dropped, axis=1)
    feature = clean_data.drop('salary', axis=1).values
    target = data['salary'].values
    X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.3, random_state=100)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    performance = mape(y_test, y_pred)
    return performance


pairs = itertools.combinations(correlated_features, 2)
results = {}

for var in correlated_features:
    results[var,] = train_no_correlation(var)

for pair in pairs:
    results[pair] = train_no_correlation(list(pair))

best_combination = min(results, key=results.get)
best_performance = round(results[best_combination], 5)
# print(best_performance)

clean_data = data.drop(list(best_combination), axis=1)
feature = clean_data.drop('salary', axis=1).values
target = clean_data['salary'].values
X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.3, random_state=100)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
non_negative = np.where(y_pred < 0, 0, y_pred)
impute_mean = np.where(y_pred < 0, y_train.mean(), y_pred)
performance_1 = mape(y_test, non_negative)
performance_2 = mape(y_test, impute_mean)
result = round(min(performance_1, performance_2), 5)
print(result)
