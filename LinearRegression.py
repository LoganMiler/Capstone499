import pandas as pd
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np
import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn import preprocessing, cross_validation, svm
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
import math
from functions import cost



# df = pd.read_csv('EOG.csv',header=0,index_col='Date',parse_dates=True)
# newPrices = pd.read_csv('EOG_updates_new.csv',header=0,index_col='Date',parse_dates=True)

df = pd.read_csv('APC.csv', header=0, index_col='Date', parse_dates=True)
newPrices = pd.read_csv('APC_updates_new.csv', header=0, index_col='Date', parse_dates=True)

# newPrices.dropna()
# newPrices = newPrices[pd.notnull(newPrices['Close'])]
df = df[['Close']]
# print('df: ', df.tail())
newPrices = newPrices[['Close']]
# print('newPrices: ', newPrices)


forecast_out = int(10)
# make sure our new Prices array fits with our prediction for plotting (for values other than 10)
newPrices = newPrices.iloc[0:forecast_out]
print('newPrices: ', newPrices)
df['Label'] = df.shift(-forecast_out)
# df = df.iloc[:-3] #needed for APA updates because we deleted the last three rows-->need to eliminate NaN
print('df with label: ', df)

# Define features and labels
X = np.array(df.drop(['Label'], 1))
print('X: ', X)
X = preprocessing.scale(X)

X_forecast = X[-forecast_out:]
print('X_forecast: ', X_forecast)
print(X_forecast.shape)
X = X[:-forecast_out]
print('X: ', X)
print(X.shape)

y = np.array(df['Label'])
print('initial y: ', y)
print(y.shape)
y = y[:-forecast_out]
print('y new: ', y)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = LinearRegression()
clf.fit(X_train, y_train)

confidence = clf.score(X_test, y_test)
print("confidence: ", confidence)

forecast_prediction = clf.predict(X_forecast)
print('forecast_prediction: ', forecast_prediction)
print('type: ', type(forecast_prediction))
print(forecast_prediction.shape)
# prediction = pd.DataFrame(data=forecast_prediction)
# print(type(prediction))
# print(prediction)
# print(newPrices)

predicted_Vals = np.zeros((len(df)))
predicted_Vals.fill(np.nan)
predicted_Vals[-forecast_out:] = forecast_prediction
df['Predicted Values'] = predicted_Vals
true_future_Prices = np.zeros((len(df)))
true_future_Prices.fill(np.nan)
true_future_Prices[-forecast_out:] = newPrices['Close']
df['Real Prices'] = true_future_Prices
# print('true future prices: ', true_future_Prices)
true_Prices = true_future_Prices[-forecast_out:]
# print('True Prices: ', true_Prices)
print('Forecast pred: ', forecast_prediction)
format_col = df['Close']
format_col[-forecast_out:] = np.nan
df['Close'] = format_col
print('Output dataframe: ', df)
# calculating error
rmse = sqrt(mean_squared_error(forecast_prediction, true_Prices))
print('Test RMSE: %.4f' % rmse)

# plotting
# df['Close'].plot()
# df['Predicted Values'].plot()
# df['Real Prices'].plot()
# plt.legend()
# plt.title('Stock Price History Versus Prediction')
# plt.ylabel('Stock Price($)')
# plt.xlabel('Date')
# plt.show()

# the array we predicted
array = forecast_prediction
# our true values
true_array = df['Real Prices'].iloc[-(forecast_out+1):]
true_array = true_array.values
# initial invest moneys
init_invest = 10000
#initializing array with "leftover" money holdings
money = np.zeros(len(array)-1)
#initializing array to track alue of our stock hodlings:
holdings = np.zeros(len(array)-1)

cost(array, true_array, init_invest, )