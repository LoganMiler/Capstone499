import math
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from math import sqrt
import numpy as np
import pandas as pd
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from functions import cost


#load data
def parser(x):
    return datetime.strptime(x, '%m/%d/%Y')
# load dataset
#dataset = read_csv('APC_new1.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser= parser)
#newPrices = read_csv('APC_updates_new.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser= parser)
#print(newPrices)
#print(type(dataset))
#values = dataset.values

#print(dataset.head())

######PLOTTING OUR INDIVIDUAL FEATURES
#specify columns to plot
#groups = [0, 1, 2, 3, 4, 5,6]
#i = 1
#plot each column
#pyplot.figure()
#for group in groups:
    #pyplot.subplot(len(groups), 1, i)
    #pyplot.plot(values[:,group])
    #pyplot.title(dataset.columns[group], y=0.5, loc='right')
    #i += 1
#pyplot.show()

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


# load dataset
dataset = read_csv('ADI.csv', header=0, index_col=0)
values = dataset.values
# integer encode direction
encoder = LabelEncoder()
values[:, 4] = encoder.fit_transform(values[:, 4])
# ensure all data is float
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#NEW: specify number of lag days
n_days = 365
n_features = 7
forecast_out = 10
# frame as supervised learning
reframed = series_to_supervised(scaled, n_days, 1)
reframed[['var7(t)']] = reframed[['var7(t)']].shift(-forecast_out)
reframed[['var7(t)']] = reframed[['var7(t)']].iloc[:-forecast_out]

print('Reframed shape: ', reframed.shape)
print(reframed)

# split into train and test sets
values = reframed.values
#n_train_days = int(0.8*len(reframed['var1(t)']))
n_train_days = int(len(reframed['var1(t)']))-forecast_out
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
n_obs = n_days * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)
# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
test_X = test_X.reshape((test_X.shape[0], n_days, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=1, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.xlabel('Epoch')
#pyplot.ylabel('Loss')
#pyplot.title('Loss Throughout Training')
#pyplot.show()

# make a prediction
#newPrices_vals = newPrices.values
#newPrices = newPrices_vals.reshape((newPrices_vals.shape[0], n_days, n_features))
yhat = model.predict(test_X)
test_X = test_X.reshape((test_X.shape[0], n_days*n_features))
# invert scaling for forecast
inv_yhat = concatenate((yhat, test_X[:, -6:]), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:,0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = concatenate((test_y, test_X[:, -6:]), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
# calculate RMSE
rmse = sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.4f' % rmse)
#print(len(inv_y))
#print(len(inv_yhat))
#print(inv_yhat)

forecast_prediction = inv_yhat
print(forecast_prediction.shape)

predicted_Vals = np.zeros((len(dataset['Views'])))
predicted_Vals.fill(np.nan)
predicted_Vals[-(len(inv_yhat)):] = inv_yhat
#print(predicted_Vals)
#print(len(predicted_Vals))
#print(len(dataset['Views']))
#print(type(predicted_Vals))
result = dataset
result['Prediction'] = predicted_Vals
#print(result)


result['Close'].plot()
result['Prediction'].plot()

pyplot.legend()
pyplot.title('Actual Stock Prices vs Predictions')
pyplot.ylabel('Stock Price')
pyplot.xlabel('Time')
pyplot.show()


# the array we predicted
array = forecast_prediction[-forecast_out:]
# our true values
true_array = dataset['Close'].iloc[-(forecast_out+1):]
true_array = true_array.values
# initial invest moneys
init_invest = 10000
# initializing array of zeros to hold value of our stock holdings
holdings = np.zeros(len(array)-1)  # the monetary stock we bought based on our prediction (for n value)
# initialized as an empty array
# initializing array with money value
money = np.zeros(len(array)-1)
money[0] = init_invest

a, b, c = cost(array, true_array, init_invest, money, holdings)
print(a,b,c)