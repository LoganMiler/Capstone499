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


#load data
def parser(x):
    return datetime.strptime(x, '%m/%d/%Y')
#load dataset
#gives us dataframe
dataset = read_csv('testCont.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser= parser)
#convert to numpy array
values = dataset.values

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
values = dataset.values
values = values.astype('float32')
# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)
#NEW: specify number of lag days
n_days = 1
n_features = 5
forecast_out = 1
# frame as supervised learning
reframed = series_to_supervised(values, n_days, 1)
reframed = reframed.drop(['var2(t)', 'var3(t)', 'var4(t)', 'var5(t)'], axis = 1) #only want closing price as our target variable
#take last row out of our dataframe that we're predcting on (next day value)
reframed = reframed.drop(reframed.index[-1])
#prevDay is the last day that we will use to predict our next day stock prices
prevDay = reframed[-1:]
prevDay = prevDay.drop(['var1(t)'], axis = 1) #get rid of our target column
prevDay = prevDay.values
values = reframed.values #get our input data as numpy array, shape is (1,2555)
truePrice = dataset[-1:]
truePrice = truePrice['Close']
truePrice = truePrice.values #get the true closing price we are trying to predict

# split into train and test sets
n_train_days = int(0.8*len(reframed['var1(t)']))#using 80% of our data as our training set
n_test_days = int(len(reframed['var1(t)'])-n_train_days)
train = values[:n_train_days, :]
test = values[n_train_days:, :]
# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:,:-1], test[:, -1]
#reshape to be 3d input [samples, timesteps, features]
#samples are the number of training/ tesing days
#time steps are 365 time steps for year of days
#features are our 7 inputs
train_X = train_X.reshape(n_train_days, n_days, n_features)
test_X = test_X.reshape(n_test_days, n_days, n_features)
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
#input shape has dimesions (time step, features)
model.add(LSTM(50, input_shape= (1,5))) #feeding in 1 time step  with 7 features at a time
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=1000, batch_size= 50, validation_data=(test_X, test_y), verbose=2, shuffle=True)
# plot history
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.xlabel('Epoch')
#pyplot.ylabel('Loss')
#pyplot.title('Loss Throughout Training')
#pyplot.show()

print('shapes: ', test_X.shape[0], n_features * n_features)
print('please work')



# make a prediction
#newPrices_vals = newPrices.values
#newPrices = newPrices_vals.reshape((newPrices_vals.shape[0], n_days, n_features))
prevDay = prevDay.reshape(1,1, 5)
yhat = model.predict(prevDay)
print('yhat: ', yhat)
yhat = yhat.reshape(1,1)
prevDay_data = prevDay[:,:,:4]
print('prevDay: ', prevDay_data.shape, prevDay_data)
#prevDay_data = prevDay[:,-6:,6]
yhat = np.append(prevDay_data, yhat)
yhat = yhat.reshape(1,5)
yhat = scaler.inverse_transform(yhat)
print('inverted yhat: ', yhat)
prediction = (yhat[0][-1])
truePrice = (truePrice[0])
print('prediction: ', type(prediction), prediction)
print('true value: ', type(prediction), truePrice)
rmse = sqrt(((prediction - truePrice)**2).mean())
print('Test RMSE: %.4f' % rmse)
print('yay')