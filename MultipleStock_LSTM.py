from glob import glob
import logging
import pandas as pd
import numpy as np
import math
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot
from math import sqrt
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import save_model
from keras.models import load_model
from functions import cost
logging.basicConfig(format='%(asctime)s %(message)s',datefmt= '%m/%d/%y %I:%M:%S %p', level= logging.DEBUG)
log = logging.getLogger(__name__)

#read in all of our csv files at once from directory
#filesnames = glob('A*.csv')
filesnames = glob('G*.csv')
dataframes = [pd.read_csv(f, header = 0, index_col= 0) for f in filesnames]
numstocks = (len(filesnames))
log.debug(numstocks)
df = pd.concat(dataframes, axis = 1,  sort= False)
df = df.drop(df.index[-1]) #to drop last row of nan that we have
lastVals = df.tail(1)
lastVals = lastVals.values
print(lastVals.shape)
values = df.values

i =1
a = 3
#loop to train multiple machine learning models and save them separately to use for later prediction
for x in range(1,numstocks+1):

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


    # integer encode direction
    # ensure all data is float
    values = values.astype('float32')
    # normalize features
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    print('scaled: ', scaled.shape)
    # NEW: specify number of lag days
    n_days = 1
    n_features = 12 #got from varNo in view of reframed Dataframe
    forecast_out = 1
    # frame as supervised learning
    reframed = series_to_supervised(scaled, n_days, 1)
    target = np.zeros((len(reframed['var1(t)'])))
    target.fill(np.nan)
    reframed['Target'] = target
    name = 'var' + str(a) + '(t)'
    print(name)
    if (a+6) < len(reframed):
        a = a +6
    reframed['Target'] = reframed[name].shift(-forecast_out) ############figure out how to tell where our target closing price is (need to increment by adding interval traveling over up to certain point --> since also have
    reframed['Target'] = reframed['Target'].iloc[:-forecast_out]
    reframed = reframed.drop(reframed.index[-1])
    ##drop old target column
    # drop last row
    print('Reframed shape: ', reframed.shape)
    print(reframed)

    # split into train and test sets
    values = reframed.values
    # n_train_days = int(0.8*len(reframed['var1(t)']))
    n_train_days = int(len(reframed['var1(t)'])*.80)
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]
    # split into input and outputs
    n_obs = n_days * n_features
    train_X, train_y = train[:, :n_obs], train[:, -1]
    test_X, test_y = test[:, :n_obs], test[:, -1]
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
    history = model.fit(train_X, train_y, epochs=10, batch_size=72, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)
    filename = "model" + str(i)
    i +=1
    filename = filename + '.h5'
    model.save(filename)
    del model
    #MemoryError (on only one epoch....)

y = 1
#array = np.zeros(1, len(numstocks)) #figure out where to store our predications
array = np.zeros(1)
for x in range(1,numstocks):
    modelName = 'model' + str(y) +'.h5'
    print(modelName + 'running')
    model = load_model(modelName)
    # make a prediction
    lastVals = lastVals.reshape((1,1,12))
    yhat = model.predict(lastVals)



    del model
    y += 1
