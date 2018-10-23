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
from keras.callbacks import ModelCheckpoint
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
#values = df.values
stockNum = 2
truePrice_array = np.zeros(stockNum)
prevPrice_array = df.iloc[-2]
prevPrice_array = prevPrice_array['Close']
print(prevPrice_array)
d = 0

i =1
a = 3
y = 4
#loop to train multiple machine learning models and save them separately to use for later prediction
for x in range(1,numstocks+1):
    values = df.values
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
    #specify the name of our target variable
    target = 'var' + str(y) + '(t)'
    y += 6 #so that each time we go through loop we get new target variable
    # NEW: specify number of lag days
    n_days = 1
    n_features = 12 #got from varNo in view of reframed Dataframe
    forecast_out = 1
    # frame as supervised learning
    reframed = series_to_supervised(values, n_days, 1)
    for x in range(1,n_features+1):
        named = 'var' + str(x) + '(t)'
        if named != target:
            reframed = reframed.drop(named, axis = 1)

    print('Reframed in loop: ', reframed.head())

    # prevDay is the last day that we will use to predict our next day stock prices
    prevDay = reframed[-1:]
    prevDay = prevDay.drop([target], axis=1)  # get rid of our target column
    prevDay = prevDay.values
    values = reframed.values  # get our input data as numpy array, shape is (1,2555)
    truePrice = reframed[-1:]
    truePrice = truePrice[target]
    truePrice = truePrice.values  # get the true closing price we are trying to predict
    truePrice_array[d] = truePrice

    # split into train and test sets
    n_train_days = int(0.8 * len(reframed['var1(t-1)']))  # using 80% of our data as our training set
    n_test_days = int(len(reframed['var1(t-1)']) - n_train_days)
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X, test_y = test[:, :-1], test[:, -1]
    # reshape to be 3d input [samples, timesteps, features]
    # samples are the number of training/ tesing days
    # time steps are 365 time steps for year of days
    # features are our 7 inputs
    train_X = train_X.reshape(n_train_days, n_days, n_features)
    test_X = test_X.reshape(n_test_days, n_days, n_features)
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    # input shape has dimesions (time step, features)
    model.add(LSTM(50, input_shape=(1, n_features)))  # feeding in 1 time step  with 7 features at a time
    model.add(Dense(1))
    model.compile(loss='mae', optimizer='adam')
    # fit network
    history = model.fit(train_X, train_y, epochs=1, batch_size=50, validation_data=(test_X, test_y), verbose=2,
                        shuffle=True)


    filename = "model" + str(i)
    i +=1
    d +=1
    filename = filename + '.h5'
    #model.save(filename)
    del model

    print('completed: ', filename)


print('we made it through the loop')
y = 1
log.debug(y)
g = 0

#array = np.zeros(1, len(numstocks)) #figure out where to store our predications
array = np.zeros(stockNum)
rmse_array = np.zeros(stockNum)
for x in range(1,numstocks+1):

    modelName = 'model' + str(y) +'.h5'
    print(modelName + 'running')
    #model = load_model(modelName)
    # make a prediction
    lastVals = lastVals.reshape((1,1,12))
    yhat = model.predict(lastVals)
    array[g] = yhat
    rmse_array[g] = sqrt(((array[g] - truePrice_array[g]) ** 2).mean())
    #print('Test RMSE: %.4f' % rmse)
    print('yay')



    del model
    y += 1
    g += 1

filesnames_Index = [w[0:3] for w in filesnames]

#output_DF = pd.DataFrame([list(truePrice_array), list(array)],columns = columns)
output_DF = pd.DataFrame(index = filesnames_Index,columns = ['true', 'predicted'])
output_DF['true'] = truePrice_array
output_DF['predicted'] = array
diff = array - prevPrice_array
diff = np.array(diff)
output_DF['Difference'] = diff
count = 0
stockList = []
while count < (stockNum):
    if diff[count] > 0:
        stockList.append('UP')
    elif diff[count]:
        stockList.append('DOWN')
    count += 1

output_DF['Direction'] = stockList
df_UP = output_DF.loc[output_DF['Direction']=='UP']
df_DOWN = output_DF.loc[output_DF['Direction']=='DOWN']
df_UP = df_UP.sort_values('Difference', ascending = False)
df_DOWN = df_DOWN.sort_values('Difference')
print(type(stockList))
print(stockList)
print('done')