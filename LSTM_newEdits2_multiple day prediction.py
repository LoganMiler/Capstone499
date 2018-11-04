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
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import save_model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
#from functions import cost
logging.basicConfig(format='%(asctime)s %(message)s',datefmt= '%m/%d/%y %I:%M:%S %p', level= logging.DEBUG)
log = logging.getLogger(__name__)
#read in all of our csv files at once from directory
#filesnames = glob('A*.csv')
filesnames = glob('IT_AAPL*.csv')
dataframes = [pd.read_csv(f, header = 0, index_col= 0) for f in filesnames]
numstocks = (len(filesnames))
log.debug(numstocks)
df = pd.concat(dataframes, axis = 1,  sort= False)

lastIndex = len(df['Close']) #get last index of dataframe
numForecastDays = 10 #variable for number of days we want to forecast out
forecastDays_Index = lastIndex - numForecastDays #index to take days we want to forecast out off of datafram

dfTrueVals = df.iloc[forecastDays_Index:lastIndex] #slice off the days that we are predicting
dfTrueVals =dfTrueVals['Close'] #keep only the closing prices

df = df.drop(df.index[forecastDays_Index:lastIndex])#drop the days that we are going to predict out on
values = df.values #convert out dataframe

lastVals = df.iloc[-1] #last day of our dataframe to start predictions on next days

#get the previous closing day to determine magnitude of price movement if we use differencing
#prevPrice_array = lastVals['Close']  #will use this to determine magnitude of price movement at end

stockNum = numstocks
truePrice_array = np.zeros(stockNum)
x = 1
d = 0
i = 4
while x<=stockNum:
    truePrice_array[d] = lastVals.iloc[i]
    i += 6
    x +=1
    d +=1

lastVals = lastVals.values #convert to numpy array

i =1
a = 3
y = 4 ######INITIALIZE Y HERE TO SET OUR TARGET VARIABLE
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
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # NEW
    values = scaler.fit_transform(values)  # NEW
    #specify the name of our target variable
    #target = 'var' + str(y) + '(t)'
    target = []
    y = 1
    for x in range(1,len(df.columns)+1):
        target.append('var' + str(y) + '(t)')
        y +=1

    print(target)


    interval = 6 #CHANGES DEPENDING ON INPUTS
    y += interval #so that each time we go through loop we get new target variable
    # NEW: specify number of lag days
    n_days = 1
    n_features = len(df.columns) #got from varNo in view of reframed Dataframe
    forecast_out = 1
    # frame as supervised learning
    reframed = series_to_supervised(values, n_days, 1)
    b = 0
    for x in range(1,n_features+1):
        named = 'var' + str(x) + '(t)'
        targetName = target[b]
        b += 1
        if named != targetName:
            reframed = reframed.drop(named, axis = 1)


    print('REFRAMED:', reframed)


    values = reframed.values  # convert reframed dataframe into numpy array

    # split into train and test sets
    n_train_days = int(0.8 * len(reframed['var1(t-1)']))  # using 80% of our data as our training set
    n_test_days = int(len(reframed['var1(t-1)']) - n_train_days)
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-6], train[:, -6:]
    test_X, test_y = test[:, :-6], test[:, -6:]
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
    numEpochs = 1
    numBatch = 50
    model.add(LSTM(50, input_shape=(1, n_features)))  # feeding in 1 time step  with 7 features at a time
    model.add(Dense(6))
    model.compile(loss='mae', optimizer='adam')


    ####define the checkpoint:
    #checkpoint = ModelCheckpoint(filename, monitor='loss', verbose=1, mode='min')
    #callbacks_list = [checkpoint]


    # fit network
    model.fit(train_X, train_y, epochs=numEpochs, batch_size=numBatch, validation_data=(test_X, test_y), verbose=2,
                        shuffle=False)


    filename = "model" + str(i)
    i += 1
    d += 1
    filename = filename + '.h5'
    model.save(filename)
    model = load_model(filename)
    model.fit(test_X, test_y, epochs = numEpochs, batch_size = numBatch, verbose = 2, shuffle = False) #add state back in, or differencing technique
    model.save(filename)
    del model

    print('completed: ', filename)


print('we made it through the loop')
y = 1
log.debug(y)
g = 0

#array = np.zeros(1, len(numstocks)) #figure out where to store our predications
pred_array = np.zeros(stockNum)
rmse_array = np.zeros(stockNum)



for x in range(1,numstocks+1):
    modelName = 'model' + str(y) +'.h5'
    print(modelName + 'running')
    model = load_model(modelName)
    # make a prediction
    lastVals = lastVals.reshape((1,1,n_features))
    #new = test_X[-1].reshape((1,1,n_features))
    #yhat = model.predict(test_X[-1])
    yhat = model.predict(lastVals)
    df_Prediction = pd.DataFrame(data = yhat)



    i = 0
    for x in range(1,numForecastDays):
        predVals = df_Prediction.iloc[i]
        predVals = predVals.values
        predVals = predVals.reshape((1,1,n_features))
        new_yhat = model.predict(predVals)
        df1 = pd.DataFrame(data=new_yhat)
        df_Prediction = df_Prediction.append(df1, ignore_index= False)
        i +=1

    print('yay')
    del model
    y += 1
    g += 1

invertVals = df_Prediction.values
invertVals = scaler.inverse_transform(invertVals)
df_InvertedVals = pd.DataFrame(data=invertVals)
df_InvertedVals.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
predictedVals = df_InvertedVals['Close'].values#Get out predicted values into a dataframe
predictedVals.reshape(numForecastDays,)
df_predictedVals = pd.DataFrame(data = predictedVals)
df_predictedVals.columns = ['Close']
diff = dfTrueVals.values - predictedVals

#calculate the error between each prediction (record to graph to see if error increases over time)
#LAST THING TO INCLUDE IS THE DIFFERENCING

print('DIFF:', diff)




#CODED UP TO HERE


#output_DF = pd.DataFrame([list(truePrice_array), list(array)],columns = columns)
output_DF = pd.DataFrame(index = filesnames_Index,columns = ['true', 'predicted'])
output_DF['true'] = truePrice_array
output_DF['predicted'] = pred_array
diff = pred_array - prevPrice_array
accuracy = (abs(diff)/truePrice_array) *100
accuracy = np.array(accuracy)
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
print('Accuracy: ', accuracy)
output_DF['Error'] = accuracy
print('Output DF: ', output_DF)
print('done')