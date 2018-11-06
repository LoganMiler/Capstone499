from glob import glob
import logging
import pandas as pd
import numpy as np
import math
from pandas import read_csv
from datetime import datetime
from matplotlib import pyplot as plt
from math import sqrt
from numpy import concatenate
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import save_model
from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from functions import cost
from functions import series_to_supervised


logging.basicConfig(format='%(asctime)s %(message)s',datefmt= '%m/%d/%y %I:%M:%S %p', level= logging.DEBUG)
log = logging.getLogger(__name__)

#read in all of our csv files at once from directory
filesnames = glob('IT1_*.csv')

#get a list of our dataframes for each stock
dataframes = [pd.read_csv(f, header = 0, index_col= 0) for f in filesnames]
numstocks = (len(filesnames)) #save value for number of stocks we have
log.debug(numstocks)

#########################################################################################################

#initialize variable to iterate through our dataframes in loop
k = 0 #iterate for string names

#loop to train multiple machine learning models and save them separately to use for later prediction
volatility_array = np.zeros(numstocks)
prevDay_array = np.zeros(numstocks)

#initializing a list to store our lastVals dataframes in
r = 1 #iterator to keep track of lastVals dataframes

for x in range(1,numstocks+1):
    #read in our dataframe for individual stock
    num = x -1
    df = dataframes[num]
    
    #setting our target variables
    target1 = ['Open', 'High', 'Low', 'Adj Close', 'Volume']  # this will have to be changed as our inputs change
    lastIndex = len(df['Close'])  # get last index of dataframe
    numForecastDays = 1  # variable for number of days we want to forecast out
    forecastDays_Index = lastIndex - numForecastDays  # index to take days we want to forecast out off of datafram


    #######################################
    # Specify number of lag days, number of features and number of days we are forecasting out
    n_days = 1
    n_features = len(df.columns)  # got from varNo in view of reframed Dataframe
    forecast_out = 1
    
     #store the closing price so we can iterate on the next day value with our magnitudes of prediction
    prevDay = df.iloc[forecastDays_Index-1]
    prevDay = prevDay['Close']
    prevDay_array[num] = prevDay

    #recording days to input to make our prediction
    lastVals = df.iloc[(forecastDays_Index-1-n_days):(forecastDays_Index-1)]
    if x == 1:
        stockList = [lastVals]
    else:
        stockList.append(lastVals)
    r += 1

    #recording the true price values for the days we are predicting on
    trueValues = df.iloc[(forecastDays_Index):]
    trueValues = trueValues['Close'].values
    trueValues = trueValues.reshape(1,numForecastDays)

    #create a dataframe to store each of our last day values
    if x == 1:
        #create dataframe to store our true closing price values
        dfTrueVals = pd.DataFrame(data= trueValues)

    else:
        #recording true closing stock values for each stock we are predicting on
        df1 = pd.DataFrame(data= trueValues)
        dfTrueVals = dfTrueVals.append(df1)

    ###############################################################################
    #our differencing to make predictions stationary
    df = df.diff()
    df = df.drop(df.index[0]) #drop our first row of nans
    df = df.drop(df.index[forecastDays_Index:lastIndex])  # drop the days that we are going to predict out on
    values = df.values  # convert out dataframe to array of numpy values for our calculations

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
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))  # Normalizing our data
    values = scaler.fit_transform(values)

    #specify the name of our target variable
    target = [] #create an empty list to append our target names
    y = 1 #variable to iterate over in over
    for x in range(1,len(df.columns)+1):
        target.append('var' + str(y) + '(t)')
        y +=1

    # frame as supervised learning
    reframed = series_to_supervised(values, n_days, 1)
    b = 0
    for x in range(1,n_features+1):
        named = 'var' + str(x) + '(t)'
        targetName = target[b]
        b += 1
        if named != targetName:
            reframed = reframed.drop(named, axis = 1)

    values = reframed.values  # convert reframed dataframe into numpy array

    # split into train and test sets
    n_train_days = int(0.8 * len(reframed['var1(t-1)']))  # using 80% of our data as our training set
    n_test_days = int(len(reframed['var1(t-1)']) - n_train_days)
    train = values[:n_train_days, :]
    test = values[n_train_days:, :]
    # split into input and outputs
    train_X, train_y = train[:, :-n_features], train[:, -n_features:]
    test_X, test_y = test[:, :-n_features], test[:, -n_features:]
    # reshape to be 3d input [samples, timesteps, features]
    # samples are the number of training/ tesing days
    train_X = train_X.reshape(n_train_days, n_days, n_features)
    test_X = test_X.reshape(n_test_days, n_days, n_features)
    print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

    # design network
    model = Sequential()
    # input shape has dimesions (time step, features)
    numEpochs = 10
    numBatch = 25
    model.add(LSTM(50, input_shape=(n_days, n_features)))  # feeding in 1 time step  with 7 features at a time
    model.add(Dense(n_features))
    model.compile(loss='mae', optimizer='adam',metrics=['accuracy'])

    # fit network
    history = model.fit(train_X, train_y, epochs=numEpochs, batch_size=numBatch, validation_data=(test_X, test_y), verbose=0,
                        shuffle=False)
    
    # create filenames for each of our trained models to be saved 
    filename = "model" + str(k)+'.h5'
    model.save(filename)
    model = load_model(filename)
    model.fit(test_X, test_y, epochs = numEpochs, batch_size = numBatch, verbose = 2, shuffle = False) #add state back in, or differencing technique
    model.save(filename)
    del model
    k += 1
    
###################################################################### 
#initializing variables and array for next loop (making predictions of future prices) 
a = 0 #iterator for changing name of predictions for each stock exported to CSV
volatility_array = np.zeros(numstocks) #create an array to store the volatility of each stock
rmse_array = np.zeros(numstocks) #create an array to store rmse for each stock tested
y = 1#iterator for dfName only
c = 0 #iterator for prevDay array

q = 1 #iterator for lastVals names

#loop for performing predictions for each input stock data 
for x in range(1,numstocks+1):
    modelName = 'model' + str(a) +'.h5'
    stockName = filesnames[a]
    stockName = stockName[3:]
    stockName = stockName[:-4]
    dfName = 'df'+ stockName + '.csv'
    y +=1
    a += 1 #also being using for dfName
    print(modelName + 'running')
    model = load_model(modelName)

    # make a prediction
    num2 = x - 1 #the first index for lastVals
    dfLastValues = stockList[num2]
    lastVals = dfLastValues.values #turn our dataframe into an array
    lastVals = lastVals.reshape((1,n_days,n_features)) #reshape our array 

    #make prediction
    yhat = model.predict(lastVals)
    df_Prediction = pd.DataFrame(data = yhat)
    lastVals = df.iloc[(forecastDays_Index - n_days):(forecastDays_Index)].values
    lastVals = np.vstack((lastVals,yhat))
    prevDay_array2 = int(prevDay_array[c])
    c += 1

    #our loop within a loop for multiple forecasting days
    #initializing arrays for this loop 
    predictedPrice_array = np.zeros(numForecastDays)
    errorArray = np.zeros(numForecastDays)


    for x in range(1,numForecastDays+1):
        #get the predictions for change in magnitude of price
        predVals = lastVals[-n_days:] #slice so we are predicting on the last given days
        predVals = predVals.reshape((1,n_days,n_features))
        new_yhat = model.predict(predVals)#predicing next day out values
        # store our previous predicted values for sliding window
        lastVals = np.vstack((lastVals, new_yhat))  # keep stacking our output arrays
        lastVals = lastVals[-n_days:]  # slice array to keep track of our previous days 

        #invert our normalized values
        invertVals = scaler.inverse_transform(new_yhat)
        df_InvertedVals = pd.DataFrame(data=invertVals)
        #These column names for InvertedVals can be adjusted depending on our input variables names: 
        df_InvertedVals.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        #df_InvertedVals.columns = ['Open', 'High', 'Low', 'Close', 'Adj Close']
        predictedVals = df_InvertedVals['Close'].values  # Get out predicted values into a dataframe
        
        #store our predicted values in a dataframe 
        if x == 1:
            df_predictedVals = pd.DataFrame(data=predictedVals)
        else:
            df1 = pd.DataFrame(data= predictedVals)
            df_predictedVals = df_predictedVals.append(df1)

    #############################################
    df_predictedVals.reset_index(drop = True) #indexing our dataframe 
    trueVals = dfTrueVals.iloc[num2].values #putting the true stock price values in an array 
    #create an empty array to fill with rmse for each data point (to record how error is as time goes on)
    rmse_array = np.zeros(len(predictedVals))
    predictedVals = df_predictedVals.values #create an array of our predicted stock price changes
    df_predictedVals.reset_index(drop=True)

    i = 0
    w = 0
    for x in range(1,len(predictedVals)+1):
    #get predictions of real price by adding to last day we have iteratively (prevDay)
        if i == 0:
            #This says our first actual price equals the previous price + change in price magnitude
            #predictedPrice_array[i] = prevDay_array[i] + predictedVals[0]
            predictedPrice_array[i] = prevDay_array2 + predictedVals[0]
            errorArray[i] = ((trueVals[i]- predictedPrice_array[i])/trueVals[i])*100

        else:
            #This says we keep adding price magnitude to the next day
            predictedPrice_array[i] = predictedPrice_array[i-1] + predictedVals[i]
            errorArray[i] = ((trueVals[i] - predictedPrice_array[i]) / trueVals[i]) * 100


        #calculating RMSE:
        #rmse_array[m] = sqrt(mean_squared_error(trueVals, predictedPrice_array))
        i += 1

    #reshape our output arrays to be the same dimensions
    predictedPrice_array = predictedPrice_array.reshape(forecast_out,1)
    errorArray = errorArray.reshape(forecast_out,1)
    trueVals = trueVals.reshape(forecast_out,1)
    #put all of our arrays into one numpy array
    allVals = np.hstack((trueVals, predictedPrice_array, predictedVals, errorArray))
    #put all of these values into a dataframe to be exported to CSV file
    df_Output = pd.DataFrame(data= allVals)
    df_Output.columns = ['True Price', 'Predicted Price', 'Predicted Price Change','Error']
    df_Output.to_csv(dfName, sep=',')

    #find volatility
    volatility = predictedVals.sum()
    volatility_array[w] = volatility
    #find RMSE
    rmse = sqrt(mean_squared_error(trueVals, predictedVals))
    rmse_array[w] = rmse

    w += 1

    del model

j = 0
for x in range(1,numstocks+1):
    name = filesnames[j]
    name = name[3:]
    name = name[:-4]
    filesnames[j] = name
    j += 1

#sort our stocks based on their volatility and direciton of their projected movements
stocks = pd.DataFrame(data= filesnames)
movement = pd.DataFrame(data= volatility_array)
table = pd.concat([stocks, movement], axis= 1)
table.columns = ['Stock', 'Value']
table = table.sort_values(by = 'Value', ascending= False)
stockRef = 3
topStocks = table.iloc[0:stockRef]
bottomStocks = table.iloc[-stockRef:]
#record the rmse of the predictions for each of our stocks
error = pd.DataFrame(data= rmse_array)
table2 = pd.concat([stocks, error], axis = 1)

