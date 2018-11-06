#import necessary libraries
import math
import numpy as np
import pandas

###############################################################################################
#cost function with inputs:
#predicted array, true array values, initial investment value, cost of transacting trades, money array to keep track of leftover funds, holdings to keep track of stock values held

def cost(array, true_array, prevDay, init_invest, tradeExec_cost, money, holdings):
    i = 0 #variable for iterating
    #create an array to store value for how many stocks we hold on daily basis
    n = np.zeros(len(true_array))

    for x in range(1, len(array)+1):

        # determinining the previous day value to predict if stock will go up or down
        if x == 1:
            previousDay = prevDay
        else:
            previousDay = array[i-1]

        # determining the money value (value of leftover money)
        if x == 1:
            leftoverMoney = money[0]
            # if we are on the first day, we assume we have no previous stock holdings, so initialize value to 0 (before transactions)
        else:
            leftoverMoney = money[i-1]

        # determining the value of our current stock holdings
        if x == 1:
            stockHoldings = 0
            #if we are on the first day, we assume we have no previous stock holdings, so initialize value to 0 (before transactions)
        else:
            stockHoldings = holdings[i-1]
            #otherwise, the current stockholdings (prior to transaction) are equal to the previous day stock holdings

        # determining whether to buy or sell (subtract next day from previous day)
        order = array[i] - previousDay
        print('#########################')
        print('order: ', order)
        print('money: ', leftoverMoney)

        # buy order condition (need at least 500 to invest and must predict that stock will increase)
        if leftoverMoney >= 500 and order > 0:
            print('buy')
            # creating value for number of stocks to buy
            n[x-1] = math.floor((leftoverMoney-tradeExec_cost) / true_array[x-1])  # buy number of stocks based on our predicted value

            # calculate the actual value of stocks being bought (must use real price of stock)
            buy = float(n[x-1] * true_array[x-1])

            # subtract amount paid from our money funds (subtract amount of stock bought and amount paid to execute transaction)
            money[x-1] = leftoverMoney - buy - tradeExec_cost

            # stock holdings value (add new holds to previous holdings)
            holdings[x-1] = stockHoldings + buy

        elif (leftoverMoney < 500 and order >= 0):
            print('hold')
            # record the value of our holdings
            #condition if it is day 1 (so no previous holdings)
            if x==1:
                num = 0
            else:
                num = n[x-2]
            holdings[x-1] = num * true_array[x-1]
            # record the amount of stocks we are currently holding (this number has not changed from previous day)
            n[x-1] = num
            # record the amount of leftover money
            money[x-1] = leftoverMoney
        elif (order < 0):
            print('sell')
            # value of stock holdings goes to zero since we are selling all
            holdings[x-1] = 0
            # value of stocks we are currently holding also goes to 0
            n[x-1] = 0
            # record the amount of money we have "leftover"
            money[x-1] = stockHoldings + leftoverMoney
        elif (array[i] == array[i - 1]):
            print('hold')
            # condition if it is day 1 (so no previous holdings)
            if x == 1:
                num = 0
            else:
                num = n[x - 2]

            money[x-1] = leftoverMoney
            n[x] = num
            holdings[x] = stockHoldings

        # use i as a counter
        i += 1

    #end resulting portfolio value
    result = holdings[-1] + money[-1]
    #calculating the percent returns on our investment
    percent_return = ((result - init_invest) / init_invest) * 100
    #array tracking daily portfolio value
    portfolio_value = holdings + money

    #return daily portfolio value and percent returns
    return percent_return, portfolio_value


##############################################################################

# convert time series data to supervised learning
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

######################################################################################

#create hold stock function to give a market baseline
#Inputs:
    ##truePrice_array = array of actual stock prices
    ##init_invest = initial invest value
    ##tradeExec_Cost = cost of executing a trade
#Outputs: percent returns, daily portfolio value

def HoldStock(truePrice_array, init_invest, tradeExec_Cost):
    # initialize number of stocks we are able to buy (based on initial investment value and stock prices)
    n = math.floor(
        (init_invest - tradeExec_Cost) / truePrice_array[0])  # only buy as much as we can the first day of investing

    # create an array for our leftover funds (money left after buy stocks with initial investment allocation):
    leftover = init_invest - (n * truePrice_array[0])

    # initialize array to put stock values in:
    stock_tracker = np.zeros([len(truePrice_array), 1])
    # initialize counter for stock tracker numpy array
    y = 0

    #track the daily value of our stock holdings
    for x in range(1, len(truePrice_array) + 1):
        stock_tracker[y] = n * truePrice_array[y]
        print(stock_tracker[y])
        print('#################')
        y += 1

    # final day portfolio value
    final_result = stock_tracker[-1] + leftover
    # calculate the percent return on our stock
    percent_change = ((final_result - init_invest) / init_invest) * 100
    # calculate the portfolio value for each day of investment
    portfolio_value = stock_tracker + leftover

    return percent_change, portfolio_value
