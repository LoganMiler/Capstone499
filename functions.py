import math
import numpy as np
#cost function with inputs:
#predicted array, true array values, initial investment value, cost of transacting trades, money array to keep track of leftover funds, holdings to keep track of stock values held

def cost(array, true_array, init_invest, money, holdings):
    i =1
    n = np.zeros(len(array)-1)

    for x in range(1, len(array) - 1):

        order = array[x + 1] - array[x]
        print('#########################')
        print('order: ', order)
        print('money: ', money[x - 1])

        # buy order condition (need at least 500 to invest and must predict that stock will increase)
        if ((money[x - 1]) >= 500 and order > 0):
            print('buy')
            if x == 0:
                x = 1
            # creating value for number of stocks to buy
            n[x] = math.floor((money[x - 1]) / true_array[x])  # buy number of stocks based on our predicted value
            # calculate the actual value of stocks being bought (must use real price of stock)
            #print('n: ', n[x])
            buy = float(n[x] * true_array[x])
            #print('true array: ', true_array)
            #print('bought: ', buy)
            # subtract amount paid from our money funds (subtract amount of stock bought and amount paid to execute transaction)
            money[x] = money[x - 1] - buy
            # stock holdings value (add new holds to previous holdings)
            holdings[x] = holdings[x - 1] + buy
        elif (money[x] < 500 and order >= 0):
            print('hold')
            # record the value of our holdings
            if x == 0:
                x = 1
            holdings[x] = n[x - 1] * true_array[x]
            # record the amount of stocks we are currently holding (this number has not changed from previous day)
            n[x] = n[x - 1]
            # record the amount of leftover money
            money[x] = money[x - 1]
        elif (order < 0):
            print('sell')
            if x == 0:
                x = 1
            # value of stock holdings goes to zero since we are selling all
            holdings[x] = 0
            # value of stocks we are currently holding also goes to 0
            n[x] = 0
            # record the amount of money we have "leftover"
            money[x] = (holdings[x - 1]) + money[x - 1]
        elif (array[i] == array[i - 1]):
            print('hold')
            if x == 0:
                x = 1
            money[x] = money[x - 1]
            n[x] = n[x - 1]
            holdings = holdings[x - 1]

        # use i as a counter
        i += 1

    result = holdings[-1] + money[-1]
    percent_return = ((result - init_invest) / init_invest) * 100
    portfolio_value = holdings + money
    return  percent_return, portfolio_value

##############################################################################

