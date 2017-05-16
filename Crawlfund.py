from bs4 import BeautifulSoup
import numpy as np
import pandas as pd
import pickle
import requests
from datetime import date
from sklearn import preprocessing, cross_validation, neighbors
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

#def getFundData(fundName, url):
def getFundData():
    # r = requests.get(url + )
    r = requests.get("http://fund.bot.com.tw/w/bcd/tBCDNavList.djbcd?a=ACDD01&b=0&c=1997-1-1&d=2017-12-31")
    
    data = dict()
    splitData = r.text.split(' ')
    dates = splitData[0].split(',')
    prices = splitData[1].split(',')
    data['date'] = dates
    data['price'] = prices
    data['price'] = list(map(float, data['price']))
    
    dataChange = [np.nan]
    for i in range(1, len(dates)):
        if(prices[i] > prices[i - 1]):
            dataChange.append(1)
        else:
            dataChange.append(0)
    data['dataChange'] = dataChange
                     
    df = pd.DataFrame(data, columns = ['date', 'price', 'dataChange'])
    df.set_index('date', inplace = True)
    
    # moving average
    ma5 = [np.nan for x in range(4)]
    ma20 = [np.nan for x in range(19)]
    movingAverage(ma5, 5, df)
    movingAverage(ma20, 20, df)
    df['ma5'] = ma5
    df['ma20'] = ma20
    
    # rsv
    df['RSV'] = 100 * ((df['price'] - df['price'].rolling(window = 9).min()) / (df['price'].rolling(window = 9).max() - df['price'].rolling(window = 9).min()))
    df['RSV'].fillna(method = 'bfill', inplace = True)
    
    # KD
    data = {'K9': [50], 'D9': [50]}
    for i in range(1, len(df.index)):
        k9 = (1 / 3) * df['RSV'][i] + (2 / 3) * data['K9'][i - 1]
        data['K9'].append(k9)
        d9 = (1 / 3) * data['K9'][i] + (2 / 3) * data['D9'][i - 1]
        data['D9'].append(d9)
        
    df_KD = pd.DataFrame(data)
    df_KD.set_index(df.index, inplace = True)
    df = pd.concat([df, df_KD], axis = 1, join_axes = [df.index])
    
    #train
    df_train = df.loc['20060101' : '20111231']
    
    x = np.array(df_train[['ma5', 'ma20']])
    x = preprocessing.scale(x)
    y = np.array(df_train['dataChange'])
    x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y)
    
    clf = neighbors.KNeighborsClassifier()
    clf.fit(x_train, y_train)
    accuracy = clf.score(x_test, y_test)
    print('Accuracy', accuracy)
    
    backtest(df, '20060101', clf)
    backtest(df, '20010101', clf)
    print(df)

#inList: ma的list; days: ma的天數; d
def movingAverage(inList, days, dataframe):
    fundSum = 0
    for i in range(days):
        fundSum += dataframe['price'][i]
    
    ma = fundSum / days
    inList.append(ma)
    
    for x in range(days, len(dataframe['price'])):
        fundSum -= dataframe['price'][x - days]
        fundSum += dataframe['price'][x]
        ma = fundSum / days
        inList.append(ma)
        
def backtest(dataframe, startday, classifier):
    df_backtest = dataframe[startday:]
    
    x_backtest = np.array(df_backtest[['ma5', 'ma20']])
    x_backtest = preprocessing.scale(x_backtest)
    forecast = classifier.predict(x_backtest)
    df_backtest['forecast'] = forecast
    rightPredict = 0
    for i in range(len(df_backtest.index)):
        if df_backtest['dataChange'][i] == df_backtest['forecast'][i]:
            rightPredict += 1
            
    print(df_backtest[['dataChange', 'forecast']])
    print('Accuracy', rightPredict / len(df_backtest.index))

getFundData()