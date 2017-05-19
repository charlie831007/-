import requests
from bs4 import BeautifulSoup
from datetime import date, timedelta
import numpy as np
import pandas as pd
import pickle
from sklearn import preprocessing, cross_validation, neighbors
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

#inList: ma的list, days: ma的天數
def movingAverage(inList, days, dataframe):
	priceSum = 0
	for i in range(days):
		priceSum += dataframe['close'][i]

	ma = priceSum / days
	inList.append(ma)

	for x in range(days, len(dataframe['close'])):
		priceSum -= dataframe['close'][x - days]
		priceSum += dataframe['close'][x]
		ma = priceSum / days
		inList.append(ma)

def KD(dataframe):
	dataframe['RSV'] = 100 * ((dataframe['close'] - dataframe['close'].rolling(window = 9).min()) / (dataframe['close'].rolling(window = 9).max() - dataframe['close'].rolling(window = 9).min()))
	dataframe['RSV'].fillna(method = 'bfill', inplace = True)

	data = {'K9': [50], 'D9': [50]}
	for i in range(1, len(dataframe)):
		k9 = (1 / 3) * dataframe['RSV'][i] + (2 / 3) * data['K9'][i - 1]
		data['K9'].append(k9)
		d9 = (1 / 3) * data['K9'][i] + (2 / 3) * data['D9'][i - 1]
		data['D9'].append(d9)
	dataframe['K9'] = data['K9']
	dataframe['D9'] = data['D9']

#黃金交叉(buy): 1, 死亡交叉(sell): -1, 皆無: 0
def KDCross(dataframe):
	cross =  [0]
	for i in range(1, len(dataframe)):
		if((dataframe['K9'][i] <= 30) & (dataframe['D9'][i] <= 30)):	#找黃金交叉
			if((dataframe['K9'][i] >= dataframe['D9'][i]) & (dataframe['K9'][i - 1] < dataframe['D9'][i - 1])):
				cross.append(1)
			else:
				cross.append(0)
		elif((dataframe['K9'][i] >= 70) & (dataframe['D9'][i] >= 70)):
			if((dataframe['K9'][i] <= dataframe['D9'][i]) & (dataframe['K9'][i - 1] > dataframe['D9'][i - 1])):
				cross.append(-1)
			else:
				cross.append(0)
		else:
			cross.append(0)

	dataframe['KDCross'] = cross

def relativeStrength(prices, n):
    deltas = np.diff(prices)
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)
 
    for i in range(n, len(prices)):
        delta = deltas[i-1]
 
        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta
 
        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n
 
        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)
 
    return rsi

#rsi 70 up(sell): -1, 30 down(buy): 1
def rsiReverse(dataframe, n):
	reverse = [0 for x in range(n)]
	for i in range(n, len(dataframe)):
		if(dataframe['rsi' + str(n)][i] >= 70):
			reverse.append(-1)
		elif(dataframe['rsi' + str(n)][i] <= 30):
			reverse.append(1)
		else:
			reverse.append(0)

	dataframe['rsiRev'] = reverse

def getTodayIndex(stkNum, year, month):
	r = requests.get('http://www.tse.com.tw/ch/trading/exchange/STOCK_DAY/genpage/Report' + str(year) + month + '/' + str(year) + month + '_F3_1_8_' + str(stkNum) + '.php')
	r.encoding = 'utf-8'
	soup = BeautifulSoup(r.text, 'lxml')
	soupData = soup.find('table', class_ = 'board_trad').select('tr')[2 : -5]
	for i in range(len(soupData)):
		soupData[i] = soupData[i].select('td')
	return soupData

def getData(startYear, startMon):
	data = {'date': [], 'open': [], 'high': [], 'low': [], 'close': []}
	thisYear = date.today().year
	thisMonth = date.today().month
	year = int(startYear)
	month = int(startMon)

	while(year <= thisYear):
		if(year == thisYear):
			while(month <= thisMonth):
				soupData = getTodayIndex(2330, year, str(month).zfill(2))
				for i in range(len(soupData)):
					data['date'].append(soupData[i][0].text)
					data['open'].append(soupData[i][3].text)
					data['high'].append(soupData[i][4].text)
					data['low'].append(soupData[i][5].text)
					data['close'].append(soupData[i][6].text)

				month += 1
		else:
			while(month <= 12):
				soupData = getTodayIndex(2330, year, str(month).zfill(2))
				for i in range(len(soupData)):
					data['date'].append(soupData[i][0].text)
					data['open'].append(soupData[i][3].text)
					data['high'].append(soupData[i][4].text)
					data['low'].append(soupData[i][5].text)
					data['close'].append(soupData[i][6].text)

				month += 1

		month = 1
		year += 1

	data['open'] = list(map(float, data['open']))
	data['high'] = list(map(float, data['high']))
	data['low'] = list(map(float, data['low']))
	data['close'] = list(map(float, data['close']))
	df = pd.DataFrame(data, columns = ['date', 'open', 'high', 'low', 'close'])
	df.set_index('date', inplace = True)
	return df

# profit & risk: 0~1
def train(profit, risk, days, dataframe):
	level = []
	for i in range(len(dataframe) - days):
		if((dataframe['close'][i + days] >= dataframe['close'][i] * (1 + profit)) & (dataframe['close'][i : i + days + 1].min() > dataframe['close'][i] * (1 - risk))):
			level.append(4)
		elif((dataframe['close'][i + days] >= dataframe['close'][i] * (1 + profit)) & (dataframe['close'][i : i + days + 1].min() <= dataframe['close'][i] * (1 - risk))):
			level.append(3)
		elif((dataframe['close'][i + days] < dataframe['close'][i] * (1 + profit)) & (dataframe['close'][i : i + days + 1].min() > dataframe['close'][i] * (1 - risk))):
			level.append(2)
		else:
			level.append(1)


	for x in range(days):
		level.append(2)

	dataframe['label'] = level

def KNN(startDate, stopDate, dataframe):
	df_train = dataframe.loc[startDate : stopDate]
	x = np.array(df_train[['ma5', 'ma20', 'KDCross', 'rsiRev']])
	x = preprocessing.scale(x)
	y = np.array(df_train['label'])
	x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y)

	clf = neighbors.KNeighborsClassifier()
	clf.fit(x_train, y_train)
	accuracy = clf.score(x_test, y_test)
	print('Accuracy:', accuracy)
	return clf

def predict(startDate, stopDate, dataframe, classifier):
	df_test = dataframe[startDate : stopDate]
	x = np.array(df_test[['ma5', 'ma20', 'KDCross', 'rsiRev']]) 
	x = preprocessing.scale(x)
	forecast = classifier.predict(x)
	df_test['predict'] = forecast

	rightPredict = 0
	for i in range(len(df_test)):
		if(df_test['predict'][i] == df_test['label'][i]):
			rightPredict += 1
	print('Accuracy:', rightPredict / len(df_test))

	stockNum = 0
	stock = []
	initDeposit = 100000
	deposit = []
	for i in range(len(df_test)):
		if((df_test['predict'][i] == 4 or df_test['predict'][i] == 3) & (float(initDeposit) > df_test['close'][i])):
			stockNum += 1
			stock.append(stockNum)
			initDeposit -= df_test['close'][i]
			deposit.append(initDeposit)
		elif((df_test['predict'][i] == 1) & (stockNum > 0)):
			initDeposit += df_test['close'][i] * stockNum
			deposit.append(initDeposit)
			stockNum = 0
			stock.append(stockNum)
		else:
			stock.append(stockNum)
			deposit.append(initDeposit)

	df_test['stockNum'] = stock
	df_test['deposit'] = deposit

	print('擁有股票張數:', stockNum)
	print('資產:', initDeposit)

	ax1 = plt.subplot2grid((2,1), (0,0))
	ax2 = plt.subplot2grid((2,1), (1,0), sharex = ax1)
	df_test[['ma5', 'ma20']].plot(ax = ax1, linewidth = 2, color = ['r', 'g'])
	df_test['deposit'].plot(ax = ax2, linewidth = 2, color = 'k', label = 'Deposit')
	for label in ax2.xaxis.get_ticklabels():
		label.set_rotation(45)
	ax2.legend(loc = 1)
	plt.show()


# df = getData(1994, 9)
# # print(len(df))
# # print(df)

# # moving average
# ma5 = [np.nan for x in range(4)]
# ma20 = [np.nan for x in range(19)]
# movingAverage(ma5, 5, df)
# movingAverage(ma20, 20, df)
# df['ma5'] = ma5
# df['ma20'] = ma20

# # KD
# KD(df)

# del df['RSV']

# rsi
# df['rsi10'] = relativeStrength(df['close'], 10)

# KD交叉
# KDCross(df)

# rsi reverse
# rsiReverse(df, 10)

# train(0.1, 0.1, 10, df)
# print(df)
# print('4:', len(df['label'][df['label'] == 4]))
# print('3:', len(df['label'][df['label'] == 3]))
# print('2:', len(df['label'][df['label'] == 2]))
# print('1:', len(df['label'][df['label'] == 1]))

# with open('2330.pickle', 'wb') as f:
# 	pickle.dump(df, f)

pickle_in = open('2330.pickle', 'rb')
df = pickle.load(pickle_in)

# train
clf = KNN('89/01/04', '104/12/31', df)
predict('89/01/04', '104/12/31', df, clf)
predict('83/09/30', '106/05/18', df, clf)
predict('105/01/04', '106/05/18', df, clf)
