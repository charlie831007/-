import pickle
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors
from datetime import date
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

pickle_in = open('TWfuture.pickle', 'rb')
df = pickle.load(pickle_in)

df = df[['open', 'high', 'low', 'settlement', 'volume', 'close', 'open_int', 'close_best_bid', 'close_best_ask']]

# fill the NaN with 'settlement' value
df['close_adjusted'] = df['close']								
df['close_adjusted'].fillna(df['settlement'], inplace = True)
df['close_best_bid'].fillna(df['settlement'], inplace = True)
df['close_best_ask'].fillna(df['settlement'], inplace = True)

# moving average
df['close_mvag5'] = df['close_adjusted'].rolling(window = 5).mean()		# ma5
df['close_mvag20'] = df['close_adjusted'].rolling(window = 20).mean()	# ma20

# raw stochastic value
df['RSV'] = 100 * ((df['close'] - df['low'].rolling(window = 9).min()) / (df['high'].rolling(window = 9).max() - df['low'].rolling(window = 9).min()))
df['RSV'].fillna(method = 'bfill', inplace = True)

data = {'K9': [17], 'D9': [39]}		# the 0th value of K and D

# KD
for i in range(1, len(df.index)):
	K9_value = (1 / 3) * df['RSV'][i] + (2 / 3) * data['K9'][i - 1]
	data['K9'].append(K9_value)
	D9_value = (1 / 3) * data['K9'][i] + (2 / 3) * data['D9'][i - 1]
	data['D9'].append(D9_value)

# put KD into dataframe
df_KD = pd.DataFrame(data)
df_KD.set_index(df.index, inplace = True)
df = pd.concat([df, df_KD], axis = 1, join_axes = [df.index])

df[['y_close_mvag5', 'y_close_mvag20', 'y_K9', 'y_D9']] = df[['close_mvag5', 'close_mvag20', 'K9', 'D9']].shift(1)

# print(df.tail())

# plot graph
# ax1 = plt.subplot2grid((2, 1), (0, 0))
# ax2 = plt.subplot2grid((2, 1), (1, 0), sharex = ax1)

# df[['close', 'close_mvag5', 'close_mvag20']].plot(ax = ax1, linewidth = 2, color = ['k', 'r', 'b'])
# df[['RSV', 'K9', 'D9']].plot(ax = ax2, linewidth = 2, color = ['r', 'g', 'b'])
# plt.show()

# training
df_temp = df.loc[date(2000, 1, 1) : date(2015, 12, 31)]

labels = []

for i in range(len(df_temp.index)):
	if df_temp['open'][i] < df_temp['settlement'][i]:
		labels.append(1)
	else:
		labels.append(0)

df_temp['labels'] = pd.Series(labels, index = df_temp.index)

x = np.array(df_temp[['y_K9', 'y_D9', 'y_close_mvag5', 'y_close_mvag20']])
x = preprocessing.scale(x)
y = np.array(df_temp['labels'])

x_train, x_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size = 0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
print('accuracy', accuracy)

# 做多的賺賠 = 收盤最佳買價 - 開盤價
df['long'] = df['close_best_bid'] - df['open']
df_temp['long'] = df_temp['close_best_bid'] - df_temp['open']
# 做空的賺賠 = 開盤價 - 收盤最佳賣價
df['short'] = df['open'] - df['close_best_ask']
df_temp['short'] = df_temp['open'] - df_temp['close_best_ask']

# backtest
def backtest(dataframe, startday, classifier):
	# 取出要回測的資料
	df_backtest = dataframe.loc[startday :]

	# 匯入特徵值
	x_backtest = np.array(df_backtest[['y_K9', 'y_D9', 'y_close_mvag5', 'y_close_mvag20']])
	x_backtest = preprocessing.scale(x_backtest)

	# Machine prediction
	forecast = classifier.predict(x_backtest)

	# Assume initial asset = 100000
	deposit = 100000
	deposits = []
	right_call = 0
	right_put = 0
	wrong_call = 0
	wrong_put = 0

	#每一天的交易情形
	for i in range(len(df_backtest.index)):
		if forecast[i] == 1:
			# 假設買一口大台，一點賺賠是200元
			deposit += (df_backtest['long'][i] * 200)
			print('%d %s long, because forecast is: %d, you earn: %d' % (df_backtest['open'][i], df_backtest.index[i].isoformat(), forecast[i], df_backtest['long'][i]))
			
			if df_backtest['long'][i] > 0:
				right_call += 1
			else:
				wrong_call += 1
		elif forecast[i] == 0:
			deposit += (df_backtest['short'][i] * 200)
			print('%d %s short, because forecast is: %d, you earn: %d' % (df_backtest['open'][i], df_backtest.index[i].isoformat(), forecast[i], df_backtest['short'][i]))

			if df_backtest['short'][i] > 0:
				right_put +=1
			else:
				wrong_put += 1

		deposits.append(deposit)
		print(deposit)

	#結果視覺化
	df_backtest['deposits'] = deposits
	right_score = right_call + right_put
	wrong_score = wrong_call + wrong_put
	call_accuracy = right_call / float(right_call + wrong_call)
	put_accuracy = right_put / float(right_put + wrong_put)
	accuracy = right_score / float(right_score + wrong_score)

	print('origin deposit: $100000')
	print('final deposit: $%d' % deposit)
	print('accuracy: %f' %  accuracy)
	# print('right call & put: %f %f wrong call & put: %f %f' % (right_call, right_put, wrong_call, wrong_put))
	# print('call accuracy: %f, put accuracy: %f, accuracy: %f' % (call_accuracy, put_accuracy, accuracy))

	ax1 = plt.subplot2grid((2, 1), (0, 0))
	ax2 = plt.subplot2grid((2, 1), (1, 0), sharex = ax1)
	df_backtest[['close', 'close_mvag5', 'close_mvag20']].plot(ax = ax1, linewidth = 2, color = ['k', 'r', 'g', 'b'])
	df_backtest['deposits'].plot(ax = ax2, linewidth = 2, color = 'k', label = "deposits")
	for label in ax2.xaxis.get_ticklabels():
		label.set_rotation(45)
	ax2.legend(loc = 1)
	plt.show()

# backtest from 2000/1/1
#df_temp中的資料範圍為訓練特徵的時間範圍，所以拿這段時間做回測通常較為準確
backtest(df_temp, date(2000, 1, 1), clf)

# forecast
backtest(df, date(2016, 1, 1), clf)
