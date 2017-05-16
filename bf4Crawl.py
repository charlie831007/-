import requests
from bs4 import BeautifulSoup
from datetime import timedelta, date
import numpy as np
import pandas as pd
import pickle

def getTodayIndex(year, month, day):
	keys = {'syear': year, 'smonth': month, 'sday': day}
	r = requests.post('http://www.taifex.com.tw/chinese/3/3_1_1.asp', data = keys)
	r.encoding = 'utf-8'
	soup = BeautifulSoup(r.text, "lxml")
	soup_data = soup.select('table')[2].select('table')[1].select('td')
	return soup_data

def getData(start):
	data = {'date': [], 'open': [], 'high': [], 'low': [], 'settlement': [], 'volume': [], 'close': [], 'open_int': [], 'close_best_bid': [], 'close_best_ask': []}
	day = date(start[0], start[1], start[2])
	today = date.today()
	span = ((today - day).days) + 1
	interval = timedelta(days = 1)

	def addToData(column, index):
		try:
			data[column].append(int(soup_data[index].text))
		except Exception as e:
			print(e)
			data[column].append(np.nan)

	for _ in range(span):
		try:
			soup_data = getTodayIndex(day.year, day.month, day.day)
			data['date'].append(day)
			addToData('open', 2)
			addToData('high', 3)
			addToData('low', 4)
			addToData('settlement', 5)
			addToData('volume', 8)
			addToData('close', 9)
			addToData('open_int', 10)
			addToData('close_best_bid', 11)
			addToData('close_best_ask', 12)
			print('%d %d %d is loaded' %(day.year, day.month, day.day))
		except Exception as e:
			print('stock market is not open on %d %d %d' %(day.year, day.month, day.day))

		day += interval

	df = pd.DataFrame(data, columns = ['date', 'open', 'high', 'low', 'settlement', 'volume', 'close', 'open_int', 'close_best_bid', 'close_best_ask'])
	df.set_index('date', inplace = True)
	return df

df = getData([1998, 7, 21])

# 存成pickle檔以便於讀寫
with open('TWfuture.pickle', 'wb') as f:
	pickle.dump(df, f)

pickle_in = open('TWfuture.pickle', 'rb')
df = pickle.load(pickle_in)

print(df)
