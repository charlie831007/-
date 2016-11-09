import urllib.request
from openpyxl import Workbook
import numpy as np

def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    smas = np.convolve(values, weights, 'valid')
    return smas

def crawl(stock):
    wb = Workbook()
    ws = wb.active
    
    url = "http://chartapi.finance.yahoo.com/instrument/1.0/" + stock + "/chartdata;type=quote;range=1y/csv"
    request = urllib.request.urlopen(url).read().decode('utf-8')
    
    aryName = []
    stockData = []
    splitSource = request.split('\n')
    for line in splitSource:
        splitData = line.split(',')
        if len(splitData) == 6:
            if 'values' in line:
                aryName.append(splitData)
            else:
                stockData.append(splitData)
    
    aryName[0].append('ma5')
    aryName[0].append('ma10')
    ws.append(aryName[0])
    
    close = []                  
    for row in range(len(stockData)):
        close.append(stockData[row][1])
     
    closep = np.loadtxt(close, float)
     
    ma5 = moving_average(closep, 5)
    ma10 = moving_average(closep, 10)
     
    for i in range(9, len(closep)):
        stockData[i].append(ma5[i - 9])
        stockData[i].append(ma10[i - 9])
        ws.append(stockData[i])
        
    wb.save(stock + '.xlsx')

crawl('goog')