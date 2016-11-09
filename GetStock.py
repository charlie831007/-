'''
Created on 2016年8月10日

@author: Charlie
'''
import urllib.request
import time

def crawlStock(stock):
    print("Currently pulling " + stock)
    print(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
    
    f = open(stock + '.txt', 'w')
    
    url = "http://chartapi.finance.yahoo.com/instrument/1.0/" + stock + "/chartdata;type=quote;range=1y/csv"
    request = urllib.request.urlopen(url).read().decode("utf-8")

    splitSource = request.split('\n')    
    for eachLine in splitSource:
        splitData = eachLine.split(',')
        if len(splitData) == 6:
            if 'values' not in eachLine:
                f.write(eachLine + '\n')
    
    f.close()
    time.sleep(1)                   
    
crawlStock('goog')

stocks = 'AAPL', 'TSLA', 'GOOG', 'CMG', 'EBAY', 'MSFT', 'AMZN'
for stock in stocks:
    crawlStock(stock)

