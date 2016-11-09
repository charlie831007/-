'''
Created on 2016年8月25日

@author: Charlie
'''
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from matplotlib.finance import candlestick_ohlc
from matplotlib import style
import numpy as np
import urllib
import datetime as dt

style.use('fivethirtyeight')

MA1 = 10
MA2 = 30

def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    smas = np.convolve(values, weights, 'valid')
    return smas

def high_minus_low(highs, lows):
    return highs - lows

def bytespdate2num(fmt, encoding='utf-8'):
    strconverter = mdates.strpdate2num(fmt)
    def bytesconverter(b):
        s = b.decode(encoding)
        return strconverter(s)
    return bytesconverter
    

def graph_data(stock):

    fig = plt.figure(facecolor = '#f0f0f0')
    
    ax1 = plt.subplot2grid((6, 1), (0, 0), rowspan = 1, colspan = 1)
    plt.title(stock)
    plt.ylabel('H-L')
    ax2 = plt.subplot2grid((6, 1), (1, 0), rowspan = 4, colspan = 1, sharex = ax1)
    plt.ylabel('Price')
    ax2v = ax2.twinx()
    ax3 = plt.subplot2grid((6, 1), (5, 0), rowspan = 1, colspan = 1, sharex = ax1)
    plt.xlabel('Date')
    plt.ylabel('MAvgs')
    
    url = 'http://chartapi.finance.yahoo.com/instrument/1.0/'+stock+'/chartdata;type=quote;range=1y/csv'
    request = urllib.request.urlopen(url).read().decode()
    stock_data = []
    split_source = request.split('\n')
    for line in split_source:
        split_line = line.split(',')
        if len(split_line) == 6:
            if 'values' not in line:
                stock_data.append(line)
    
    # unix time converter
    '''            
    date, closep, highp, lowp, openp, volume = np.loadtxt(stock_data,
                                                          delimiter=',',
                                                          unpack=True)
     
    dateconv = np.vectorize(dt.datetime.fromtimestamp)
    date = dateconv(date)
    '''
                
    date, closep, highp, lowp, openp, volume = np.loadtxt(stock_data,
                                                          delimiter=',',
                                                          unpack=True,
                                                          # %Y = full year 2016
                                                          # %y = partial year 16
                                                          # %m = number month
                                                          # %d = number day
                                                          # %H = hours
                                                          # %M = minutes
                                                          # %S = seconds
                                                          # 12-06-2014
                                                          # %m-%d-%Y
                                                          converters={0: bytespdate2num('%Y%m%d')})
    
    ma1 = moving_average(closep, MA1)
    ma2 = moving_average(closep, MA2)
    start = len(date[MA2 - 1:])
    
    h_l = list(map(high_minus_low, highp, lowp))
    
    ax1.plot_date(date[-start:], h_l[-start:], '-', c = '#0000C6', linewidth = 2, label = 'H-L')
    ax1.yaxis.set_major_locator(mticker.MaxNLocator(nbins = 4, prune = 'lower'))
    
    x = 0
    y = len(date)
    ohlc = []
    while x < y:
        append_me = date[x], closep[x], highp[x], lowp[x], openp[x], volume[x]
        ohlc.append(append_me)
        x += 1
      
    candlestick_ohlc(ax2, ohlc[-start:], width = 0.4, colorup = 'r', colordown = '#3bfb10')
    
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(nbins = 7, prune = 'upper'))
    
    bbox_props = dict(boxstyle = 'round', fc = 'w', ec = 'k', lw = 1)
    ax2.annotate(str(closep[-1]),
                 (date[-1], closep[-1]),
                 xytext = (date[-1] + 4, closep[-1]),
                 bbox = bbox_props)
    
    ax2v.fill_between(date[-start:], 0, volume[-start:], facecolor = '#0079a3', alpha = 0.4)
    ax2v.axes.yaxis.set_ticklabels([])
    ax2v.grid(False)
    ax2v.set_ylim(0, 2 * volume.max())
    ax2v.plot([], [], color = '#0079a3', alpha = 0.4, label = 'Volume')
    
    #Annotation example
    '''
    ax1.annotate('Annotate Test!',
                 (date[9], lowp[9]),
                 xytext = (0.8, 0.9),
                 textcoords = 'axes fraction',
                 arrowprops = dict(color = 'gray'))
    ''' 
    
    #Text and fontdict example
    '''
    font_dict = {'family':'serif',
                 'color':'darkred',
                 'size':15}
    ax1.text(date[10], closep[1], 'Text Example', fontdict = font_dict)
    '''
    
#     ax1.plot_date(date, closep, '-', label = 'Price')
#     ax1.fill_between(date, closep, closep[0], where = (closep > closep[0]), facecolor = 'r', alpha = 0.5, label = 'gain')
#     ax1.fill_between(date, closep, closep[0], where = (closep < closep[0]), facecolor = 'g', alpha = 0.5, label = 'loss')
#       
#     #表格邊框
#     ax1.spines['left'].set_color('r')
#     ax1.spines['top'].set_visible(False)
#     ax1.spines['right'].set_visible(False)
#       
#     #X/Y軸數字參數
#     ax1.tick_params(axis = 'x', colors = '#A02992')
#       
#     ax1.axhline(closep[0], color = 'k', linewidth = 3)
#      
#     for label in ax1.xaxis.get_ticklabels():
#         label.set_rotation(45)
#     ax1.grid(True)
#     ax1.xaxis.label.set_color('r')
#     ax1.yaxis.label.set_color('g')
    
    
    ax3.plot(date[-start:], ma1[-start:], linewidth = 1, label = str(MA1) + 'MA')
    ax3.plot(date[-start:], ma2[-start:], linewidth = 1, label = str(MA2) + 'MA')
    
    ax3.yaxis.set_major_locator(mticker.MaxNLocator(nbins = 4, prune = 'upper'))
    
    ax3.fill_between(date[-start:], ma2[-start:],
                     ma1[-start:], 
                     where = (ma1[-start:] > ma2[-start:]),
                     facecolor = 'g', edgecolor = 'g', alpha = 0.5)
    ax3.fill_between(date[-start:], ma2[-start:],
                     ma1[-start:], 
                     where = (ma1[-start:] < ma2[-start:]),
                     facecolor = 'r', edgecolor = 'r', alpha = 0.5)
    
    for label in ax3.xaxis.get_ticklabels():
        label.set_rotation(45)
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    ax3.xaxis.set_major_locator(mticker.MaxNLocator(10))
    
    plt.setp(ax1.get_xticklabels(), visible = False)
    plt.setp(ax2.get_xticklabels(), visible = False)
    plt.grid(True)
    ax1.legend()
    leg = ax1.legend(loc = 9, ncol = 1, prop = {'size': 11})
    leg.get_frame().set_alpha(0.4)
    ax2v.legend(loc = 9, ncol = 1, prop = {'size': 11}).get_frame().set_alpha(0.4)
    ax3.legend(loc = 9, ncol = 2, prop = {'size': 11}).get_frame().set_alpha(0.4)
    plt.subplots_adjust(right = 0.87, left = 0.1, bottom = 0.2, top = 0.95)
    plt.show()
    fig.savefig(stock, facecolor = fig.get_facecolor())

graph_data('GOOG')


