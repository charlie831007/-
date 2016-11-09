'''
Created on 2016年9月15日

@author: Charlie
'''
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt

m = Basemap(projection = 'mill',
            llcrnrlat = 25,
            llcrnrlon = -130,
            urcrnrlat = 50,
            urcrnrlon = -60)
m.drawcoastlines()
# m.fillcontinents()
m.drawcountries(linewidth = 2)
m.drawstates(color = 'b')
# m.etopo()
# m.bluemarble()

xs = []
ys = []

NYClon, NYClat = -74.0059, 40.7127
xpt, ypt = m(NYClon, NYClat)
xs.append(xpt)
ys.append(ypt)
m.plot(xpt, ypt, 'm*', markersize = 15)

LAlon, LAlat = -118.25, 34.05
xpt, ypt = m(LAlon, LAlat)
xs.append(xpt)
ys.append(ypt)
m.plot(xpt, ypt, 'm^', markersize = 15)

m.plot(xs, ys, color = 'r', linewidth = 3, label = 'Straight line')
m.drawgreatcircle(NYClon, NYClat, LAlon, LAlat, color = 'c', linewidth = 3, label = 'Arc')

plt.legend(loc = 4)
plt.title('Basemap Tutorial')
plt.show()