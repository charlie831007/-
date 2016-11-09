'''
Created on 2016年9月19日

@author: Charlie
'''
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

style.use('fivethirtyeight')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1, projection = '3d')

# x = [1,2,3,4,5,6,7,8,9,10]
# y = [5,6,7,8,2,5,6,3,7,2]
# z = [1,2,6,3,2,7,3,3,7,2]
# 
# ax1.plot_wireframe(x, y, z)
# 
# x2 = [-1,-2,-3,-4,-5,-6,-7,-8,-9,-10]
# y2 = [-5,-6,-7,-8,-2,-5,-6,-3,-7,-2]
# z2 = [1,2,6,3,2,7,3,3,7,2]
# 
# ax1.scatter(x, y, z, color = 'g', marker = '*')
# ax1.scatter(x2, y2, z2, color = 'r', marker = '^')
# 
# x3 = [1,2,3,4,5,6,7,8,9,10]
# y3 = [5,6,7,8,2,5,6,3,7,2]
# z3 = np.zeros(10)
# dx = np.ones(10)
# dy = np.ones(10)
# dz = [5,6,7,8,2,5,6,3,7,2]
# 
# ax1.bar3d(x3, y3, z3, dx, dy, dz)

x, y, z = axes3d.get_test_data()

ax1.plot_wireframe(x, y, z, rstride = 5, cstride = 5)

ax1.set_xlabel('x-axis')
ax1.set_ylabel('y-axis')
ax1.set_zlabel('z-axis')

plt.show()