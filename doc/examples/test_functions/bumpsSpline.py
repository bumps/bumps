# -*- coding: utf-8 -*-
"""
Created on Mon Jun 22 11:17:02 2015

@author: kwj
"""
import numpy as np
from bumps.names import *
import bumps.mono as mon
import matplotlib.pyplot as plt
#
def spline(x, x1, x2, x3, x4, x5, x6, x7, x8, y1, y2, y3, y4, y5, y6, y7, y8):
    return mon.monospline([x1, x2, x3, x4, x5, x6, x7, x8], [y1, y2, y3, y4, y5, y6, y7, y8], x)
#def spline(x, *points):
#    return mon.monospline(points[slice(0, int(len(points)/2))], points[slice(int(len(points)/2), len(points))], [x])

x = np.linspace(5, 10, 8)
y = np.random.random(8)
#y = [4.0,4.0,6.3,8.03,9.6,11.9]
y = [.5, .245, .267, .5, .8, .3, .4, .9]
xt = np.linspace(5,10,50)
yt = mon.monospline(x, y, xt)
dyt = np.ones_like(yt) * 0.01

plt.plot(x, y, 'o')
#plt.plot(xt, yt)
#plt.show()
p = [5,6,7,8,9,0.1,0.2,0.3,0.4,0.5]
#M = Curve(spline, xt, yt , dyt,2x1=5, x2=6 , x3=7, x4=8, x5=9, y1=0.1, y2=0.2, y3=0.3, y4=0.4, y5=.5)
M = Curve(spline, xt, yt , dyt, x1=5, x2=6 , x3=7, x4=7.5, x5=8, x6=8.5, x7=9, x8=9.5, y1=0.1, y2=0.2, y3=0.3, y4=0.4, y5=.5, y6=.6, y7=.7, y8=.8)
M.x1.range(5, 10)
M.x2.range(5, 10)
M.x3.range(5, 10)
M.x4.range(5, 10)
M.x5.range(5, 10)
M.x6.range(5, 10)
M.x7.range(5, 10)
M.x8.range(5, 10)
M.y1.range(0,1)
M.y2.range(0,1)
M.y3.range(0,1)
M.y4.range(0,1)
M.y5.range(0,1)
M.y6.range(0,1)
M.y7.range(0,1)
M.y8.range(0,1)
#p = [1,1, 2,3, 3,1]
#xt = np.linspace(-5, 5, 50)
#y = spline(xt, *p)
#print(xt)
#print(y)
#M = Curve(spline, xt,y)
problem = FitProblem(M)
#print(x)
#print(y)
#

