# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 16:34:27 2023

@author: ADMIN
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
import math
h=0.01
x=np.linspace(-0.1,1.1,num=100)
fx= lambda x: np.sqrt(np.tan(x))
def func_der(x,h,func):
    der=(1/2*h)*(-3*(func(x)+4*func(x+h)-func(x+2*h)))
    return der

y=fx(x)

dy=func_der(x,h,fx)

plt.plot(x,y,color='red')
plt.plot(x,dy,color='green')
plt.grid()
plt.show