# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 17:59:59 2023

@author: ADMIN
"""

import numpy as np
import matplotlib.pyplot as plt

import sympy as sym
import math




def fsemiesf(x,y):
    
    if (x**2+y**2)<=1:
        
        return np.sqrt(1-(x**2-y**2))
    
    else:
        
        return 0
    
def Volumen(n,a,b,R):
    vol=0
    
    h=(b-a)/n
    
    for i in range(n):
        for j in range(n):
            
            x=-R+(i*0.5)*h
            
            y=-R+(j*0.5)*h
            
            valprom = (fsemiesf(x - 0.5 * h, y - 0.5 * h) +
                             fsemiesf(x + 0.5 * h, y - 0.5 * h) +
                             fsemiesf(x - 0.5 * h, y + 0.5 * h) +
                             fsemiesf(x + 0.5 * h, y + 0.5 * h)) / 4.0
            
            vol+=valprom*h**2
    return vol

R=1
a=-R
b=R
n=1000

print(Volumen(n, a, b, R))
    






        
    



