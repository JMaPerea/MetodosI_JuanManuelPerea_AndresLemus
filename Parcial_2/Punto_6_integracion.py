# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 21:51:37 2023

@author: POWER
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
import math

def Simpson_Int(func,a,b,n):
    n=int(n)
    h=(b-a)/n
    x=[]
    fx=[]
    
    for i in range(n+1):
        xi=a+i*h
        x.append(xi)
        fx.append(func(xi))
        
        
        
    Int=fx[0]+fx[n]
    
    for i in range(1,n):
        if i % 2 ==0:
            Int+=2*fx[i]
        else:
            Int+=4*fx[i]
    
    return Int*(h/3)

# funcion= lambda x: x**2
# a=0
# b=1
# n=50
#print(Simpson_Int(funcion, a, b, n))

R=0.5
a=0.05
Induct= lambda x: np.sqrt(a**2-x**2)/R+x
Realsol=np.pi*(R-np.sqrt(R**2-a**2))
Intsol=Simpson_Int(Induct, -a, a, 1e6)
error=(abs(Intsol-Realsol)/Realsol)*100
print(f'La solucion a la integral esperada es {Realsol}, la obtenida usando el metodo es {Intsol} el error es {error}%')
