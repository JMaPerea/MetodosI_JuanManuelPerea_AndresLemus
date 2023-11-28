# -*- coding: utf-8 -*-
"""
Created on Wed Sep 27 12:37:33 2023

@author: ADMIN
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
import math

x = sym.Symbol('x',real=True)
y = sym.Symbol('y',real=True)

def ObtLaguerreRodrigues(n,x):
    sys=sym.exp(-x)*x**n
    if n==0:
        poly=1
    if n==1:
        poly=-x+1
    else:
        sys=sym.exp(-x)*x**n
        poly=((sym.exp(x))/(sym.factorial(n)))*sym.diff(sys,x,n)
    
        
            
    return poly







def ObtPolinomiosRaices(n,x):
    
    polinomios=[]
    
    roots=[]
    for i in range(n+1):
        polinomios.append(ObtLaguerreRodrigues(i, x))
        
    i=0
    for k in polinomios:
        if k==1:
            root=None
        else:
            root=sym.nroots(k)
            
        roots.append(root)
        
        print(f'Grado {i} Raices {root}')
        i+=1
    return roots
     

ObtPolinomiosRaices(20, x)

