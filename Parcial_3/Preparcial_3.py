# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 14:46:46 2023

@author: ADMIN
"""

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

th=sym.symbols('theta', real=True)
def func(theta):
    
    C=
    return 

f=func(th)

def newton(fun,th0,maxiter=100,tol=1e-6):
    df=sym.diff(fun,th)
    thn=th0
    f=sym.lambdify(th,fun)
    df=sym.lambdify(th,df)
    
    
    for i in range(maxiter):
        thn=thn-((f(thn))/(df(thn)))
        if abs(thn)<tol:
            break
        
    return thn

root=newton(f,(0.40*2*np.pi))
print(root)