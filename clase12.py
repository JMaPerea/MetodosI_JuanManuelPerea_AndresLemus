# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 17:08:49 2023

@author: ADMIN
"""

import numpy as np
import matplotlib.pyplot as plt

import sympy as sym
import math

x=sym.Symbol('x',real=True)

def Obtlegendre(n,x):
    sys=(x**2-1)**n
    if n==0:
        poly=1
    if n==1:
        poly=x
    else:
        poly=1/(2**n*sym.factorial(n))*sym.diff(sys,x,n)
        
    return sym.expand(poly)



def Obt_roots_weight(n):
    
    poly=Obtlegendre(n, x)
    
    roots=np.array(sym.nroots(poly))
    
    Dpoly = sym.lambdify(x,sym.diff(poly,x))
    Weights = 2/((1-roots**2)*Dpoly(roots)**2)
    
    
    
    return Weights, roots

def Estimar_integral(ck,xk,func,n,a,b):
    sol=0
    for i in range(n):
        val=ck[i]*func(xk[i]*((b-a)/2+(b+a)/2))
        sol+=val
        val
    return sol

k=3
func=lambda x: -x+5
xk=Obt_roots_weight(k)[1]
ck=Obt_roots_weight(k)[0]
print(ck)
print(Estimar_integral(ck, xk, func, k, -100, 200))


    
    