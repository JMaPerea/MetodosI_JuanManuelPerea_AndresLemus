# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 22:05:47 2023

@author: POWER
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
import math

x=sym.Symbol('x',real=True)

def ObtHermiteRodrigues(n,x):
    sys=sym.exp(-x**2)
    if n==0:
        poly=1
    if n==1:
        poly=2*x
    else:
        poly=(-1)**n*sym.exp(x**2)*sym.diff(sys,x,n)
    return sym.expand(poly)


def Obt_roots(n):
    poly=ObtHermiteRodrigues(n, x)
    roots=sym.nroots(poly)
    return roots

def ObtWeightsHerm(n):

    Roots = np.array(Obt_roots(n))

    

    poly = sym.lambdify(x,ObtHermiteRodrigues(n-1, x))   
    
    Weights = (2**(n-1)*sym.factorial(n)*sym.sqrt(np.pi)) / (n**2*poly(Roots)**2)
    
    
    return Weights

def Estimar_integral(ck,xk,func,n):
    sol=0
    
    for i in range(n):
        val=ck[i]*func(xk[i])
        sol+=val
        #print(val)
        
    return sol
    



#punto a

n=20
xk=Obt_roots(n)
ck=ObtWeightsHerm(n)
print(f'Grado: {n} \n Raices: {xk} \n Pesos {ck}')


#punto b

funint= lambda x: (1/sym.sqrt(2) * (1 / np.pi)**(1/4) * sym.exp(-x**2/2) * 2 * x)**2
#transformaciones de la funcion para usar gauss hermite
gx= lambda x: funint(x)*x**2
dx=lambda x: gx(x)*sym.exp(x**2)

ValInt=Estimar_integral(ck, xk, dx, 20)
print(f'El valor estimado de la integral  es : {ValInt}')
