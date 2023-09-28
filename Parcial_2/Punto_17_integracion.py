# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:40:49 2023

@author: POWER
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
import math

x = sym.Symbol('x',real=True)

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


def Obt_roots(n):
    poly=ObtLaguerreRodrigues(n, x)
    roots=sym.nroots(poly)
    return roots

def ObtWeightsLag(n):

    Roots = np.array(Obt_roots(n))

    

    poly = sym.lambdify(x,ObtLaguerreRodrigues(n+1, x))   
    
    Weights = Roots / ((n+1)**2*(poly(Roots))**2)
    
    
    return Weights


def Estimar_integral(ck,xk,func,n):
    sol=0
    
    for i in range(n):
        val=ck[i]*func(xk[i])
        sol=sol+val
        
    return sol
    
def Valor(k):
    
    
    funcion=lambda x: (x**3)/(sym.exp(x)-1)
    gx=lambda x: funcion(x)*sym.exp(x)
    
    
    xk=Obt_roots(k)
    ck=ObtWeightsLag(k)
    ValInt=Estimar_integral(ck, xk, gx, k)
    
    
    return ValInt

def Error(k):
    ValInt=Valor(k)
    ValTeo=(np.pi**4)/15
    error=abs(ValInt-ValTeo)/ValTeo
    return error

# punto a
print(f'El valor de la integral estimada para n=3 es: {Valor(3)}')

#punto b

listerr=[]

for i in range(3,11):

    err=Error(i)
    listerr.append(err)

    
n=np.arange(3,11)
plt.plot(n,listerr)
plt.title('Error por numero de puntos de evaluación')
plt.xlabel('n punto de evaluación')
plt.ylabel('error')
plt.grid()
plt.show





