# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:32:08 2023

@author: ADMIN
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
import math

x=sym.Symbol('x',real=True)

def ObtLaguerreRodrigues(n):
    sys=sym.exp(-x)*x**n
    if n==0:
        poly=1
    if n==1:
        poly=-x+1
    else:
        sys=sym.exp(-x)*x**n
        poly=((sym.exp(x))/(sym.factorial(n)))*sym.diff(sys,x,n)
    return poly


def Obt_roots_Lag(n):
    poly=ObtLaguerreRodrigues(n)
    roots=sym.nroots(poly)
    return roots

def ObtWeightsLag(n):

    Roots = np.array(Obt_roots_Lag(n))
    
    poly = sym.lambdify(x,ObtLaguerreRodrigues(n+1))   
    
    Weights = Roots / ((n+1)**2*(poly(Roots))**2)
    
    
    return Weights


def ObtHermiteRodrigues(n):
    sys=sym.exp(-x**2)
    if n==0:
        poly=1
    if n==1:
        poly=2*x
    else:
        poly=(-1)**n*sym.exp(x**2)*sym.diff(sys,x,n)
    return sym.expand(poly)


def Obt_roots_Herm(n):
    poly=ObtHermiteRodrigues(n)
    roots=sym.nroots(poly)
    return roots

def ObtWeightsHerm(n):

    Roots = np.array(Obt_roots_Herm(n))

    

    poly = sym.lambdify(x,ObtHermiteRodrigues(n-1))   
    
    Weights = (2**(n-1)*sym.factorial(n)*sym.sqrt(np.pi)) / (n**2*poly(Roots)**2)
    
    
    return Weights

def ObtLegendreRodrigues(n):
    sys=(x**2-1)**n
    if n==0:
        poly=1
    if n==1:
        poly=x
    else:
        poly=1/(2**n*sym.factorial(n))*sym.diff(sys,x,n)
        
    return sym.expand(poly)

def Obt_roots_Leg(n):
    poly=ObtLegendreRodrigues(n)
    roots=sym.nroots(poly)
    return roots

def ObtWeightsLeg(n):
    
    poly=ObtLegendreRodrigues(n)
    
    roots=Obt_roots_Leg(n)
    
    Dpoly = sym.lambdify(x,sym.diff(poly,x))
    Weights = 2/((1-roots**2)*Dpoly(roots)**2)
    return Weights

def Estimar_integral(ck,xk,func,n):
    sol=0
    
    for i in range(n):
        val=ck[i]*func(xk[i])
        sol+=val
        #print(val)
        
    return sol

def Estimar_integral_leg(ck,xk,func,n,a,b):
    sol=0
    
    for i in range(n):
        
        val=ck[i]*func((((b-a)/2)*xk[i])+((a+b)/2))
        sol+=val
        #print(val)
        
    return ((b-a)/2)*sol



    