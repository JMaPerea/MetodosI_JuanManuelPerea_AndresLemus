# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 11:45:19 2023

@author: ADMIN
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
import math
import funciones_integracion as fint

#Denominador
lag=np.polynomial.laguerre.laggauss(20)
#Numerador
leg=np.polynomial.legendre.leggauss(20)


#Denominador valor
funcd= lambda x: (x**3)/(np.exp(x)-1)
gx=lambda x: funcd(x)*np.exp(x)
valintd=fint.Estimar_integral(lag[1], lag[0],gx , 20)
#print(valintd)

#numerador valor 

def obt_limites_int(lam):
    h=6.626e-34
    k=1.3806e-23
    c=3e8
    T=5772
    lam=lam*10e-10
    lim=(h*(c/lam))/(k*T)
    return lim

lim1=obt_limites_int(100)
lim0=obt_limites_int(400)
print(lim0,lim1)

valintn=fint.Estimar_integral_leg(leg[1], leg[0] ,funcd , 20, lim0,lim1)

print(valintn)

f=valintn/valintd
print(f*100)

"""
25 punto teorico

"""

poly=fint.ObtLaguerreRodrigues(2)
raices=fint.Obt_roots_Lag(2)
weights=fint.ObtWeightsLag(2)


print(poly)
print(raices,weights)
funcionteo=lambda x: sym.exp(-x)*x**3
gxft=lambda x: funcionteo(x)*sym.exp(x)
xk=fint.Obt_roots_Lag(3)
ck=fint.ObtWeightsLag(3)
valorteo=fint.Estimar_integral(ck, xk, gxft , 3)
print(valorteo)



    