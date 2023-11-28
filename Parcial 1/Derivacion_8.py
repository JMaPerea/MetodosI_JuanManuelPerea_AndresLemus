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

x=np.linspace(0.1,1.1,100)
h=x[1]-x[0]
graf= lambda x :   (np.sin(x)/np.cos(x))**0.5
f = graf(x)
 
def DerivativeC(x,f,h):  #derivada central ,
    return (f(x+h)-f(x-h))/(2*h)
DerC= DerivativeC(x,graf,h)
DervidaVerdadera=  lambda f,x,h=0.01 : (1/(np.cos(x))**2) /(2* math.sqrt(np.sin(x)/np.cos(x)))  #sec2(x) / 2 raiz(tanx))
Der = lambda f,x,h=0.01 : (1/(2*h))*(-3*f(x)+4*f(x+h)-f(x+2*h))
X = np.zeros((100))
for i in range(100):
    X[i] = Der(graf,x[i],h=0.01)
XV = np.zeros((100))
for i in range(100):
    XV[i] = Der(graf,x[i],h=0.01)
fig = plt.figure(figsize=(20,8),edgecolor='red')

ax = fig.add_subplot(2,2,1)
plt.title("función y su Derivada real  ")

ax1 = fig.add_subplot(2,2,2)
plt.title ("Derivada Progresiva")
ax2 = fig.add_subplot(2,2,3)
plt.title ("diferencia entre cada punto nodal")
ax3= fig.add_subplot (2,2,4)
plt.title ("derivada central")


ax.scatter(x,f)  #1 grafica de la función sin derivar
ax1.scatter(x,X)#2° derivada progresiva (azul)
ax.scatter(x,XV)# 2° derivada verdadera (naranja)
"diferencia entre cada punto nodal "
ax2.scatter(x,np.abs(XV-X)) # 3° diferencia entre cada punto nodal entre verdadera y progresiva("azul")
ax2.scatter(x,np.abs(XV-DerC)) #diferencia Verdader-central (naranja)
ax3.scatter (x,DerC) #4° Derivada Central