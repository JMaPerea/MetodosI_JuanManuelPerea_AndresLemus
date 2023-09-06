# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 17:05:02 2023

@author: ADMIN
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
import math

x= np.linspace(-2,2,num=20)

func= lambda x: (np.exp(-x))-x
plt.plot(x,func(x))
plt.grid()

"""
funcion retorna raiz=x3
"""
def muller(x0,x1,x2,func,max_it=100,tol=1e-10,):
    for i in range(max_it):
        h0=x1-x0
        h1=x2-x1
        func0=func(x0)
        func1=func(x1)
        func2=func(x2)
        div0= (func1-func0)/h0
        div1= (func2-func1)/h1
        a=(div1-div0)/(h1+h0)
        b=a*h1+div1
        c=func2
        disc=b**2-4*a*c
        x3_pos=(x2-((2*c))/(b+np.sqrt(disc)))
        x3_neg=(x2-((2*c))/(b-np.sqrt(disc)))
        
        if abs(x3_pos-x2) < abs(x3_neg-x2):
            x3=x3_pos
        
        else:
            x3=x3_neg
        """
        Criterio de parada
        
        """
        if abs((x3-x2)/x3) < tol:
            return x3,'Iteraciones',i
        else:
            x0=x1
            x1=x2
            x2=x3
        
"""
Elgimos el intervalo de [0,1] ya que al evaluar la función en los extremos
obtemos signos opuestos lo que indica que hay una raiz en el intervalo
x1 es 1/2 dado el intervalo [0,1]


"""
x0,x1,x2=0,0.5,1

print(muller(x0,x1,x2,func))


        
            
        
        
        
        
    