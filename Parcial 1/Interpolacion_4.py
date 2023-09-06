# -*- coding: utf-8 -*-
"""
Created on Wed Sep  6 15:28:33 2023

@author: ADMIN
"""

import numpy as np
import sympy as sym
import pandas as pd
#import os.path as path
#import wget
g = -9.8
Path_="Parabolico.csv"
# url = 'https://raw.githubusercontent.com/asegura4488/Database/main/MetodosComputacionalesReforma/Parabolico.csv'
# if not path.exists(file):
#     Path_ = wget.download(url,file)
# else:
#     print('--File found---')
#     Path_ = file

Data = pd.read_csv(Path_,sep=',')
X = np.float64(Data['X'])
Y = np.float64(Data['Y'])
def Lagrange(x,xi,j):
    prod = 1.0
    n = len(xi)
    for i in range(n):
        if i != j:
            prod *= (x - xi[i])/(xi[j]-xi[i])
    return prod
def Poly(x,xi,yi):
    Sum = 0.
    n = len(xi)
    for j in range(n):
        Sum += yi[j]*Lagrange(x,xi,j)
    return Sum
x = sym.Symbol('x')
f = Poly(x,X,Y)
f = sym.expand(f)
w = np.polyfit(X,Y,len(X)-1)
Vx = (g/(2*w[0]))**0.5
Vy = Vx*w[1]
Vo = (Vx**2 + Vy**2)**0.5
round(Vo,2)
θ = np.degrees(np.arctan(Vy/Vx))
round(θ,2)

print(Vo,θ)