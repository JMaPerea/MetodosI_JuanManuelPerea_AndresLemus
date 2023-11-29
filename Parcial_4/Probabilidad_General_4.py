# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 21:52:51 2023

@author: POWER
"""

import matplotlib.pyplot as plt


def calcular_probabilidad_diferentes(n):
    probabilidad = 1.0
    for i in range(n):
        probabilidad *= (365 - i) / 365
    return probabilidad


n_valores = list(range(1, 81))
probabilidades=[]
for n in n_valores:
    
    probabilidades.append(calcular_probabilidad_diferentes(n))


plt.plot(n_valores, probabilidades, marker='o')
plt.xlabel('n')
plt.ylabel('Pn')
plt.grid(True)
plt.show()
