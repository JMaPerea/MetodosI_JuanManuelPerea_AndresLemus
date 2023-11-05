# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 21:47:02 2023

@author: POWER
"""

import numpy as np

def pot_inversa(A, toleranc=1e-6, Iterac=10000):

    x = np.array([1,1,1])
    
    for _ in range(Iterac):
        
        y = np.linalg.solve(A, x)
        val = np.dot(x, y)
        x = y / np.linalg.norm(y)
        
        
    
    val = 1 / val
    
    return val, x


A = np.array([[1,2,-1],[1,0,1], [4,-4,5]])
sol=pot_inversa(A)
print(f"Para 10000 iteraciones:\nValor de Eo: {sol[0]}\nVector asociado: {sol[1]}")