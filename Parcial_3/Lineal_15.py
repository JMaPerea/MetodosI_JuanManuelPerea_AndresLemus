# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 12:21:04 2023

@author: POWER
"""

import sympy as sym

σx = sym.Matrix([[0, 1], [1, 0]])
σy = sym.Matrix([[0, -sym.I], [sym.I, 0]])
σz = sym.Matrix([[1, 0], [0, -1]])


ε = sym.symbols('ε')


σ = [σx, σy, σz]

for i in range(3):
    for j in range(3):
        comm = σ[i] * σ[j] - σ[j] * σ[i]
        epsilon_valor = 2 * sym.I * ε * σ[(i + 1) % 3]
        simplif=sym.simplify(comm - epsilon_valor)
        
        print(f"i={i+1}  j={j+1}  \n[σ{i+1}, σ{j+1}] = 2iε_{i+1}{(i+1) % 3 + 1}{j+1}σ{(i + 1) % 3 + 1}:  \n\n", simplif,"\n\n" )
