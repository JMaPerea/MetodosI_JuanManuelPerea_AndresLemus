# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 12:48:53 2023

@author: POWER
"""

import sympy as sym


x, y, z, λ = sym.symbols('x y z λ')
f = x**2 + y**2 + z**2 - 2*z + 1
g = 2*x - 4*y + 5*z - 2


L = f - λ*g


dL_dx = sym.diff(L, x)
dL_dy = sym.diff(L, y)
dL_dz = sym.diff(L, z)
dL_dλ = sym.diff(L, λ)


sistema_ecuaciones = [sym.Eq(dL_dx, 0), sym.Eq(dL_dy, 0), sym.Eq(dL_dz, 0), sym.Eq(dL_dλ, 0)]
soluciones = sym.solve(sistema_ecuaciones, (x, y, z, λ), dict=True)


print("Soluciones:")
for solucion in soluciones:
    print(solucion)


valores_f = [f.subs({x: sol[x], y: sol[y], z: sol[z]}) for sol in soluciones]
minimo = min(valores_f)

print("\nEl mínimo de la función f(x, y, z) sujeto a la restricción g(x, y, z) es:", minimo)

