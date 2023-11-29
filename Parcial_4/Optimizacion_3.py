# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 16:34:16 2023

@author: POWER
"""



import scipy as sci
import sympy as sym


def  fun_volumen(dim):
    x,y,z=dim
    return -x*y*z

def rest_area(dim,max_area):
    x,y,z=dim
    return 2*x*y+2*z*y+x*z-max_area

dim_inicial = [1.0, 1.0, 1.0]


target_area = 12.0
constraint = {'type': 'eq', 'fun': rest_area, 'args': (target_area,)}


bounds = ((0, None), (0, None), (0, None))


result = sci.optimize.minimize(fun_volumen, dim_inicial, method='SLSQP', constraints=[constraint], bounds=bounds)


dimensionesmax = result.x

volumenmax = -result.fun  


print("Usando scipy.optimize \n","Dimensiones maximas:", dimensionesmax)
print("Maximo volumen:", volumenmax)


x, y, z, λ = sym.symbols('x y z λ')
V = -x*y*z
A = x*z+2*x*y+2*z*y-12


L = V + λ*A


dL_dx = sym.diff(L, x)
dL_dy = sym.diff(L, y)
dL_dz = sym.diff(L, z)
dL_dλ = sym.diff(L, λ)

print("\n Usando Lagrange")
sistema_ecuaciones = [sym.Eq(dL_dx, 0), sym.Eq(dL_dy, 0), sym.Eq(dL_dz, 0), sym.Eq(dL_dλ, 0)]
soluciones = sym.solve(sistema_ecuaciones, (x, y, z, λ), dict=True)


print("Soluciones:")
for solucion in soluciones:
    print(solucion)


valores_f = [V.subs({x: sol[x], y: sol[y], z: sol[z]}) for sol in soluciones]
minimo = min(valores_f)


print("\nEl maxímo de la función V sujeto a la restricción A es:", -minimo)


