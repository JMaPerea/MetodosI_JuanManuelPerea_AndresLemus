# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 17:05:32 2023

@author: ADMIN
"""

import numpy as np
import sympy as sym

x0,x1,x2,x3,x4=sym.symbols('x0 x1 x2 x3 x4',real=True)

A=sym.Matrix([[0.2,0.1,1,1,0]
              ,[0.1,4,-1,1,-1],
              [1,-1,60,0,-2],
              [1,1,0,8,4],
              [0,-1,-2,4,700]])
x=sym.Matrix([[x0],[x1],[x2],[x3],[x4]])
b=sym.Matrix([[1],[2],[3],[4],[5]])
#sym.init_printing(use_latex=True)
x_0=sym.ones(5,1)
#print(x_0)
def descenso_conj(A,b,x_0,tol=0.01):
    r0=A*x_0-b
    p0=-r0
    k=0
    rk=r0
    pk=p0
    xk=x_0
    while (rk.norm(5)) >tol:
        anum=(rk.T*pk)
        aden=(pk.T*A*pk)
        alphak=anum[0]/aden[0]
        xk=xk-(alphak*pk)
        rk=A*xk-b
        bnum=(rk.T*A*pk)
        bden=(pk.T*A*pk)
        betak=bnum[0]/bden[0]
        pk=-rk+betak*pk
        k=k+1
        print(f'num iteraciones es {k}')
    return np.array(xk)

root=descenso_conj(A, b, x_0)



print(root)
A=np.array([[0.2,0.1,1,1,0]
              ,[0.1,4,-1,1,-1],
              [1,-1,60,0,-2],
              [1,1,0,8,4],
              [0,-1,-2,4,700]])
b=np.array([[1],[2],[3],[4],[5]])
solreal=np.linalg.solve(A,b)
print('\n',solreal)        
        