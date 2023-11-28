# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 18:34:06 2023

@author: ADMIN
"""




import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

th=sym.symbols('theta', real=True)
def func(theta):
    
    k=9e9
    q=2e-4
    w=114.6
    return ((-k*q**2)/(50*sym.sin(theta)**2))-((k*q**2)/(20*sym.sqrt(2)*sym.sin(theta)**2))+(w*sym.sqrt(2)/2)*sym.tan(theta)

f=func(th)

def newton(fun,th0,maxiter=100,tol=1e-6):
    df=sym.diff(fun,th)
    thn=th0
    f=sym.lambdify(th,fun)
    df=sym.lambdify(th,df)
    
    
    for i in range(maxiter):
        thn=thn-((f(thn))/(df(thn)))
        if abs(thn)<tol:
            break
        
    return thn



#
angulo_equlibrio=newton(f,36)
#print(root)
eq=(np.sin(angulo_equlibrio)**6)+0.0729*(np.sin(angulo_equlibrio)**2)
print(f'{eq}')

o1,o2,o3,o4,o5,o6=sym.symbols("o1 o2 o3 o4 o5 o6")
varilist=[o1,o2,o3,o4,o5,o6]
x0=np.ones(len(varilist))

def func(varilist,C=0.0729):
    M=sym.zeros(len(varilist),len(varilist))
    for i in range(len(varilist)):
        for j in range(len(varilist)):
            f=(sym.sin(varilist[i])**6)+C*(sym.sin(varilist[j])**2)-C
            M[i,j]=f
    return M

M=func(varilist)
Jacob = sym.Matrix([[func(varilist)[i, j].diff(var) 
                        for var in varilist] 
                       for i in range(len(varilist) )
                       for j in range(len(varilist))])
    
def newton_raphson(f, J, x0,varilist, tol, maxit):
    f_np=sym.lambdify([varilist],f,'numpy')
    
    J_np=sym.lambdify([varilist],J,'numpy')
    x = x0
    #print(f_np(x))
    a=0.01
    for i in range(maxit):
        grad = np.matmul(-J_np(x).T,  f_np(x).reshape(-1, 1).flatten())
        x = x + a * grad
        
        if np.linalg.norm(grad) < tol:
            break
    return x             
             
    return x
roots=newton_raphson(M, Jacob, x0, varilist, 1e-6, 100)
roots=roots/(np.pi)
roots=roots*180

print(roots)
        
    