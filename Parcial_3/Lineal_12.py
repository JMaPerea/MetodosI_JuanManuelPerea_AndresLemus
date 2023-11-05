# -*- coding: utf-8 -*-
"""
Created on Sun Nov  5 15:36:54 2023

@author: POWER
"""
import numpy as np
import sympy as sym

x1,x2,x3=sym.symbols('x1 x2 x3', Real=True)

def func1(x1,x2):
    f1=sym.log(x1+x2)-sym.sin(x1*x2)-sym.log(2)-sym.log(np.pi)
    f2=sym.exp(x1-x2)+sym.cos(x1*x2)
    return sym.Matrix([f1,f2])

#print(func(x1,x2))
def func2(x1,x2,x3):
    f1=6*x1 - 2 * sym.cos(x2 * x3) - 1
    f2=9 * x2 + sym.sqrt(x1**2 + sym.sin(x3) + 1.06) + 0.9
    f3=60 * x3 + 3 * sym.exp(-x1 * x2) + 10 * sym.pi - 3
    return sym.Matrix([f1,f2,f3])

def jacob1(f):
    df1_x1=sym.diff(f[0],x1)
    df1_x2=sym.diff(f[0],x2)
    df2_x1=sym.diff(f[1],x1)
    df2_x2=sym.diff(f[1],x2)
    return sym.Matrix([[df1_x1,df2_x1],[df1_x2,df2_x2]])

#mat=sym.Matrix(jacob(func(x1,x2)))
#sym.init_printing(use_latex=True)

def jacob2(f):
    df1_x1=sym.diff(f[0],x1)
    df1_x2=sym.diff(f[0],x2)
    df1_x3=sym.diff(f[0],x3)
    df2_x1=sym.diff(f[1],x1)
    df2_x2=sym.diff(f[1],x2)
    df2_x3=sym.diff(f[1],x3)
    df3_x1=sym.diff(f[2],x1)
    df3_x2=sym.diff(f[2],x2)
    df3_x3=sym.diff(f[2],x3)
    return sym.Matrix([[df1_x1,df2_x1,df3_x1],[df1_x2,df2_x2,df3_x2],[df1_x3,df2_x3,df3_x3]])


def dec_grad(f, J, x0, a, tol, maxit):
    f_np=sym.lambdify([[x1,x2]],f,'numpy')
    
    J_np=sym.lambdify([[x1,x2]],J,'numpy')
    x = x0
    #print(f_np(x))
    
    for i in range(maxit):
        grad = np.matmul(-J_np(x).T,  f_np(x).reshape(-1, 1).flatten())
        x = x + a * grad
        
        if np.linalg.norm(grad) < tol:
            break
    return x

def newton_raphson(f, J, x0, tol, maxit):
    f_np=sym.lambdify([[x1,x2]],f,'numpy')
    
    J_np=sym.lambdify([[x1,x2]],J,'numpy')
    
    x = x0
    
    for i in range(maxit):
        delta_x = np.linalg.solve(J_np(x), -f_np(x).reshape(-1, 1).flatten())
        x = x + delta_x
        
        if np.linalg.norm(delta_x) < tol:
            break
        
    return x

def dec_grad2(f, J, x0, a, tol, maxit):
    f_np=sym.lambdify([[x1,x2,x3]],f,'numpy')
    
    J_np=sym.lambdify([[x1,x2,x3]],J,'numpy')
    x = x0
    #print(x0)
    #print(f_np(x))
    
    for i in range(maxit):
        grad = np.matmul(-J_np(x).T,  f_np(x).reshape(1,-1).flatten())
        #print(grad)
        x = x + a * grad
        #print(x)
        if np.linalg.norm(grad) < tol:
            break
    
    return x

def newton_raphson2(f, J, x0, tol, maxit):
    f_np=sym.lambdify([[x1,x2,x3]],f,'numpy')
    
    J_np=sym.lambdify([[x1,x2,x3]],J,'numpy')
    
    x = x0
    #print(x0)
    for i in range(maxit):
        delta_x = np.linalg.solve(J_np(x), -f_np(x).reshape(1, -1).flatten())
        x = x + delta_x
        
        if np.linalg.norm(delta_x) < tol:
            break
        
    return x

f=func1(x1, x2)
J=jacob1(f)
tol=1e-10
maxit=10000
a=1e-6
x0=np.array([2,2])


#pru=sym.lambdify([[x1,x2]],J,'numpy')
#fpru=sym.lambdify([[x1,x2]],f,'numpy')
#print(np.linalg.solve(pru(x0),fpru(x0)))
#print(sym.lambdify([[x1,x2]],J,'numpy'))


est_inicial=dec_grad(f, J, x0, a, tol, maxit)
sol_final=newton_raphson(f, J, est_inicial, tol, maxit)

print(f'La solucion del sistema 1 dado x0=[2,2] es: \n [x1,x2] \n {sol_final}')

f2=func2(x1, x2, x3)
J2=jacob2(f2)



xi2=np.array([0,0,0])


est_inicial2=dec_grad2(f2, J2, xi2, a, tol, maxit)
sol_final2=newton_raphson2(f2, J2, est_inicial2, tol, maxit)
print(f'La solucion del sistema 2 dado x0=[0,0,0] es: \n [x1,x2,x3] \n {sol_final2}')
