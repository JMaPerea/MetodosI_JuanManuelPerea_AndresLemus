# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:33:56 2023

@author: POWER
"""

import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

x=sym.symbols('x',real=True)
y=sym.symbols('y',real=True)
z=x+sym.I*y
vari=[x,y]

def fun(z):
    return (z**3)-1

F=[sym.re(fun(z)),sym.im(fun(z))]

def jacob(F,var):
    M=sym.zeros(len(F),len(var))
    for i in range(len(F)):
        for h in range(len(var)):
            df=sym.diff(F[i],var[h])
            M[i,h]=df
    return M

def newton(z0,F,J,maxiter=100,tol=1e-7):
    zn=z0
    lz=[z0]
    for i in range(maxiter):
        #print(J(zn[0],zn[1]))
        Jinv=np.linalg.inv(J(zn[0],zn[1]))
        Fzn=F(zn[0],zn[1])
        zn=zn-np.dot(Jinv,Fzn)
        lz.append(zn)
        
        if abs(np.linalg.norm(lz[i+1])-np.linalg.norm(lz[i]))<tol:
            break
        
    return zn
    
    return zn

def frac_build(xv,yv,rootlist,F,J,N):
    Frac=np.zeros((N,N), np.int64)
    for i in range(N):
        for j in range(N):
            z0=np.array([xv[i],yv[j]])
            
            root=newton(z0, F, J)
            imroot=root[1]
            #print(imroot)
            
            
            for z in rootlist:
                min_dist=abs(imroot-z)
                if z==rootlist[0]:
                    if min_dist < 1e-3:
                        #print(f'el punto {imroot} esta a distancia {min_dist} de la raiz {z}')
                        Frac[i, j] = 20
                    
                        
                        
                if z==rootlist[1]:
                    
                    if min_dist < 1e-3:
                    
                        Frac[i, j] = 100
                        
                
                
            # if imroot==rootlist[0]:
            #     Frac[i,j]=100
            # if imroot==rootlist[1]:
            #     Frac[i,j]=20
            # else:
            #     Frac[i,j]=255
    
    
    return Frac
J=jacob(F,vari)
Fn=sym.lambdify([x,y],F,'numpy')
Jac=sym.lambdify([x,y],J,'numpy')

N=300
xv=np.linspace(-1,1,N)
yv=np.linspace(-1,1,N)
rootlist=[0.8660254037844387,-0.8660254037844387,0]
#Fractal=np.zeros((N,N), np.int64)



r=frac_build(xv, yv, rootlist, Fn, Jac,N)

plt.imshow(r, cmap='coolwarm' ,extent=[-1,1,-1,1])

r=sym.Matrix(r)


#z0=np.array([0.5,0.5])
#root=newton(z0,Fn,Jac)
#print(root)
sym.init_printing(use_latex=True)


