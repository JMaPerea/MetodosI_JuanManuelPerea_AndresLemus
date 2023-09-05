"taller 1"
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
import math
"""
Ejercicio Derivacion 6

"""

x=np.linspace(-4, 4,num=25)
y=np.linspace(-4, 4,num=25)
R=2
V=2
h=0.001

fl=lambda x,y: V*x*(1-(R**2/(x**2+y**2)))

Ind= lambda x,y: (x**2+y**2)<2*R

def potflujo(x,y,fl,h,M=None):
    
    X,Y=np.meshgrid(x,y)
    if M:
        X,Y=((np.ma.masked_where(M(X,Y), X)),(np.ma.masked_where(M(X,Y), Y)))
    Vx=(fl(X+h,Y)-fl(X,Y))/(2*h)
    Vy=(fl(X,Y-h)-fl(X,Y))/(2*h)
    
    plt.figure(figsize=(4,4),dpi=100)
    
    plt.quiver(X,Y,Vx,Vy,scale=30, width=.008,color="Blue")
    
    return None



"""
Ejercicio Derivacion 7

"""
x1=np.linspace(0, 1, num=25)
y1=np.linspace(0, 1, num=25)
r=np.array((0.51, 0.21), dtype=float)
d=r[1]
Vp= lambda x,y: (1/np.sqrt((x-r[0])**2+(y-d)**2))-(1/np.sqrt((x-r[0])**2+(y+d)**2))

z=np.zeros(25)


def vflujo(x,y,fn):
    X,Y=np.meshgrid(x,y)
    
    Vx=(fn(X+h,Y)-fn(X,Y))/(h)
    Vy=(fn(X,Y+h)-fn(X,Y))/(h)
    Vx= Vx / np.sqrt(Vx**2 + Vy**2)
    Vy= Vy / np.sqrt(Vx**2 + Vy**2)
    plt.xlim(0, 1)
    plt.ylim(-0.02, 1)
    plt.plot(r[0], r[1], marker="o", markersize=5, markeredgecolor="Blue", markerfacecolor="Blue")
    plt.plot(x1,z, color="Black")
    plt.quiver(X,Y,-Vx,-Vy,scale=30, width=.005,color="Red")
    
    return None


#vflujo(x1, y1, Vp)
#potflujo(x, y, fl, h,Ind)

"""
Ejercicio interpolación 4

"""
par=pd.read_csv("Parabolico.csv")
X=[]
Y=[]
for i,o in zip(par['X'],par['Y']):
    X.append(i)
    Y.append(o)
X=np.array(X)
Y=np.array(Y)

#print(X,Y)

def Lagrange(x,X,i):   
    L = 1    
    for j in range(X.shape[0]):
        if i != j:
            L *= (x - X[j])/(X[i]-X[j])
            
    return L


def Interpolate(x,X,Y):    
    Poly = 0
    for i in range(X.shape[0]):
        Poly += Lagrange(x,X,i)*Y[i]
        
    return Poly
x=np.linspace(1,6,num=100)






_x = sym.Symbol('x',real=True)
y_x = Interpolate(_x,X,Y)
y_x = sym.simplify(y_x)
#dy_x = sym.diff(y_x,_x,1)
#d2y_x = sym.diff(y_x,_x,2)
val=sym.Poly(y_x)
val=val.coeffs()
print(val)

"""
Dada la operación en el punto 4 de la sección interpolacion en Taller 1.pdf
Sabemos que
val[0]=tan(theta)
val[1]=g/2vo^2cos^s(theta)

"""
g=9.8
theta=math.atan(-val[0])
vo=(g/val[1])/2*((np.cos(theta))**2)
print(vo,theta*180/np.pi)



