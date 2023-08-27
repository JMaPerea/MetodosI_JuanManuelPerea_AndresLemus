"taller 1"
import numpy as np
import matplotlib.pyplot as plt

"""
Ejercicio Derivacion 6

"""

x=np.linspace(-4, 4,num=25)
y=np.linspace(-4, 4,num=25)
R=2
V=2
h=0.001

fl=lambda x,y: V*x*(1-R**2/(x**2+y**2))

Ind= lambda x,y: (x**2+y**2)<R

def potflujo(x,y,fl,h,M=None):
    
    X,Y=np.meshgrid(x,y)
    if M:
        X,Y=((np.ma.masked_where(M(X,Y), X)),(np.ma.masked_where(M(X,Y), Y)))
    Vx=(fl(X+h,Y)-fl(X,Y))/(2*h)
    Vy=(fl(X,Y-h)-fl(X,Y))/(2*h)
    plt.figure(figsize=(4,4),dpi=100)
    plt.quiver(X,Y,Vx,Vy,scale=30, width=.008,color="Blue")
    
    return plt.show
#potflujo(x, y, fl, h,Ind)


"""
Ejercicio Derivacion 7

"""
x1=np.linspace(0, 1, num=25)
y1=np.linspace(0, 1, num=25)
r=np.array((0.51, 0.21), dtype=float)
d=r[1]
Vp= lambda x,y: (1/np.sqrt((x-r[0])**2+(y-d)**2))-(1/np.sqrt((x-r[0])**2+(y+d)**2))
#print(Vp(x,y))
Ind1= lambda x,y: np.sqrt((x-r[0])**2+(y-d)**2)<0.1
Ind2= lambda x,y: np.sqrt((x-r[0])**2+(y+d)**2)<0.1


def vflujo(x,y,fn,M=None, M2=None):
    X,Y=np.meshgrid(x,y)
    if M:
        X,Y=((np.ma.masked_where(M(X,Y), X)),(np.ma.masked_where(M(X,Y), Y)))
        X,Y=((np.ma.masked_where(M2(X,Y), X)),(np.ma.masked_where(M2(X,Y), Y)))
    Vx=(fn(X+h,Y)-fn(X,Y))/(h)
    Vy=(fn(X,Y+h)-fn(X,Y))/(h)
    plt.quiver(X,Y,-Vx,-Vy,scale=400, width=.003,color="Red")
vflujo(x1, y1, Vp, Ind1,Ind2)
