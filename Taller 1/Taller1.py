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

fl=lambda x,y: V*x*(1-((R**2)/(x**2+y**2)))
#fl=lambda x,y: x
Ind= lambda x,y: x**2+y**2<R
def potflujo(x,y,fl,h,M=None):
    X,Y=np.meshgrid(x,y)
    if M:
        X,Y=((np.ma.masked_where(M(X,Y), X)),(np.ma.masked_where(M(X,Y), Y)))
    Vx=(fl(X+h,Y)-fl(X,Y))/(2*h)
    Vy=(fl(X,Y+h)-fl(X,Y))/(2*h)
    plt.figure(figsize=(4,4),dpi=100)
    plt.quiver(X,Y,Vx,Vy,scale=30, width=.005,color="Blue")
    #plt.contour(X,Y,fl(X,Y),levels=20)
    return plt.show
potflujo(x, y, fl, h,Ind)
    

