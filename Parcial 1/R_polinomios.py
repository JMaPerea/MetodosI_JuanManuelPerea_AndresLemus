import numpy as np
# Newton-Raphson
def Newton_raphson(funcion,derivada,x_0):
    iteraciones_max=100
    tolerancia=1e-6
    x=x_0
    for i in range (iteraciones_max):
        x_next= x - funcion(x)/derivada(x)
        error_relativo= abs(x_next-x)/abs(x_next)
        if error_relativo < tolerancia:
            return x_next, "menor que el error relativo en la iteración:", i
        x=x_next

#Raices de polinomios
def funcion (x):    
    funcion= 3*x**5 + 5*x**4 -x**3
    return funcion
def derivada(x):
    derivada=15*x**4 + 20*x**3 -3*x**2
    return derivada
"""print(Newton_raphson(funcion, derivada, 2.5))"""
""""
Polinomios_legendre= np.zeros(20)
derivadas_Polinomios=np.zeros(20)
def polinomio (x):
    for i in range (1,21):
        polinomios
    polinomio"""