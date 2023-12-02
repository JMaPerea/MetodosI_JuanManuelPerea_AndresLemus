import sympy as sym
from sympy import symbols, integrate, Piecewise, solve


x, y = symbols('x y')

# Definición de la función de densidad conjunta
f = Piecewise((2/3 * (x + 2*y), (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)), (0, True))
resultado = integrate(f, (x, -float('inf'), float('inf')), (y, -float('inf'), float('inf')))
print("punto A")
print(resultado)


def punto_b():
    x, y = symbols('x y')

    funcion = Piecewise((2/3 * (x + 2*y), (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)), (0, True))
    g_de_x = integrate(funcion, (y, -float('inf'), float('inf')))
    h_de_y = integrate(funcion, (x, -float('inf'), float('inf')))
    print("respuest punto B")
    print("resultado g(x):", g_de_x )
    print("resultado h(y):", h_de_y)
    
punto_b()

def punto_c():
    x, y = symbols('x y')

    funcion = Piecewise((2/3 * (x + 2*y), (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)), (0, True))
    g_de_x = integrate(funcion, (y, -float('inf'), float('inf')))
    E_x = sym.integrate(x*g_de_x,(x,0,1))
    print("Respuesta punto c")
    print("E(x) =",E_x)

punto_c()

def punto_d():
    x, y = symbols('x y')

    funcion = Piecewise((2/3 * (x + 2*y), (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)), (0, True))
   
    h_de_y = integrate(funcion, (x, -float('inf'), float('inf')))
    E_y = sym.integrate(y*h_de_y,(y,0,1))
    print("E(y) =",E_y)
punto_d()

def punto_e():
    x, y = symbols('x y')
    funcion = Piecewise((2/3 * (x + 2*y), (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)), (0, True))
    Integral_porx = sym.integrate(x*funcion,(x,0,1))
    integral_pory = sym.integrate(y*Integral_porx,(y,0,1))
    g_de_x = integrate(funcion, (y, -float('inf'), float('inf')))
    h_de_y = integrate(funcion, (x, -float('inf'), float('inf')))
    E_x = sym.integrate(x*g_de_x,(x,0,1))
    E_y = sym.integrate(y*h_de_y,(y,0,1))
    Covar = integral_pory - (E_x*E_y)
    print("punto e, respuesta:", Covar)
punto_e()

def punto_f():
    x, y = symbols('x y')
    funcion = Piecewise((2/3 * (x + 2*y), (x >= 0) & (x <= 1) & (y >= 0) & (y <= 1)), (0, True))
    g_de_x = integrate(funcion, (y, -float('inf'), float('inf')))
    h_de_y = integrate(funcion, (x, -float('inf'), float('inf')))
    E_x = sym.integrate(x*g_de_x,(x,0,1))
    E_y = sym.integrate(y*h_de_y,(y,0,1))
    IntInterna = sym.integrate((x-E_x)*funcion,(x,0,1)) 
    CovarianzaB=sym.integrate(IntInterna*(y-E_y),(y,0,1))

    print("punto f, respuesta:", CovarianzaB)
punto_f()

#Punto G teorico:
""" 
No son linealmente independientes, covarianza distinta de 0, tal como se indica en el inciso e y f.
"""









