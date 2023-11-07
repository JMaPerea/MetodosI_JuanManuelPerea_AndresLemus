# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 16:02:42 2023

@author: POWER
"""

import numpy as np
import matplotlib.pyplot as plt

# Define la función f(z) y su derivada
def f(z):
    return z**3 - 1

def df(z):
    return 3 * z**2

# Función para encontrar la raíz de f(z) utilizando el método de Newton
def newton_method(z, max_iter=100, tol=1e-6):
    for i in range(max_iter):
        z = z - f(z) / df(z)
        if abs(f(z)) < tol:
            return z
    return None

# Rango de valores de x e y para el gráfico
x_range = np.linspace(-1, 1, 300)
y_range = np.linspace(-1, 1, 300)

# Matriz para almacenar el valor de la raíz encontrada en cada punto del plano complejo
root_matrix = np.zeros((len(x_range), len(y_range)), dtype=complex)

# Itera a través de todos los puntos del plano complejo
for i, x in enumerate(x_range):
    for j, y in enumerate(y_range):
        initial_guess = complex(x, y)
        root = newton_method(initial_guess)
        if root is not None:
            root_matrix[i, j] = root

# Grafica el resultado
#print(root_matrix)
plt.imshow(np.angle(root_matrix), extent=(x_range.min(), x_range.max(), y_range.min(), y_range.max()), cmap='coolwarm')
plt.colorbar()
plt.title("Fractales de Newton para $f(z) = z^3 - 1$")
plt.show()

