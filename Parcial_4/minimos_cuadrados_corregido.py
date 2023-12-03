import numpy as np
import matplotlib.pyplot as plt

# Definir las ecuaciones de las tres líneas
def recta1(x):
    return 2*x - 2

def recta2(x):
    return (1 - x) / 2

def recta3(x):
    return 4 - x
x_values = np.arange(-5, 5.01, 0.01)
y_values = np.arange(-5, 5.01, 0.01)
distancia_min = float('inf')
punto_min = None
for x in x_values:
    for y in y_values:
        # Calcular la distancia desde el punto (x, y) a cada línea
        distancia1 = np.abs(y - recta1(x)) / np.sqrt(5)
        distancia2 = np.abs(y - recta2(x)) / np.sqrt(5)
        distancia3 = np.abs(y - recta3(x)) / np.sqrt(2)

        # Calcular la distancia total sumando las distancias de las tres líneas
        distancia_tot= distancia1 + distancia2 + distancia3

        # Actualizar el resultado si encontramos una distancia menor
        if distancia_tot < distancia_min:
            distancia_min = distancia_tot
            punto_min = (x, y)

# Imprimir el resultado
print("Coordenadas del punto que minimiza la distancia:", punto_min)
print("Distancia mínima:", distancia_min)

# Graficar las líneas y el punto encontrado
plt.plot(x_values, recta1(x_values), label='2x - y = 2')
plt.plot(x_values, recta2(x_values), label='x + 2y = 1')
plt.plot(x_values, recta3(x_values), label='x + y = 4')
plt.scatter(*punto_min, color='red', label='Punto mínimo')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()



A = np.array([[3, 1, -1], [1, 2, 0], [0, 1, 2], [1, 1, -1]])
b = np.array([-3, -3, 8, 9])

# Calcular la solución usando mínimos cuadrados
x = np.linalg.lstsq(A, b, rcond=None)[0]


proyeccion = np.dot(A, x)

print("Sol minimos cuadrados (x):", x)
print("Proyección :", proyeccion)



u1 = np.array([3, 1, 0, 1])
u2 = np.array([1, 2, 1, 1])
u3 = np.array([-1, 0, 2, -1])

v1 = u1


proy_v1_u2 = np.dot(u2, v1) / np.dot(v1, v1) * v1
v2 = u2 - proy_v1_u2

proy_v1_u3 = np.dot(u3, v1) / np.dot(v1, v1) * v1
proy_v2_u3 = np.dot(u3, v2) / np.dot(v2, v2) * v2
v3 = u3 - proy_v1_u3 - proy_v2_u3

v1 = v1 / np.linalg.norm(v1)
v2 = v2 / np.linalg.norm(v2)
v3 = v3 / np.linalg.norm(v3)


c1 = np.dot(u1, v1)
c2 = np.dot(u2, v2)
c3 = np.dot(u3, v3)


proj_W_b = c1 * v1 + c2 * v2 + c3 * v3

print("Base ortonormal:")
print("v1:", v1)
print("v2:", v2)
print("v3:", v3)

print("\nCoeficientes c1, c2, c3:")
print("c1:", c1)
print("c2:", c2)
print("c3:", c3)

print("\nProyección ortogonal sobre la base (proj_W_b):", proj_W_b)

