import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0,10,10)
y_1 = 2*x - 2 
y_2 = (1 - x)/2
y_3 = 4 - x
a_t = np.array([[2,1,1],[-1,2,1]])
A = a_t.T
b = np.array([2,1,4])
a_t = np.dot(A.T,A)
b_t = A.T @ b
x_solucion = np.linalg.solve(a_t,b_t)
plt.title("Gráfica de las tres rectas")
plt.ylabel("y")
plt.xlabel("x")
plt.plot(x,y_1)
plt.plot(x,y_2)
plt.plot(x,y_3)
plt.scatter(1.43,0.43, label="Punto común")
plt.legend()
plt.show()
print("El sistema no tiene solución, no se intersecan")