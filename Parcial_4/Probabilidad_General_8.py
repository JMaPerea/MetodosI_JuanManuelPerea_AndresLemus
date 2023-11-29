# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 22:00:23 2023

@author: POWER
"""

import numpy as np



n_lanzamientos = 4
k_caras = 2
p_experimental = 3/8
N = 10**5


contador_eventos = 0


for _ in range(N):
    resultados = np.random.choice([-1, 1], size=n_lanzamientos)
    if np.sum(resultados == 1) == k_caras:
        contador_eventos += 1


probabilidad_empirica = contador_eventos / N
error=abs((p_experimental-probabilidad_empirica)/p_experimental)*100

print(f"Probabilidad teórica: {p_experimental}")
print(f"Probabilidad empírica: {probabilidad_empirica}")
print(f"Error: {error}%")
