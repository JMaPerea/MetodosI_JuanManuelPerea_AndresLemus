# -*- coding: utf-8 -*-#
"""
Created on Mon Nov  6 14:15:54 2023

@author: POWER
"""

import sympy as sp


gamma0 = sp.Matrix([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, -1]])

gamma1 = sp.Matrix([[0, 0, 0, 1],
                   [0, 0, 1, 0],
                   [0, -1, 0, 0],
                   [-1, 0, 0, 0]])

gamma2 = sp.Matrix([[0, 0, 0, -sp.I],
                   [0, 0, sp.I, 0],
                   [0, sp.I, 0, 0],
                   [-sp.I, 0, 0, 0]])

gamma3 = sp.Matrix([[0, 0, 1, 0],
                   [0, 0, 0, -1],
                   [-1, 0, 0, 0],
                   [0, 1, 0, 0]])

gamma=[gamma0,gamma1,gamma2,gamma3]


eta = sp.Matrix([[1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, -1, 0],
                [0, 0, 0, -1]])



Valor = sp.Matrix.zeros(4, 4)

I = sp.eye(4)

for u in range(4):
    for v in range(4):
        #print(gamma[nu])
        anticomm = (gamma[u] * gamma[v]) + (gamma[v] * gamma[u])


        Valor += anticomm * eta
        #print(anticomm)



        expected = 2 * eta * I
        #print('real',anticomm)
        #print('expect',expected)
        print(f"Para: µ={u+1}   v={v+1}")
        if Valor == expected:
            print(f"La relación de anticonmutación se verifica y es igual a 2η^µνI=2^{u+1}{v+1}*I.")
        else:
            print("La relación de anticonmutación no se verifica.")
