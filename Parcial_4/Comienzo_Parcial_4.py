# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 13:18:13 2023

@author: POWER
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import copy


sigm = lambda x: 1/(1+np.exp(-x))
class Layer:
    
    
    def __init__(self,NC,NN,ActFun,rate=4): # Jugar con la tasa de mutacion
        
        self.NC = NC
        self.NN = NN
        self.ActFunc = ActFun
        self.rate = rate
        
        self.W = np.random.uniform( -10.,10.,(self.NC,self.NN) )
        self.b = np.random.uniform( -10.,10.,(1,self.NN) )
        
    def Activation(self,x):
        z = np.dot(x,self.W) + self.b
        return self.ActFunc( z )[0]
    
    def Mutate(self):
    
        #self.W += np.random.normal( loc=0., scale=self.rate, size=(self.NC,self.NN))
        #self.b += np.random.normal( loc=0., scale=self.rate, size=(1,self.NN))
        
        self.W += np.random.uniform( -self.rate, self.rate, size=(self.NC,self.NN))
        self.b += np.random.uniform( -self.rate, self.rate, size=(1,self.NN))
def GetBrain():
    l0 = Layer(1,5,sigm)
    l1 = Layer(5,1,sigm)
    #l2 = Layer(2,1,sigm)
    Brain = [l0,l1]
    return Brain    
 

class Robot:
    
    def __init__(self, dt, Layers, Id=0):
        
        self.Id = Id
        self.dt = dt
        
        
        self.r = np.random.uniform([0.,0.])
        theta = 0.
        self.v = np.array([1.*np.cos(theta),1.*np.sin(theta)])

        
        # Capacidad o aptitud del individuo
        self.Fitness = np.inf
        self.Steps = 0

        # Brain
        self.Layers = Layers
        
    def GetR(self):
        return self.r
    
    def Evolution(self):
        self.r += self.v*self.dt # Euler integration (Metodos 2)
        
        
        
        # Cada generación regresamos el robot al origin
        # Y volvemos a estimar su fitness
    def Reset(self):
        self.Steps = 0
        self.r = np.array([0.,0.])
        self.Fitness = np.inf
        
    # Aca debes definir que es mejorar en tu proceso evolutivo
    def SetFitness(self):
        
        if -1<self.r[0]<1:
            self.Fitness = (1 / self.Steps)
        else:
            self.Fitness=np.inf
        if self.Fitness < 0:
            self.Fitness=np.inf
        
        return self.Fitness
       # Brain stuff
    def BrainActivation(self,x,threshold=0.7): 
        # El umbral (threshold) cerebral es a tu gusto!
        # cercano a 1 es exigente
        # cercano a 0 es sindrome de down
        
        # Forward pass - la infomación fluye por el modelo hacia adelante
        for i in range(len(self.Layers)):         
            if i == 0:
                output = self.Layers[i].Activation(x)
            else:
                output = self.Layers[i].Activation(output)
        
        self.Activation = np.round(output,4)
        
        # Cambiamos el vector velocidad
        if self.Activation[0] > threshold :
            
            self.v = -self.v
            
            
            self.Steps=self.Steps-0.9
            
            
    
        return self.Activation
    
    # Aca mutamos (cambiar de parametros) para poder "aprender"
    def Mutate(self):
        for i in range(len(self.Layers)):
            self.Layers[i].Mutate()
    
    # Devolvemos la red neuronal ya entrenada
    def GetBrain(self):
        return self.Layers
    
    
def GetRobots(N):
    
    Robots = []
    
    for i in range(N):
        
        Brain = GetBrain()
        r = Robot(dt,Brain,Id=i)
        Robots.append(r)
        
    return Robots
dt = 0.1
t = np.arange(0.,5.,dt)
Robots = GetRobots(10)
#print(Robots)
def GetPlot():
    
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(1,2,1)
    ax1 = fig.add_subplot(1,2,2)
    
    ax.set_xlim(-1.,1.)
    ax.set_ylim(-1.,1.)
 
    return ax,ax1

def TimeEvolution(Robots,e,Plot=True):
    
  
    for it in range(t.shape[0]):
        
        if Plot:
        
            clear_output(wait=True)
        
            ax,ax1 = GetPlot()
            ax1.set_ylim(0.,1.)
        
            ax.set_title('t = {:.3f}'.format(t[it]))
        
        Activation = np.zeros(len(Robots))
        
        for i,p in enumerate(Robots):
            p.Evolution()
         
            # Activacion cerebral
            Act = p.BrainActivation(p.GetR()[0])
            Activation[i] = Act
            # Region donde aumentamos los pasos para el fitness
            if -1<p.r[0]<1:
                p.Steps+=1
                
            else:
                p.Steps-=1
                
            #p.Steps+=1
                
            if Plot and i < 5: # Solo pintamos los primeros 5, por tiempo de computo
                ax.scatter(p.r[0],p.r[1],label='Id: {}, Steps: {:.0f}'.format(p.Id,p.Steps))
                ax.quiver(p.r[0],p.r[1],p.v[0],p.v[1])
                
        #Pintamos la activaciones de los primeros 5
        
        if Plot:
            ax1.plot(np.arange(0,len(Robots[:5]),1),Activation[:5],marker='o',color='b',label='Activation')
            ax1.axhline(y=0.7,color='r')
        
        if Plot:
        
            ax.legend(loc=0)  
            ax1.legend(loc=0)
            plt.show()
            time.sleep(0.001)

# Definimos la rutina de entrenamiento
def Genetic(Robots, epochs = 200, Plot = True, Plottime=False):
    
    
    FitVector = np.array([])
    
    
    x = np.linspace(-1,1,20)
    Act = np.zeros_like(x)
    
    for e in range(int(epochs)):
        
        # Reiniciamos y mutamos los pesos
        
        for p in Robots:
            p.Mutate()
            p.Reset()
            
        # Evolucionamos
        TimeEvolution(Robots,e,Plottime) # Apagar dibujar la evolución para entrenar
        
        # Actualizamos fitness de cada robot
        for p in Robots:
            p.SetFitness()
            #print(p.Fitness)
            #print(p.Steps)
        
        # Aca va toda la rutina de ordenar los bots del más apto al menos apto
        scores=[(p.Fitness,p) for p in Robots]
        scores.sort(key=lambda t: t[0], reverse=False)
        print(scores)
        
        
        # Guardamos el mejor fitness y le mejor robot
        

        best_fitness = scores[0][0]
        best_bot = scores[0][1]
        FitVector = np.append(FitVector, best_fitness)
        
        
        """
        Para esta parte se probo con N=30%
        Al igual que convertir la siguiente generacion a copias del mejor individuo
        
        """
        for i,r in enumerate(Robots):
            
            #if i <2:
            #    Robots[i]=copy.deepcopy(scores[i][1])
                
            Robots[i]=copy.deepcopy(best_bot)
            

        
        
        # best_fitness = scores[0][0]
        # best_bot = scores[0][1]
        # FitVector = np.append(FitVector, best_fitness)
        
        for i in range(len(x)):
            Act[i] = best_bot.BrainActivation(x[i])
        
            clear_output(wait=True)
            
            print('Epoch:', e)
                    
            # Last fitness
            print('Last Fitness:', FitVector[-1])
            
        
        if Plot:
            
            ax,ax1 = GetPlot()
            ax.plot(x,Act,color='k')
            ax.set_ylim(0.,1)
            ax.axhline(y=0.75,ls='--',color='r',label='Threshold')
            
            ax1.set_title('Fitness')
            ax1.plot(FitVector)
        
            ax.legend(loc=0)
            
            plt.show()
            
            time.sleep(0.01)
        
        
    
    return best_bot, FitVector

Robots = GetRobots(10)
Best, FitVector = Genetic(Robots)#,Plot=True,Plottime=True) # Apagar Plottime para el entrenamiento

"""
testing

"""
Bestl=[Best]
def Timetest(Robots,e,Plot=True):
    
  
    for it in range(t.shape[0]):
        
        if Plot:
        
            clear_output(wait=True)
        
            ax,ax1 = GetPlot()
            ax1.set_ylim(0.,1.)
        
            ax.set_title('t = {:.3f}'.format(t[it]))
        
        Activation = np.zeros(len(Robots))
        
        for i,p in enumerate(Robots):
            p.Evolution()
         
            # Activacion cerebral
            Act = p.BrainActivation(p.GetR()[0])
            Activation[i] = Act
            # Region donde aumentamos los pasos para el fitness
            if -1<p.r[0]<1:
                p.Steps+=1
                
            else:
                p.Steps+=0
                p.v = -p.v
            #p.Steps+=1
                
            if Plot and i < 5: # Solo pintamos los primeros 5, por tiempo de computo
                ax.scatter(p.r[0],p.r[1],label='Id: {}, Steps: {:.0f}'.format(p.Id,p.Steps))
                ax.quiver(p.r[0],p.r[1],p.v[0],p.v[1])
                
        #Pintamos la activaciones de los primeros 5
        
        if Plot:
            ax1.plot(np.arange(0,len(Robots[:5]),1),Activation[:5],marker='o',color='b',label='Activation')
            ax1.axhline(y=0.7,color='r')
        
        if Plot:
        
            ax.legend(loc=0)  
            ax1.legend(loc=0)
            plt.show()
            time.sleep(0.001)
            
Timetest(Bestl, 1)
Red=Best.GetBrain()

for i, layer in enumerate(Red):
    print(f"Layer {i + 1} - Weights:")
    print(layer.W)
    print(f"Layer {i + 1} - Biases:")
    print(layer.b)
    print()
    
"""
Pesos para Robots atrapados:
    
Bot 1

Layer 1 - Weights:
[[24.45430144 -3.40376846 12.85752974 32.50226244  3.89633125]]
Layer 1 - Biases:
[[ 30.22667576  -4.42991983 -13.09103441  28.71561198  -2.39270148]]

Layer 2 - Weights:
[[-28.32271339]
 [  1.3325299 ]
 [ 26.01589688]
 [ -9.96809212]
 [  8.67368046]]
Layer 2 - Biases:
[[33.51701247]]

Bot 2

Layer 1 - Weights:
[[-26.07197827  22.05514158 -14.75954112 -23.78392298 -22.51983821]]
Layer 1 - Biases:
[[ 14.69792756   8.26169945 -16.71072592 -41.41625565  -1.37092952]]

Layer 2 - Weights:
[[-27.290973  ]
 [  9.84748031]
 [ 51.17683145]
 [  3.62861649]
 [ 40.91531173]]
Layer 2 - Biases:
[[-5.98082697]]

Bot 3

Layer 1 - Weights:
[[-39.40950508 -34.98899462 -14.97411491  18.90964715  13.36004939]]
Layer 1 - Biases:
[[ 8.53089678 18.56926249 -9.95875943 12.03811985 -2.72962767]]

Layer 2 - Weights:
[[ -8.04576832]
 [ -2.77665607]
 [  8.86492155]
 [-23.81087712]
 [ 18.64476079]]
Layer 2 - Biases:
[[18.7690196]]

Bot 4

"""