# -*- coding: utf-8 -*-
"""
Directional Field 2D normalizado
de  y'(x) = 1/cos(x)

Curso:  MCA 4 2025-2 

Tema:   Análisis Cualitativo de la EDO  y' = f(x) 

Referencias:
  * How to Plot a Direction Field with Python, Odubanjo Sec 6
     link web: https://pubhtml5.com/enyy/ikvo/basic/
  * web https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.quiver.html 

Software:
    Pyton 3.12.4
    Spyder 6.0.1
    Anaconda 2.6.4
    
@author: Roberto Méndez Méndez    
Editado  Jan 30 2025 v2
"""
import numpy as np
import matplotlib.pyplot as plt

# Definicón de la Derivada
def f(x,y):
    return 1/np.cos(x); 

nx, ny = .2, .2
x = np.arange(-2.5, 2.5, nx) 
y = np.arange(-3, 3, ny)

# Contrucción de los puntos de evaluación
X, Y = np.meshgrid(x, y)

# Derivative
dy = f(X,Y)
dx =np.ones(X.shape)

# Normalizar
dyu = dy/np.sqrt(dx**2 + dy**2)
dxu = dx/np.sqrt(dx**2 + dy**2)

# Gráficas
plt.quiver(X,Y,dxu,dyu, color = "orange", headwidth = 2)
plt.xticks(x, rotation = 60)
plt.yticks(y, fontsize=7)
plt.title("Directional Field y' = 1/cos(x)", 
          color='blue', fontsize ='x-large')
plt.show()

