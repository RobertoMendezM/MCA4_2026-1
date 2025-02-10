# -*- coding: utf-8 -*-
"""
Curso:  MCA 4 2025-2
 
Campo Direccional de la EDO separable

                       y' = (4 - 2x)/(3y^2 - 5)
               
           

Tema:  Interpretaradecuadamente el campo direccional  e Identidicar 
       correctamente las curvas solución de la EDO

Referencias:
    * Edwards & Penney (2015). Differential Equations and Boundary 
      Value Problems, 5th edition, Pearson. pag. 32 
    
Software:
    Pyton 3.12.4
    Spyder 6.0.3
    Anaconda 2.6.4
    
Editor:: Roberto Méndez Méndez    
Editado  Feb 10 2025 v1
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Definicónde la Derivadak
def f(x,y):
    return (4 - 2*x)/(3*y**2 - 5)

#  nx, ny = .3, .3
nx, ny = .3, .3
x = np.arange(-2, 5, nx)
y = np.arange(-3, 3, ny)

# MESHGRID
X, Y = np.meshgrid(x, y)

# Derivative
dy = f(X,Y)
dx =np.ones(X.shape)

# Normalización
dyu = dy/np.sqrt(dx**2 + dy**2)
dxu = dx/np.sqrt(dx**2 + dy**2)

# Solución con odeint
steps = 50
t1 = np.linspace(0,4, steps)
sol1 = odeint(f, [1.3], t1, tfirst=True)
y0 = 1.2
t2 = np.linspace(0,4,steps)
sol2 = odeint(f, y0, t2, tfirst=True)

t3 = np.linspace(-0.8,4.6,steps)
sol3 = odeint(f, -2.6, t3, tfirst=True)

# Gráfica Directional Field
plt.quiver(X,Y,dxu,dyu, color = "orange",  headwidth = 2)

# Grafica Curvas Integrales
plt.plot(t1, sol1, color='red')
plt.plot(t2, sol2, color='red')
plt.plot(t3, sol3, color='red')

plt.xticks(x, rotation = 60, fontsize=8)
plt.yticks(y, fontsize=8 )
plt.title("Campo direccional y' = (4 - 2x)/(3y^2 - 5)", color='blue',
          fontsize ='large')
plt.show()