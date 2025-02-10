# -*- coding: utf-8 -*-
"""
Curso:  MCA 4 2025-2
 
Gráficas de la solución implícita 

                   y**3 - 5*y + x**2  - 4*x = C

para distintos C,  de la EDO separable

                 y' = (4 - 2x)/(3y^2 - 5)
           

Tema:  Identidicar adecuadamente las curvas solución de la EDO

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
from sympy import symbols, Eq
from sympy import plot_implicit
import math

x, y = symbols('x y')

# Gráfica de todas separadas
p1 = plot_implicit(Eq(y**3 - 5*y + x**2  - 4*x, -6), (x, -2, 6),
              (y, -3, 3), line_color = 'darkgoldenrod', 
               title = "y^3 - 5y + x^2  - 4x = -6 ",
               fontsize = 'medium');
p2 = plot_implicit(Eq(y**3 - 5*y + x**2  - 4*x, 0), (x, -2, 6),
              (y, -3, 3), line_color = 'darkorange', 
               title = "y^3 - 5y + x^2  - 4x = 0 ",
               fontsize = 'medium');
p3 = plot_implicit(Eq(y**3 - 5*y + x**2  - 4*x, .5), (x, -2, 6),
              (y, -3, 3), line_color = 'olivedrab', 
               title = "y^3 - 5y + x^2  - 4x = 0.5 ",
               fontsize = 'medium');
p4 = plot_implicit(Eq(y**3 - 5*y + x**2  - 4*x, 6), (x, -2, 6),
              (y, -3, 3), line_color = 'crimson', 
               title = "Gráficas de la soución  y^3 - 5y + x^2  - 4x = 6 ",
               fontsize = 'medium');

# Gráfica de todas juntas
p5 = plot_implicit(Eq(y, math.sqrt(5/3)), (x, -2, 6),
              (y, -3, 3), line_color = 'black', show=False);
p6 = plot_implicit(Eq(y, -math.sqrt(5/3)), (x, -2, 6),
              (y, -3, 3), line_color = 'black', show=False);
p7 = plot_implicit(Eq(y**3 - 5*y + x**2  - 4*x, -6), (x, -2, 6),
              (y, -3, 3), line_color = 'darkgoldenrod', 
               title = "Gráficas de la solución  y^3 - 5y + x^2  - 4x = c ",
               fontsize = 'medium', show=False);
p7.append(p2[0])
p7.append(p3[0])
p7.append(p4[0])
p7.append(p5[0])
p7.append(p6[0])
p7.show()