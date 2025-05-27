# -*- coding: utf-8 -*-
"""
Models
    2da forma de creear un modelo con  la clase Model

Información:
Código adaptado de la primera y segunda referencia (similares 
entre ellos) con los parametros de layer de la referencia 3, e 
inicializadores de 4). 

Referencias:
* 1) Ahlawat (2023). Reinforcement Learning for Finance, Apress. 
  pag 25.
* 2) https://keras-io.translate.goog/api/models/model/?_
  x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc  
* 3) https://keras.io/api/layers/core_layers/dense/  
* 4) https://keras.io/api/layers/initializers/

---
Editor: Roberto Méndez
Created on Wed May 7 2025
Editado: 15 May 25
"""

import tensorflow as tf
import keras
from keras import Model

class CustomSequentialModel(Model):
  def __init__(self, name=None, **kwargs):
    super().__init__(name=name, **kwargs)
    self.layer1 = keras.layers.Dense(4,
                     #activation="tanh",
                     activation="relu",
                     #use_bias = False,
                     use_bias = True,
                     kernel_initializer="Ones",
                     #kernel_initializer="Orthogonal",
                     bias_initializer="GlorotNormal")
    self.layer2 = keras.layers.Softmax()

  def call(self, inputs, training=None, mask=None):
    x = self.layer1(inputs)
    tf.print("Layer 1 \n", x)
    return self.layer2(x)

myModel = CustomSequentialModel()
#myModel.summary()
# input = tf.random.normal((1, 3 ), dtype=tf.float32)
input = tf.ones((1, 1 ), dtype=tf.float32)
tf.print("Input = \n", input)
output = myModel(input)
tf.print("Cálculo del modelo \n", output)