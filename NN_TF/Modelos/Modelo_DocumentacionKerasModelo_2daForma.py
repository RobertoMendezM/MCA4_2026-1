# -*- coding: utf-8 -*-
"""
Models

Código adaptado de la primera referencia, similar al 
dado en la segunda referencia

2da forma de creear un modelo con  la clase Model

Referencias:  
* 1) https://keras-io.translate.goog/api/models/model/?_
  x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc
* 2) Ahlawat (2023). Reinforcement Learning for Finance, Apress. 
  pag 25

---
Editor: Roberto Méndez
Created on Thu May 8 2025
"""

import tensorflow as tf
import keras
from keras import Model

class CustomSequentialModel2(Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(5, activation="softmax")

    def call(self, inputs):
        x = self.dense1(inputs)
        print("Layer 1 \n", x)
        return self.dense2(x)


myModel = CustomSequentialModel2()
myModel.summary()
input = tf.random.normal((1, 2 ), dtype=tf.float32)
print("Input = \n", input)
output = myModel(input)
print("Cálculo del modelo \n", output)


