# -*- coding: utf-8 -*-
"""
Models with training

Código adaptado de la primera referencia, similar al 
dado en la segunda referencia.

2da forma de creear un modelo con  la clase Model y 
preparalo para training

La referencia 3) explica para que sirve el dropout

Referencias:  
* 1) https://keras-io.translate.goog/api/models/model/?_
  x_tr_sl=en&_x_tr_tl=es&_x_tr_hl=es&_x_tr_pto=tc

* 2) Ahlawat (2023). Reinforcement Learning for Finance, Apress. 
  pag 25

* 3) https://medium.com/@csediveyanand/
      what-is-dropout-in-keras-deep-learning-c2d0913439f3
      
* 4) https://keras.io/api/layers/regularization_layers/dropout/  

* 5) https://keras.io/api/models/model_training_apis/    
---
Editor: Roberto Méndez
Created on Thu May 8 2025
"""

import tensorflow as tf
import keras
from keras import Model

class MyModel3(Model):
    def __init__(self):
        super().__init__()
        self.dense1 = keras.layers.Dense(32, activation="relu")
        self.dense2 = keras.layers.Dense(5, activation="softmax")
        self.dropout = keras.layers.Dropout(0.5)

    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        print("Layer 1 \n", x)
        x = self.dropout(x, training=training)
        print("dropout 1 \n", x)
        return self.dense2(x)

model = MyModel3()
model.summary()
input = tf.random.normal((2, 10000 ), dtype=tf.float32)
print("Input = \n", input)
output = model(input)
print("Cálculo del modelo \n", output)


# Tomado de la documentación de keras training
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss=keras.losses.BinaryCrossentropy(),
    metrics=[
        keras.metrics.BinaryAccuracy(),
        keras.metrics.FalseNegatives(),
    ],
)

