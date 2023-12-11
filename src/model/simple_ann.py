from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class SimpleANN(tf.keras.Model):
    def __init__(self):
        super(SimpleANN, self).__init__()
        self.hidden1 = tf.keras.layers.Dense(512, activation="relu")
        self.dropout1 = tf.keras.layers.Dropout(0.2)
        self.hidden2 = tf.keras.layers.Dense(1024, activation="relu")
        self.dropout2 = tf.keras.layers.Dropout(0.2)
        self.hidden3 = tf.keras.layers.Dense(1024, activation="relu")
        self.dropout3 = tf.keras.layers.Dropout(0.2)
        self.hidden4 = tf.keras.layers.Dense(512, activation="relu")
        self.out_layer = tf.keras.layers.Dense(6, activation="softmax")


    def call(self, x):
        x = self.dropout1(self.hidden1(x))
        x = self.dropout2(self.hidden2(x))
        x = self.dropout3(self.hidden3(x))
        x = self.out_layer(x)
        return x
6