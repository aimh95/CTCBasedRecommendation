from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class LSTM(tf.keras.Model):
    def __init__(self):
        super(LSTM, self).__init__()
        self.LSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, activation="relu", return_sequences=False))
        self.manytomany_rp = tf.keras.layers.RepeatVector(1)
        self.output_layer = tf.keras.layers.Dense(74, activation="softmax")


    def call(self, x):
        x = self.LSTM(x)
        x = self.output_layer(x)

        return x