import sklearn.metrics

from model.LSTM import LSTM
import tensorflow as tf
import data.lstm_dataloader as d_loader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import numpy as np


file_path = "../data/가번1_데이터.csv"

dataframe = pd.read_csv(file_path, encoding="utf8")
dataframe.info()
dataframe.head(5)

features = ["P_YYYYMMDD", "CHNL_ESPE_POT_CHNL_STAY_TM","BASE_WEEKDAY","BASE_TM", "CHNL_PNTN_ESPE_BRD_CHNL_NO"]
x_data, y_data = d_loader._dataloading_preprocessing(dataframe, features)

# x_data, y_data = d_loader._xy_split(X, y)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)
y_train = tf.expand_dims(y_train, axis=1)
y_test = tf.expand_dims(y_test, axis=1)

optimizer = tf.optimizers.Adam(learning_rate=1e-3)
model = LSTM()

model.compile(optimizer="adam", loss='sparse_categorical_crossentropy', metrics="accuracy")
model.build(input_shape=(4, 4, 77))
# x_train = np.reshape(x_train, (x_train.shape[0], x_train[0].shape[0], 1))

model.fit(x_train,y_train, epochs=10, batch_size=2, validation_data=(x_test, y_test))

