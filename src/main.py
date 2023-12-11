import sklearn.metrics

from model.simple_ann import SimpleANN
import tensorflow as tf
import data.live_dataloader as d_loader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2



file_path = "../data/고객가입여정(UHD4이상_30개단말_3개월_ANSI_input.csv"

dataframe = pd.read_csv(file_path, encoding="euc-kr")
dataframe.info()
dataframe.head(5)

features = ["CHNL_PNTN_ESPE_BRD_CHNL_NO", "ENTR_NO","BASE_WEEKDAY","BASE_TM"]
dataframe = d_loader._dataloading_preprocessing(dataframe, features)

x_data, y_data = d_loader._xy_split(dataframe)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)


optimizer = tf.optimizers.Adam(learning_rate=1e-2)
#
# model = SimpleANN()
# model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics="accuracy")
# hist = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))
#
# model.build(input_shape=x_train.shape)
# model.save_weights("./ann_model.tflite")
#
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# tflite_model = converter.convert()
#
# with open('model.tflite', 'wb') as f:
#   f.write(tflite_model)
#
#
# fig, loss_ax = plt.subplots()
# acc_ax = loss_ax.twinx()
#
# loss_ax.plot(hist.history['loss'], label='train loss')
# loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
# loss_ax.set_xlabel('epoch')
# loss_ax.set_ylabel('loss')
# loss_ax.legend(loc='upper left')
# plt.ylim([0, 5])
#
# acc_ax.plot(hist.history['acc'], 'b', label='train acc')
# acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
# acc_ax.set_ylabel('accuracy')
# acc_ax.legend(loc='upper left')
#
# plt.show()



from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier

kfold = StratifiedKFold(n_splits=5, shuffle = True, random_state=42)

models = []
predicts = []
i = 0

# loss =
# loss = tf.keras.losses.CategoricalCrossentropy()
for train_idx, valid_idx in kfold.split(x_data, y_data):
    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)

    x_train, x_test = x_data[train_idx], x_data[valid_idx]
    y_train, y_test = y_data[train_idx], y_data[valid_idx]

    model = RandomForestClassifier(random_state=42,n_estimators=512)
    model.fit(x_train, y_train)
    models.append(model)
    predicts.append(model.predict(x_test))
    print(models[i], sklearn.metrics.accuracy_score(model.predict(x_test), y_test))
    i += 1



pred = pd.DataFrame({'pred0':predicts[0], 'pred1':predicts[1], 'pred2':predicts[2], 'pred3':predicts[3], 'pred4':predicts[4]})
pred.mode(axis=1)

print(sklearn.metrics.accuracy_score(pred , y_test))