import pandas as pd
import datetime
from collections import deque


def _dataloading_preprocessing(data_frame, features):
    data = data_frame[features]
    data = data.copy()
    for feature in features:
        if feature == "ENTR_NO":
            data = pd.get_dummies(data, columns=["ENTR_NO"])

        if feature == "BASE_WEEKDAY":
            data = pd.get_dummies(data, columns=["BASE_WEEKDAY"])


            # data_frame = data_frame.drop(columns=feature)
    return data

def _xy_split(dataframe):
    x_data = dataframe.iloc[:, 1:].values
    y_data = (dataframe.iloc[:, 0].values/100).astype(int)
    # y_data = pd.get_dummies(y_data)

    return x_data, y_data


