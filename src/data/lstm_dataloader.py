import pandas
import pandas as pd
import datetime
from collections import deque
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def _dataloading_preprocessing(data_frame, features):
    data_frame["P_YYYYMMDD"] = pd.to_datetime(data_frame["P_YYYYMMDD"])
    data_frame = data_frame.sort_values(by="P_YYYYMMDD")
    data = pd.DataFrame(data_frame[features])
    data = data.drop(columns = ["P_YYYYMMDD"])

    data_scaled = data.drop(columns = ["CHNL_PNTN_ESPE_BRD_CHNL_NO"])
    data_target = data["CHNL_PNTN_ESPE_BRD_CHNL_NO"]

    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_scaled)


    data_target = pd.get_dummies(data_target, columns=["CHNL_PNTN_ESPE_BRD_CHNL_NO"])


    data = pd.concat([pd.DataFrame(data_scaled), data_target], axis=1)

    # data = data.values

    X_data, y_data = create_sequences(data, 4)


    return X_data, y_data

# 시퀀스 데이터 생성
def create_sequences(data, seq_length):
    X_sequences = []
    y_sequences = []
    # for i in range(len(data) - seq_length):
    #     seq = data[i:i+seq_length]
    #     label = data[i+seq_length:i+seq_length+1, 3:]
    #     sequences.append((seq, label))
    for i in range(len(data) - seq_length - 1):
        X_sequences.append(data.iloc[i:i + seq_length].values)
        # y_sequences.append(data.iloc[i+1:i + seq_length+1, 3:].values)
        y_sequences.append(data.iloc[i+seq_length, 3:].values)

    return np.array(X_sequences), np.array(y_sequences)


def _xy_split(X, y):8
    x_data = X[:, 0]
    y_data = y[:, 1]
    return x_data, y_data


