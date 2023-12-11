from model.simple_ann import SimpleANN
import tensorflow as tf
import data.live_dataloader as d_loader
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import datetime


file_path = "../data/고객가입여정(UHD4이상_30개단말_3개월_ANSI_input.csv"
pd.set_option('display.max_columns', None)
dataframe = pd.read_csv(file_path, encoding="euc-kr")
print(dataframe.info())
print(dataframe.head(5))
print(dataframe.describe())
# dataframe["CHNL_NO"].describe()
features = ["BASE_WEEKDAY", "BASE_TM", "CHNL_ESPE_POT_CHNL_STAY_TM", "CHNL_PNTN_ESPE_BRD_CHNL_NO"]

dataframe = dataframe[features]
# dataframe["RCYC_STRT_DT"] = pd.to_datetime(dataframe["RCYC_STRT_DT"], format="%Y%m%d")

# dataframe["RCYC_STRT_TME"] = (dataframe["RCYC_STRT_TME"] // 10000)
# dataframe["CHNL_NO"] = dataframe["CHNL_NO"] // 100

pass

sns.countplot(data=dataframe, x="BASE_WEEKDAY")
sns.countplot(data=dataframe, x= "BASE_TM")
# sns.countplot(data=dataframe, x="CHNL_ESPE_POT_CHNL_STAY_TM")
sns.countplot(data=dataframe, x="CHNL_PNTN_ESPE_BRD_CHNL_NO")
# sns.countplot(data=dataframe, x="CHNL_NM")
# sns.countplot(data=dataframe, x="YY10_AGLV_ID")

sns.scatterplot(x = dataframe["CHNL_ESPE_POT_CHNL_STAY_TM"], y = dataframe["CHNL_PNTN_ESPE_BRD_CHNL_NO"])

sns.stripplot(data=dataframe, x = "CHNL_PNTN_ESPE_BRD_CHNL_NO", y = "CHNL_ESPE_POT_CHNL_STAY_TM")
sns.stripplot(data=dataframe, x = "PP_NM", y = "CHNL_NO")
sns.stripplot(data=dataframe, x = "RCYC_STRT_TME", y = "CHNL_NO")

dataframe = dataframe.dropna(axis=0)
["RCYC_STRT_TME", "RCYC_END_DT", "RCYC_END_TME", "RCYC_END_DOW_SORT_NO", "CHNL_NO"]
sns.heatmap(data=dataframe.corr(), annot=True)
