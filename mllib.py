from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential
import xgboost as xgb
import pandas as pd
import numpy as np

models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    xgb.XGBRegressor(objective='reg:linear',
                     n_estimators=10, random_state=123),
    xgb.XGBClassifier(n_estimators=10, random_state=123),
    Sequential()
]

model_dict = {
    model.__class__.__name__: model for model in models if model.__class__.__name__ != "Sequential"}
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))


def get_model(model_name):
    return model_dict[model_name]


class ML_lib:
    def __init__(self, dataframe, target):
        self.target = dataframe[target]
        self.affect = dataframe.drop(target, axis=1)

    def pre_processing(self):
        dummy_cols = []
        le = preprocessing.LabelEncoder()
        for col in self.affect.columns:
            col_uniques = self.affect[col].unique()
            if len(col_uniques) > 10:
                self.affect[col] = le.fit_transform(self.affect[col])
            else:
                dummy_cols.append(col)
        if dummy_cols:
            self.affect = pd.get_dummies(self.affect, columns=dummy_cols)
        return self.affect

    def train_test_split(self, is_random=True):
        if not is_random:
            data_training = pd.DataFrame(
                self.target[:int(len(self.target) * 0.70)])
            data_testing = pd.DataFrame(
                self.target[int(len(self.target) * 0.70):])
            data_training_array = scaler.fit_transform(data_training)
            x_train, y_train, x_test, y_test = [], [], [], []
            for i in range(100, data_training_array.shape[0]):
                x_train.append(data_training_array[i - 100: i])
                y_train.append(data_training_array[i, 0])
            data_testing = pd.concat(
                [data_training.tail(100), data_testing], ignore_index=True)
            data_testing_array = scaler.fit_transform(data_testing)
            for i in range(100, data_testing_array.shape[0]):
                x_test.append(data_testing_array[i - 100:i])
                y_test.append(data_testing_array[i, 0])
            return np.array(x_train), np.array(x_test), np.array(y_train), np.array(y_test)
        return train_test_split(self.affect, self.target, test_size=0.20, random_state=2)
