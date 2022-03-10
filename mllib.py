import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import sklearn as sk
import xgboost as xg
import pandas as pd

models = [
    LinearRegression(),
    DecisionTreeRegressor(),
    RandomForestRegressor(),
    xg.XGBRegressor(objective ='reg:linear', n_estimators = 10, seed = 123),
    xg.XGBClassifier(n_estimators=10, seed=123)
]
model_dict = {x.__class__.__name__: x for x in models}

def get_model(model_name):
    print(model_dict)
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

    def train_test_split(self):
        return tts(self.affect, self.target, test_size=0.20, random_state=2)
