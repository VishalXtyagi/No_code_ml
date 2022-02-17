import itertools
import os.path
import pickle

import pandas as pd
import numpy as np
from sklearn import metrics
from mllib import ML_lib, get_model, model_dict
from datetime import datetime
import seaborn as sns


def allowed_ext(ds_type=None):
    ds_dict = {'csv': ['csv'], 'excel': ['xls', 'xlsx']}
    all_ext = list(itertools.chain.from_iterable(ds_dict.values()))
    if ds_type is None:
        return all_ext
    if ds_type in ds_dict.keys():
        return ds_dict[ds_type]
    return []


model_dir = 'static/models'
plots_dir = 'static/plots'
sns.set_theme(color_codes=True)


def available_models():
    return list(model_dict.keys())

def get_plot_image(data):
    filepath = os.path.join(plots_dir, f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}.png')
    plot = sns.regplot(x="Y-Test", y="Prediction", data=data)
    fig = plot.get_figure()
    fig.savefig(filepath)
    return filepath


def save_model_to_file(model, filetype=None):
    if filetype == 'h5':
        filepath = os.path.join(model_dir, f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}.h5')
        model.save(filepath)
    else:
        filepath = os.path.join(model_dir, f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S")}.pkl')
        with open(filepath, 'wb') as file:
            pickle.dump(model, file)
    return filepath


class Nocode:

    def __init__(self, filename, filepath):
        self.filename = filename
        self.filepath = filepath
        self.dataframe = None

    def read_file(self, header=False):
        if header:
            if any(ext in self.filename for ext in allowed_ext('csv')):
                self.dataframe = pd.read_csv(self.filepath)
            elif any(ext in self.filename for ext in allowed_ext('excel')):
                self.dataframe = pd.read_excel(self.filepath)
        else:
            if any(ext in self.filename for ext in allowed_ext('csv')):
                self.dataframe = pd.read_csv(self.filepath, header=None)
            elif any(ext in self.filename for ext in allowed_ext('excel')):
                self.dataframe = pd.read_excel(self.filepath, header=None)
        return self.dataframe

    def reset_index(self, column=None):
        if column:
            self.dataframe.set_index(column)
        else:
            self.dataframe = self.dataframe.reset_index()
        if 'index' in self.dataframe.columns:
            self.dataframe = self.dataframe.drop('index', axis=1)

    def cleaning_data(self):
        self.dataframe = self.dataframe.replace('?', np.NaN)
        for col in self.dataframe.columns:
            null_sum = self.dataframe[col].isnull().sum()
            null_percent = (null_sum / len(col)) * 100
            if null_percent > 15:
                self.dataframe.drop(col, axis=1, inplace=True)
            elif null_percent > 0:
                mode = self.dataframe[col].mode()
                self.dataframe[col] = self.dataframe[col].replace(np.NaN, mode)

    def predict_by_model(self, target, model_name):
        ml = ML_lib(self.dataframe, target)
        x_train, x_test, y_train, y_test = ml.train_test_split()
        model = get_model(model_name)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        mae = metrics.mean_absolute_error(y_test, pred)
        mse = metrics.mean_squared_error(y_test, pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
        return mae, mse, rmse
