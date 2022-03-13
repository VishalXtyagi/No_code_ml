import itertools
import os.path
import pickle

import pandas as pd
import numpy as np
from sklearn import metrics, tree
from mllib import ML_lib, get_model, model_dict, scaler
from keras.layers import Dense, Dropout, LSTM
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt

def allowed_ext(ds_type=None):
    ds_dict = {'csv': ['csv'], 'excel': ['xls', 'xlsx']}
    all_ext = list(itertools.chain.from_iterable(ds_dict.values()))
    if ds_type is None:
        return all_ext
    if ds_type in ds_dict.keys():
        return ds_dict[ds_type]
    return []


static_dir = 'static'
plot_img_path = f'plots/{datetime.now().strftime("%d%m%Y%H%M%S")}.png'
model_file_path = f'models/{datetime.now().strftime("%d%m%Y%H%M%S")}.'
sns.set_theme(color_codes=True)


def available_models():
    return list(model_dict.keys())


def get_plot_image(df, model_name):
    df = df.apply(pd.to_numeric)
    filepath = os.path.join(static_dir, plot_img_path)
    if 'LSTM' in model_name:
        plot = plt.figure()
        plt.plot(df['Actual'], 'b', label="Original Price")
        plt.plot(df['Prediction'], 'r', label="Predicted Price")
    # elif ''
    else:
        plot = sns.regplot(x="Actual", y="Prediction", data=df)
        # sns.scatterplot(x="Actual", y="Prediction", data=df)
    fig = plot.get_figure()
    fig.savefig(filepath)
    fig.clf()
    return plot_img_path


def save_model_to_file(model, filetype=None):
    if filetype == 'h5':
        model_path = model_file_path + 'h5'
        filepath = os.path.join(static_dir, model_path)
        model.save(filepath)
    else:
        model_path = model_file_path + 'pkl'
        filepath = os.path.join(static_dir, model_path)
        with open(filepath, 'wb') as file:
            pickle.dump(model, file)
    return model_path


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
        if self.dataframe is not None:
            self.dataframe = self.dataframe.apply(lambda x: x.astype(str).str.lower())
        return self.dataframe

    def drop_cols(self, columns):
        for col in columns:
            if col in self.dataframe:
                self.dataframe.drop(col, axis=1, inplace=True)
        return self.dataframe

    def reset_index(self, column=None):
        if column:
            self.dataframe.set_index(column, inplace=True)
        else:
            self.dataframe = self.dataframe.reset_index()
        if 'index' in self.dataframe.columns:
            self.dataframe.drop('index', axis=1, inplace=True)
        return self.dataframe

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
        return self.dataframe

    def split_data(self, target, is_random):
        ml = ML_lib(self.dataframe, target)
        ml.pre_processing()
        return ml.train_test_split(is_random)

    def predict_by_model(self, model_name, split_tnt_data, target):
        x_train, x_test, y_train, y_test = split_tnt_data
        model = get_model(model_name)
        if model_name == "LSTM":
            model.add(LSTM(units=50, activation='relu', return_sequences=True, input_shape=(x_train.shape[1], 1)))
            model.add(Dropout(0.2))
            model.add(LSTM(units=60, activation='relu', return_sequences=True))
            model.add(Dropout(0.3))
            model.add(LSTM(units=80, activation='relu', return_sequences=True))
            model.add(Dropout(0.4))
            model.add(LSTM(units=120, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            model.fit(x_train, y_train, epochs=50)
            pred = model.predict(x_test)
            pred = scaler.inverse_transform(pred)
            y_test = scaler.inverse_transform([y_test])
        else:
            model.fit(x_train, y_train)
            pred = model.predict(x_test)
        pred = pd.DataFrame(pred)
        test_df = pd.DataFrame()
        try:
            test_df["Actual"] = y_test.reset_index()[target]
            test_df["Prediction"] = pred.reset_index()[0]
        except:
            test_df["Actual"] = y_test[0]
            test_df["Prediction"] = pred[0]
        return model, test_df

    def error_metric(self, y_test, pred):
        mae = metrics.mean_absolute_error(y_test, pred)
        mse = metrics.mean_squared_error(y_test, pred)
        rmse = np.sqrt(metrics.mean_squared_error(y_test, pred))
        return mae, mse, rmse
