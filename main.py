import os
import shutil
import matplotlib

from flask import Flask, render_template, request, abort, jsonify, url_for
from nocode import Nocode, get_plot_image, available_models, save_model_to_file

matplotlib.use('Agg')

app = Flask(__name__)
app.secret_key = os.urandom(12).hex()
is_random = True
file_type = None

for d in ('static/models', 'static/plots'):
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

nocode = None


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        global nocode
        fileid = request.form.get('id')
        filename = request.form.get('name')
        fileurl = request.form.get('url')
        header = bool(request.form.get('header'))
        nocode = Nocode(filename, fileurl)
        df = nocode.read_file(header)
        return render_template("process.html", df=df, fileid=fileid)
    return render_template("index.html")


@app.route('/process', methods=['POST'])
def process():
    global is_random, file_type
    index = request.json.get('index')
    drop_columns = request.json.get('drop_columns')
    target = request.json.get('target')
    model_name = request.json.get('model_name')

    if drop_columns and len(drop_columns) > 0:
        nocode.drop_cols(drop_columns)

    if target and model_name:
        try:
            nocode.reset_index(index)
            nocode.cleaning_data()
            if model_name == "LSTM":
                is_random = False
                file_type = 'h5'
            split_data = nocode.split_data(target, is_random)
            model, test_df = nocode.predict_by_model(
                model_name, split_data, target)
            mae, mse, rmse = nocode.error_metric(
                test_df['Actual'], test_df['Prediction'])
            plot_path = get_plot_image(test_df, model_name)
            model_path = save_model_to_file(model, file_type)
            return jsonify({
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'plot_link': url_for('static', filename=plot_path),
                'model_link': url_for('static', filename=model_path)
            })
        except Exception as e:
            return jsonify({'message': str(e)}), 500
    return jsonify({'message': 'Please ensure target variable and model name are both selected'}), 500


@app.route('/get-models')
def get_models():
    return jsonify(available_models())


if __name__ == '__main__':
    app.run(debug=True)
