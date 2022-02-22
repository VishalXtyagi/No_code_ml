import os

from flask import Flask, render_template, request, abort, jsonify, url_for, send_from_directory
from nocode import Nocode, get_plot_image, available_models, save_model_to_file

app = Flask(__name__)
app.secret_key = os.urandom(12).hex()
os.mkdir('static/models')
os.mkdir('static/plots')
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


@app.route('/process', methods=['GET', 'POST'])
def process():
    if request.method == "POST":
        target = request.form.get('target')
        model_name = request.form.get('model_name')
        if target and model_name:
            nocode.reset_index()
            nocode.cleaning_data()
            model, test_df = nocode.predict_by_model(model_name, nocode.split_data(target), target)
            mae, mse, rmse = nocode.error_metric(test_df['Actual'], test_df['Prediction'])
            plot_path = get_plot_image(test_df)
            model_path = save_model_to_file(model)
            return jsonify({
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'plot_link': url_for('static', filename=plot_path),
                'model_link': url_for('static', filename=model_path)
            })
        return jsonify({'message': 'error'}), 500
    abort(404)


@app.route('/get-models')
def get_models():
    return jsonify(available_models())


if __name__ == '__main__':
    app.run()
