from flask import Flask, render_template, request, abort, jsonify, url_for
from nocode import Nocode, get_plot_image, save_model_to_file, plots_dir, available_models

app = Flask(__name__)
nocode = None
df = None


@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == "POST":
        global nocode, df
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
        modelname = request.form.get('model_name')
        print(target, modelname)
        if target and modelname:
            nocode.reset_index()
            nocode.cleaning_data()
            mae, mse, rmse = nocode.predict_by_model(target, modelname)
            # plot_path = get_plot_image('data')
            return jsonify({
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                # 'plot_link': url_for('static', filename=plot_path)
            })
        return jsonify({'message': 'error'}), 500
    abort(404)


@app.route('/get-models')
def get_models():
    return jsonify(available_models())

if __name__ == '__main__':
    app.run()
