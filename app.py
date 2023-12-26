from flask import Flask, jsonify, request, render_template
import pickle
import os
import secrets
from sklearn.preprocessing import StandardScaler
from flower_classifier.train_models import custompredict
from flask_wtf import FlaskForm
from wtforms import validators
from wtforms.fields import SelectField, DecimalField, SubmitField

# Getting model names
models = {}
for i in os.listdir('models'):
    models[i.replace('.pck', '')] = pickle.load(open(f'models/{i}' ,'rb'))

class Form(FlaskForm):
    model = SelectField('model', choices=models.keys())
    petal_width = DecimalField('petal_width', [validators.InputRequired()], render_kw = {'placeholder' : 'petal width'})
    petal_length = DecimalField('petal_length', [validators.InputRequired()], render_kw = {'placeholder' : 'petal length'})
    submit = SubmitField('predict')

app = Flask('app')
app.config['SECRET_KEY'] = secrets.token_hex(16)

def _predict(model, petal_width, petal_length):
    if model in models.keys():
        model = model+'model' if model.find('model')<0 else model
        model = model if model.find('.pck')<0 else model

        result = {
            'model': f'{model}',
            'petal_length': petal_length,
            'petal_width': petal_width,
            'prediction': models[model].custompredict([[petal_width, petal_length]])
        }
        return result
    else:
        result = {
            'error': f'model {model} does not exist. Existing models are {models.keys()}'
        }
        return result

@app.route('/', methods=['GET', 'POST'])
@app.route('/predict', methods=['GET', 'POST'])
def index():
    form = Form()
    result = ''
    if form.is_submitted():
        form_result = request.form
        result = _predict(form_result['model'],form_result['petal_width'], form_result['petal_length'])
        result = result['prediction'][0]
    return render_template('index.html', form=form, result=result)

@app.route('/api/predict/<model>', methods=['GET', 'POST'])
def predict(model):
    try:
        try:
            request.get_json()
            petal_width, petal_length = (request.get_json()['petal_width'], request.get_json()['petal_length'])
        except:
            petal_width, petal_length = (request.args.get('petal_width'), request.args.get('petal_length'))
            
        return jsonify(_predict(model, petal_width, petal_length))
    except Exception as e:
        result = {
            'error': 'arguments either are missing or do not exist. Args must be petal_width:float and petal_length:float'
        }
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
