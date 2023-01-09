import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__, template_folder='template')
model = pickle.load(open('model.pickle', 'rb'))

@app.route('/', methods=['GET'])
def home():
    return render_template('iris.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    For rendering results on HTML GUI
    """
    int_features = np.array([float(x) for x in request.form.values()])
    final_features = [np.array(int_features)]
    prediction = round(model.predict(final_features)[0])

    print(prediction)

    if prediction < 0.5:
        return render_template("iris.html", prediction_text='Iris Variety should be Setosa'.format(prediction))
    elif (prediction >= 0.5) and (prediction < 1.5):
        return render_template("iris.html", prediction_text='Iris Variety should be Versicolor'.format(prediction))
    else:
        return render_template("iris.html", prediction_text='Iris Variety should be Virginica'.format(prediction))

if __name__ == "__main__":
    app.run(port=5000, debug=True)
