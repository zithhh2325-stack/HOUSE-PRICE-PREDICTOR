from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

with open("HPP.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [
        float(request.form['OverallQual']),
        float(request.form['GrLivArea']),
        float(request.form['GarageCars']),
        float(request.form['TotalBsmtSF']),
        float(request.form['YearBuilt']),
        float(request.form['FullBath']),
        float(request.form['BedroomAbvGr']),
        float(request.form['LotArea'])
    ]

    prediction = model.predict([features])

    return render_template(
        'index.html',
        prediction_text=f'Predicted House Price: ${prediction[0]:,.2f}'
    )

if __name__ == '__main__':
    app.run(debug=True)