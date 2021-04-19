from flask import Flask, render_template, request
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import pickle

with open('model/model.pkl', 'rb') as file:
    model = pickle.load(file)

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        beds = request.form['beds']
        baths = request.form['baths']
        sqft = request.form['sqft']
        pred = model.predict(np.array([int(beds), int(baths), int(sqft)]))
        return render_template('index.html', pred=str(pred))

    return render_template('index.html')


if __name__ == '__main__':
    app.run()