from flask import Flask, request
import numpy as np
import pickle
import pandas as pd
import flasgger
from flasgger import Swagger

app = Flask(__name__)
Swagger(app)

pickle_in = open("logreg.pkl", "rb")
model = pickle.load(pickle_in)

@app.route('/')
def home():
    return '''
    <form action="/predict" method="get">
        Age: <input type="text" name="age"><br>
        New User (1 for yes, 0 for no): <input type="text" name="new_user"><br>
        Total Pages Visited: <input type="text" name="total_pages_visited"><br>
        <input type="submit" value="Predict">
    </form>
    '''

@app.route('/predict', methods=["GET"])
def predict_class():
    age = request.args.get("age")
    new_user = request.args.get("new_user")
    total_pages_visited = request.args.get("total_pages_visited")

    if age is None or new_user is None or total_pages_visited is None:
        return "Missing parameters", 400

    try:
        age = int(age)
        new_user = int(new_user)
        total_pages_visited = int(total_pages_visited)
    except ValueError:
        return "Invalid parameter type", 400

    prediction = model.predict([[age, new_user, total_pages_visited]])
    return f"Model prediction is: {prediction[0]}, which means the customer will {'buy' if prediction[0] == 1 else 'not buy'} the product."

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
