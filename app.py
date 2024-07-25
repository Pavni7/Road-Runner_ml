from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import joblib

app = Flask(__name__)

# Load the trained model from the file
loaded_model=joblib.load("roadrunner_speed_model.pkl")

   
@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == "POST":
        distance = float(request.form['distance'])
        terrain = float(request.form['terrain'])
        weather_conditions = float(request.form['weather_conditions'])
        new_data = pd.DataFrame({'Distance': [distance]})
        predicted_speed = loaded_model.predict(new_data)
        if distance<=0:
            return render_template('result.html', predicted_speed="we are not allowing negative and 0 values")
        else:
             return render_template('result.html', predicted_speed=predicted_speed[0])
    return render_template('main.html')


if __name__ == '__main__':
    app.run(debug=True)