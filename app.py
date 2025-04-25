
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('about.html')

@app.route('/predictor', methods=['GET', 'POST'])
def predictor():
    prediction = None
    if request.method == 'POST':
        try:
            inputs = [
                float(request.form['pH']),
                float(request.form['Temprature']),
                float(request.form['Taste']),
                float(request.form['Odor']),
                float(request.form['Fat']),
                float(request.form['Turbidity']),
                float(request.form['Colour'])
            ]
            df = pd.DataFrame([inputs], columns=['pH','Temprature','Taste','Odor','Fat ','Turbidity','Colour'])

            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df)

            model = GaussianNB()
            X = pd.read_csv('milknew.csv')
            y = X['Grade']
            X = X.drop(['Grade'], axis=1)
            scaler_model = StandardScaler().fit(X)
            X_scaled = scaler_model.transform(X)
            model.fit(X_scaled, y)

            df_scaled = scaler_model.transform(df)
            pred = model.predict(df_scaled)
            
            grade_descriptions = {
                'low': 'Low quality milk that may require additional processing or not suitable for direct consumption.',
                'medium': 'Medium quality milk that meets basic standards and is suitable for most dairy products.',
                'high': 'High quality milk that exceeds standards and is ideal for premium dairy products.'
            }
            
            grade = pred[0].lower()
            description = grade_descriptions.get(grade, 'Unknown grade quality')
            
            prediction = {
                'grade': grade,
                'description': description
            }
            
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify(prediction)
            
        except Exception as e:
            prediction = {'error': str(e)}
            if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
                return jsonify(prediction)
    
    return render_template('predictor.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)