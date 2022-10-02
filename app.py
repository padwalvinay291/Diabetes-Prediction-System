import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
modelxg = pickle.load(open('rfmodel.pkl', 'rb'))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=['POST'])
def predict():
    if request.method=="POST":
        Gender = request.form['Gender']
        Age = request.form['Age']
        BMI = request.form['BMI']
        Cholesterol = request.form['Cholesterol']
        HDLChol = request.form['HDLChol']
        CholHDLratio = request.form['CholHDLratio']
        Glucose = request.form['Glucose']

        row_df=pd.DataFrame([pd.Series([Gender, Age, BMI, Cholesterol, HDLChol, CholHDLratio, Glucose])])
        print(row_df)
        prediction = modelxg.predict_proba(row_df)

        output = '{0:.{1}f}'.format(prediction[0][1], 2)
        output = str(float(output) * 100) + '%'
        if output>str(50):
            return render_template('next.html',result=f'You have chance of having diabetes.\nProbability of having Diabetes is {output}',isIndexHigh=True)
        elif(output>str(30)and output<str(50)):
            return render_template('next.html',result=f'You are safe.\n Probability of having diabetes is {output}',isIndexMod=True)
        else:
            return render_template('next.html', result='You are safe.')


if __name__ == "__main__":
    app.run(debug=True)