from flask import Flask, request, render_template
import pickle, joblib
import pandas as pd

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
ct = joblib.load('feature_values')

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/pred')
def predict():
    return render_template("index.html")

@app.route('/out', methods=["POST"])
def output():
    feature_cols = ['Age', 'Gender', 'self_employed', 'family_history',
                    'work_interfere', 'no_employees', 'remote_work', 'tech_company',
                    'benefits', 'care_options', 'wellness_program', 'seek_help',
                    'anonymity', 'leave', 'mental_health_consequence',
                    'phys_health_consequence', 'coworkers', 'supervisor',
                    'mental_health_interview', 'phys_health_interview',
                    'mental_vs_physical', 'obs_consequence']

    data = []
    for col in feature_cols:
        data.append(request.form.get(col, ''))

    pred = model.predict(ct.transform(pd.DataFrame([data], columns=feature_cols)))[0]
    if pred:
        return render_template("output.html", y="This person requires mental health treatment")
    else:
        return render_template("output.html", y="This person doesn't require mental health treatment")

if __name__ == '__main__':
    app.run(debug=True)
