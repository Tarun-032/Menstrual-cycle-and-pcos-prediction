from flask import Flask, render_template, request
import joblib  # Import joblib to load your saved model
import numpy as np

app = Flask(__name__)

# Load your trained Linear Regression model
model = joblib.load('linear_regression_model.pkl')
pcos_model = joblib.load('pcos_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/input')
def input():
    return render_template('input.html')

@app.route('/pcos')
def pcos():
    return render_template('pcos.html')

@app.route('/symptoms')
def symptoms():
    return render_template('symptoms.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the form
    age = float(request.form['age'])
    flow_intensity = float(request.form['flow-intensity'])
    period_duration = float(request.form['period-duration'])
    lifestyle = float(request.form['lifestyle'])
    diet = float(request.form['diet'])
    issue = float(request.form['issue'])

    # Perform data preprocessing, if needed

    # Use your Linear Regression model to make predictions
    prediction = model.predict(np.array([[flow_intensity, period_duration, lifestyle, diet, issue,age]]))

    # You can send the prediction result to a new template or page for display
    return render_template('result.html', prediction=prediction[0])

@app.route('/diagnosis', methods=['POST'])
def diagnosis():
    # Get the data from the form
    agee = float(request.form['agee'])
    bmi = float(request.form['bmi'])
    cycleLength = float(request.form['cycleLength'])
    marriageStatus = float(request.form['marriageStatus'])
    pregnant = float(request.form['pregnant'])
    numAbortions = float(request.form['numAbortions'])
    skinDarkening = float(request.form['skinDarkening'])
    hairLoss = float(request.form['hairLoss'])
    follicleNoL = float(request.form['follicleNoL'])
    follicleNoR = float(request.form['follicleNoR'])

    x = [agee,bmi,cycleLength,marriageStatus,pregnant,numAbortions,skinDarkening,hairLoss,follicleNoL,follicleNoR]

    # Use your model to make predictions
    diag = pcos_model.predict([x])
    res=""

    if(diag[0]):
        res="High chances for PCOS"

    else:
        res="Minimal chances for PCOS"

    # You can send the prediction result to a new template or page for display
    return render_template('presult.html', diag=res)

if __name__ == '__main__':
    app.run(debug=True)
