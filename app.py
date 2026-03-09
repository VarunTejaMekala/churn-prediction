import numpy as np
import pandas as pd
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import shap
from flask import Flask, request, render_template
import base64
import io

app = Flask(__name__)

# Load model + preprocessor
preprocessor = pickle.load(open("final_model/preprocessor.pkl", "rb"))
model = pickle.load(open("final_model/model.pkl", "rb"))

shap.initjs()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    # Helper function for checkbox inputs
    def checkbox(name):
        return "Yes" if request.form.get(name) == "on" else "No"

    # Form inputs
    gender = request.form["gender"]

    SeniorCitizen = 1 if request.form.get("SeniorCitizen") == "on" else 0

    Partner = checkbox("Partner")
    Dependents = checkbox("Dependents")
    PhoneService = checkbox("PhoneService")
    MultipleLines = checkbox("MultipleLines")

    InternetService = request.form["InternetService"]

    OnlineSecurity = checkbox("OnlineSecurity")
    OnlineBackup = checkbox("OnlineBackup")
    DeviceProtection = checkbox("DeviceProtection")
    TechSupport = checkbox("TechSupport")

    StreamingTV = checkbox("StreamingTV")
    StreamingMovies = checkbox("StreamingMovies")

    Contract = request.form["Contract"]

    PaperlessBilling = checkbox("PaperlessBilling")

    PaymentMethod = request.form["PaymentMethod"]

    MonthlyCharges = float(request.form["MonthlyCharges"])
    Tenure = int(request.form["Tenure"])

    TotalCharges = MonthlyCharges * Tenure

    # Feature engineering (same as training)
    charges_per_tenure = MonthlyCharges / (Tenure + 1)
    total_spend = MonthlyCharges * Tenure

    # tenure group
    if Tenure <= 12:
        tenure_group = "0-1yr"
    elif Tenure <= 24:
        tenure_group = "1-2yr"
    elif Tenure <= 48:
        tenure_group = "2-4yr"
    elif Tenure <= 60:
        tenure_group = "4-5yr"
    else:
        tenure_group = "5+yr"

    # Raw dataframe (same columns used in training)
    input_df = pd.DataFrame([{
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": Tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges,

    # NEW FEATURES
    "charges_per_tenure": charges_per_tenure,
    "total_spend": total_spend,
    "tenure_group": tenure_group
}])

    # Apply preprocessing
    transformed_data = preprocessor.transform(input_df)

    # Predict probability
    prediction = model.predict_proba(transformed_data)
    churn_prob = prediction[0][1]


    # Gauge visualization
    def gauge(prob):

        fig, ax = plt.subplots()

        pos = (1 - prob) * 180

        ax.arrow(
            0, 0,
            0.5 * np.cos(np.radians(pos)),
            0.5 * np.sin(np.radians(pos)),
            width=0.03,
            head_width=0.08,
            head_length=0.1,
            fc='black'
        )

        ax.text(
            0, -0.2,
            f"Churn Probability {round(prob,2)}",
            horizontalalignment='center',
            fontsize=16,
            fontweight='bold'
        )

        ax.axis("off")

        img = io.BytesIO()

        plt.savefig(img, format="png")

        img.seek(0)

        return base64.b64encode(img.getvalue()).decode()

    gauge_url = gauge(churn_prob)

    return render_template(
        "index.html",
        prediction_text=f"Churn probability is {round(churn_prob,2)}",
        url_1=gauge_url
    )


if __name__ == "__main__":
    app.run(debug=True)