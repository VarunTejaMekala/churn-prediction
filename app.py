"""
app.py — Flask web application for Churn Prediction + Survival Analysis.

Endpoints:
    GET  /          → Main prediction form
    POST /predict   → Run churn prediction + LTV calculation
    GET  /health    → Health check
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# ── Load models ───────────────────────────────────────────────────────────────
MODEL_DIR = "final_model"

def load_artifacts():
    artifacts = {}
    try:
        with open(os.path.join(MODEL_DIR, "preprocessor.pkl"), "rb") as f:
            artifacts["preprocessor"] = pickle.load(f)
        print("✓ Preprocessor loaded")
    except Exception as e:
        print(f"✗ Preprocessor not found: {e}")

    try:
        with open(os.path.join(MODEL_DIR, "model.pkl"), "rb") as f:
            artifacts["model"] = pickle.load(f)
        print("✓ Churn model loaded")
    except Exception as e:
        print(f"✗ Churn model not found: {e}")

    try:
        with open(os.path.join(MODEL_DIR, "survival_model.pkl"), "rb") as f:
            artifacts["survival"] = pickle.load(f)
        print("✓ Survival model loaded")
    except Exception as e:
        print(f"✗ Survival model not found: {e}")

    return artifacts

ARTIFACTS = load_artifacts()

# ── Feature engineering (mirrors data_transformation.py) ─────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["total_spend"] = df["MonthlyCharges"] * df["tenure"]

    service_cols = ["OnlineSecurity","OnlineBackup","DeviceProtection",
                    "TechSupport","StreamingTV","StreamingMovies"]
    df["service_count"] = df[service_cols].apply(
        lambda row: sum(1 for v in row if str(v).lower() not in ["no","no internet service"]),
        axis=1
    )
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0,12,24,48,60,100],
        labels=["0-1yr","1-2yr","2-4yr","4-5yr","5+yr"]
    ).astype(str)
    return df

def prepare_for_survival(df: pd.DataFrame) -> pd.DataFrame:
    """Encode features the same way as the survival analysis notebook."""
    d = df.copy()
    d["TotalCharges"] = pd.to_numeric(d["TotalCharges"], errors="coerce").fillna(0)

    for col in ["Partner","Dependents","PaperlessBilling","PhoneService"]:
        d[col] = (d[col] == "Yes").astype(int)

    d["gender"] = (d["gender"] == "Female").astype(int)
    d["MultipleLines"] = d["MultipleLines"].map({"No phone service":0,"No":0,"Yes":1})

    for col in ["OnlineSecurity","OnlineBackup","DeviceProtection",
                "TechSupport","StreamingTV","StreamingMovies"]:
        d[col] = d[col].map({"No internet service":0,"No":0,"Yes":1})

    d = pd.get_dummies(d, columns=["InternetService","Contract","PaymentMethod"],
                       drop_first=True)
    d = d.astype({c: int for c in d.select_dtypes("bool").columns})
    d.drop(columns=["customerID","SeniorCitizen"], errors="ignore", inplace=True)
    return d

def calculate_ltv(survival_model, customer_row: pd.DataFrame,
                  monthly_charges: float, threshold: float = 0.1) -> dict:
    try:
        surv_func = survival_model.predict_survival_function(customer_row).squeeze()
        above = surv_func[surv_func > threshold]
        expected_life = float(above.index.max()) if len(above) > 0 else float(surv_func.index.min())
        return {
            "expected_lifetime_months": round(expected_life, 1),
            "ltv_dollars": round(expected_life * monthly_charges, 2)
        }
    except Exception:
        return {"expected_lifetime_months": None, "ltv_dollars": None}

# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "customerID":       "WEB-USER",
            "gender":           request.form.get("gender"),
            "SeniorCitizen":    int(request.form.get("senior_citizen", 0)),
            "Partner":          request.form.get("partner"),
            "Dependents":       request.form.get("dependents"),
            "tenure":           int(request.form.get("tenure", 0)),
            "PhoneService":     request.form.get("phone_service"),
            "MultipleLines":    request.form.get("multiple_lines"),
            "InternetService":  request.form.get("internet_service"),
            "OnlineSecurity":   request.form.get("online_security"),
            "OnlineBackup":     request.form.get("online_backup"),
            "DeviceProtection": request.form.get("device_protection"),
            "TechSupport":      request.form.get("tech_support"),
            "StreamingTV":      request.form.get("streaming_tv"),
            "StreamingMovies":  request.form.get("streaming_movies"),
            "Contract":         request.form.get("contract"),
            "PaperlessBilling": request.form.get("paperless_billing"),
            "PaymentMethod":    request.form.get("payment_method"),
            "MonthlyCharges":   float(request.form.get("monthly_charges", 0)),
            "TotalCharges":     str(float(request.form.get("tenure", 0)) *
                                    float(request.form.get("monthly_charges", 0))),
        }

        input_df = pd.DataFrame([data])
        monthly_charges = data["MonthlyCharges"]

        result = {
            "churn_probability": None,
            "churn_label": "Unknown",
            "risk_level": "unknown",
            "expected_lifetime_months": None,
            "ltv_dollars": None,
            "models_loaded": list(ARTIFACTS.keys()),
        }

        # ── ML prediction ──────────────────────────────────────────────────
        if "preprocessor" in ARTIFACTS and "model" in ARTIFACTS:
            fe_df = engineer_features(input_df)
            X = ARTIFACTS["preprocessor"].transform(fe_df)
            model = ARTIFACTS["model"]

            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(X)[:, 1][0])
            else:
                prob = float(model.predict(X)[0])

            result["churn_probability"] = round(prob * 100, 1)

            if prob >= 0.70:
                result["churn_label"] = "High Risk"
                result["risk_level"]  = "high"
            elif prob >= 0.40:
                result["churn_label"] = "Medium Risk"
                result["risk_level"]  = "medium"
            else:
                result["churn_label"] = "Low Risk"
                result["risk_level"]  = "low"

        # ── Survival / LTV ─────────────────────────────────────────────────
        if "survival" in ARTIFACTS:
            try:
                surv_df = prepare_for_survival(input_df.copy())
                surv_df.drop(columns=["TotalCharges", "Churn"], errors="ignore", inplace=True)

                # Align columns to what Cox model was trained on
                expected_cols = ARTIFACTS["survival"].params_.index.tolist()
                for col in expected_cols:
                    if col not in surv_df.columns:
                        surv_df[col] = 0
                surv_df = surv_df[expected_cols]

                ltv = calculate_ltv(ARTIFACTS["survival"], surv_df, monthly_charges)
                result.update(ltv)
            except Exception as e:
                print(f"Survival error: {e}")

        return jsonify({"status": "success", "result": result})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": list(ARTIFACTS.keys())
    })

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5001)