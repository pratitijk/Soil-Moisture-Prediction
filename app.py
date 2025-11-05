# ============================================================
# Soil Moisture Prediction - Flask Web Application
# ============================================================
# This app loads the trained RandomForest model and scaler,
# accepts user input (sensor readings) from a web form,
# and predicts soil moisture in real time.
# Includes custom input validation & error handling.
# ============================================================

from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# ------------------------------------------------------------
#  Loading Model and Scaler
# ------------------------------------------------------------
MODEL_PATH = "models/rf_model.pkl"
SCALER_PATH = "models/scaler.pkl"

if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        "Model or Scaler not found!"
    )

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# ------------------------------------------------------------
#   Safe Float Conversion with Validation
# ------------------------------------------------------------
def safe_float(value, field_name, min_val=None, max_val=None):
    """Converting value to float safely and validating ranges."""
    try:
        val = float(value)
    except ValueError:
        raise ValueError(f"Invalid input for {field_name}. Please enter a number.")

    if min_val is not None and val < min_val:
        raise ValueError(f"{field_name} cannot be less than {min_val}.")
    if max_val is not None and val > max_val:
        raise ValueError(f"{field_name} cannot be greater than {max_val}.")

    return val


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None

    if request.method == "POST":
        try:
            # ------------------------------------------------
            #  Collecting Input Data with Validation
            # ------------------------------------------------
            avg_pm1 = safe_float(request.form["avg_pm1"], "Avg PM1", 0, 8000)
            avg_pm2 = safe_float(request.form["avg_pm2"], "Avg PM2", 0, 8000)
            avg_pm3 = safe_float(request.form["avg_pm3"], "Avg PM3", 0, 8000)
            avg_am = safe_float(request.form["avg_am"], "Avg AM", 0, 8000)
            avg_lum = safe_float(request.form["avg_lum"], "Avg Luminosity", 0, 20000)
            avg_temp = safe_float(request.form["avg_temp"], "Avg Temperature (°C)", -20, 60)
            avg_humd = safe_float(request.form["avg_humd"], "Avg Humidity (%)", 0, 100)
            avg_pres = safe_float(request.form["avg_pres"], "Avg Pressure (hPa)", 800, 1200)

            # ------------------------------------------------
            #  Preparing Input for Model
            # ------------------------------------------------
            X = np.array([[avg_pm1, avg_pm2, avg_pm3, avg_am, avg_lum,
                           avg_temp, avg_humd, avg_pres]])
            X_scaled = scaler.transform(X)

            # ------------------------------------------------
            # Predicting Soil Moisture
            # ------------------------------------------------
            pred_raw = model.predict(X_scaled)[0]
            pred_percent = (pred_raw / 8000) * 100  # raw to percentage conversion logic

            prediction = f"{pred_percent:.2f} %"

        except Exception as e:
            prediction = f" Error: {e}"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
