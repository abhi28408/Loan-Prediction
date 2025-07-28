from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("loan_model.joblib")

# Manual mapping for categorical variables
def preprocess_input(data):
    mapping = {
        "Gender": {"Male": 1, "Female": 0},
        "Married": {"Yes": 1, "No": 0},
        "Education": {"Graduate": 1, "Not Graduate": 0},
        "Self_Employed": {"Yes": 1, "No": 0},
        "Property_Area": {"Urban": 2, "Semiurban": 1, "Rural": 0},
        "Dependents": {"0": 0, "1": 1, "2": 2, "3+": 3}
    }

    return {
        "Gender": mapping["Gender"][data["Gender"]],
        "Married": mapping["Married"][data["Married"]],
        "Dependents": mapping["Dependents"][data["Dependents"]],
        "Education": mapping["Education"][data["Education"]],
        "Self_Employed": mapping["Self_Employed"][data["Self_Employed"]],
        "ApplicantIncome": float(data["ApplicantIncome"]),
        "CoapplicantIncome": float(data["CoapplicantIncome"]),
        "LoanAmount": float(data["LoanAmount"]) / 1000,  # Ensure it's in 1000s
        "Loan_Amount_Term": float(data["Loan_Amount_Term"]),
        "Credit_History": float(data["Credit_History"]),
        "Property_Area": mapping["Property_Area"][data["Property_Area"]]
    }

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.form.to_dict()
        model_input = preprocess_input(data)
        input_df = pd.DataFrame([model_input])
        prediction = model.predict(input_df)[0]
        result = "✅ Loan Approved" if prediction == 1 else "❌ Loan Rejected"
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

