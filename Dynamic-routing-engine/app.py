from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model pipeline
model_pipeline = joblib.load("model/xgb_retry_pipeline.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        input_df = pd.DataFrame([data])
        score = model_pipeline.predict_proba(input_df)[0][1]
        return jsonify({"retry_success_probability": float(score)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=8000)
