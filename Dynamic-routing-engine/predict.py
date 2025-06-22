import pandas as pd
import joblib

# Load saved pipeline
model_pipeline = joblib.load("model/xgb_retry_pipeline.pkl")

# Define input sample (must match training schema exactly)
input_data = pd.DataFrame([{
    "corridor_id": "IN_EU",
    "status": "ACTIVE",
    "corridor_type": "primary",
    "success_rate_7d":0.89,
    "latency_ms": 180,
    "cost_score": 0.75,
    "past_retry_success_rate": 0.97
}])

# Predict probability
score = model_pipeline.predict_proba(input_data)[0][1]

print(f"✅ Retry Success Probability for {input_data['corridor_id'].iloc[0]}: {score:.4f}")
# import pandas as pd
# import xgboost as xgb
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import roc_auc_score
# from sklearn.preprocessing import OneHotEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# import joblib
# import os

# # Load dataset
# df = pd.read_csv("retry_corridor_dataset.csv")

# # Features & label
# X = df.drop(columns=["label"])
# y = df["label"]

# # Columns
# categorical = ["corridor_id", "status", "corridor_type"]
# numerical = ["success_rate_7d", "latency_ms", "cost_score", "past_retry_success_rate"]

# # Pipeline
# preprocessor = ColumnTransformer([
#     ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
#     ("num", "passthrough", numerical)
# ])

# model_pipeline = Pipeline([
#     ("preprocessor", preprocessor),
#     ("classifier", xgb.XGBClassifier(
#         objective="binary:logistic",
#         eval_metric="auc",
#         max_depth=6,
#         eta=0.1,
#         n_estimators=100,
#         random_state=42
#         # ❌ NO use_label_encoder
#     ))
# ])

# # Split & train
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# model_pipeline.fit(X_train, y_train)

# # Evaluate
# y_pred = model_pipeline.predict_proba(X_test)[:, 1]
# print(f"AUC: {roc_auc_score(y_test, y_pred):.4f}")

# # Save
# os.makedirs("model", exist_ok=True)
# joblib.dump(model_pipeline, "model/xgb_retry_pipeline.pkl", protocol=4)
# print("✅ Model retrained and saved fresh without `use_label_encoder`.")
