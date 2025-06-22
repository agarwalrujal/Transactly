import os
import json
import torch
import joblib
import numpy as np
import pandas as pd
from torch import nn

# ----------------------
# Define Wide & Deep Model
# ----------------------
class WideAndDeepModel(nn.Module):
    def __init__(self, wide_input_dim, deep_input_dim):
        super().__init__()
        self.wide = nn.Linear(wide_input_dim, 1)
        self.deep = nn.Sequential(
            nn.Linear(deep_input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, wide_x, deep_x):
        wide_out = self.wide(wide_x)
        deep_out = self.deep(deep_x)
        return torch.sigmoid(wide_out + deep_out)

# ----------------------
# 1. Load Model
# ----------------------
def model_fn(model_dir):
    metadata = joblib.load(os.path.join(model_dir, "metadata.pkl"))
    real_time_features = metadata["real_time_features"]
    user_embeddings = metadata["user_embeddings"]
    scaler = metadata["scaler"]

    model = WideAndDeepModel(
        wide_input_dim=len(real_time_features),
        deep_input_dim=len(user_embeddings)
    )
    model.load_state_dict(torch.load(os.path.join(model_dir, "wd_model.pth"), map_location=torch.device("cpu")))
    model.eval()

    return {
        "model": model,
        "scaler": scaler,
        "real_time_features": real_time_features,
        "user_embeddings": user_embeddings
    }

# ----------------------
# 2. Parse Input
# ----------------------
def input_fn(request_body, content_type='application/json'):
    if content_type == 'application/json':
        data = json.loads(request_body)
        wide_data = data["wide"]
        deep_data = data["deep"]

        wide_df = pd.DataFrame(wide_data)
        deep_arr = np.array(deep_data)

        return {
            "wide": wide_df,
            "deep": deep_arr
        }

    raise ValueError(f"Unsupported content type: {content_type}")

# ----------------------
# 3. Predict
# ----------------------
def predict_fn(input_data, model_artifacts):
    model = model_artifacts["model"]
    scaler = model_artifacts["scaler"]
    real_time_features = model_artifacts["real_time_features"]
    user_embeddings = model_artifacts["user_embeddings"]

    # ✅ Extract correctly
    wide_df = input_data["wide"]
    deep_arr = input_data["deep"]

    # ✅ Validate all expected columns are present
    missing_cols = [col for col in real_time_features if col not in wide_df.columns]
    if missing_cols:
        raise ValueError(f"Missing real-time features: {missing_cols}")

    # Encode categorical features in wide input
    for col in wide_df.columns:
        if wide_df[col].dtype == 'object':
            wide_df[col] = wide_df[col].astype('category').cat.codes

    deep_scaled = scaler.transform(deep_arr)

    wide_tensor = torch.tensor(wide_df[real_time_features].values, dtype=torch.float32)
    deep_tensor = torch.tensor(deep_scaled, dtype=torch.float32)

    with torch.no_grad():
        output = model(wide_tensor, deep_tensor).numpy().flatten()

    return output.tolist()

# ----------------------
# 4. Format Output
# ----------------------
def output_fn(prediction, accept="application/json"):
    if accept == "application/json":
        return json.dumps({"fraud_score": float(prediction[0])})
    raise ValueError(f"Unsupported accept type: {accept}")
