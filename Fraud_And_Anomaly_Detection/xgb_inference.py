import os
import json
import xgboost as xgb
import numpy as np
import pandas as pd
import joblib

from typing import Any
from sagemaker_inference import content_types, decoder, default_inference_handler


# Load model when container starts
def model_fn(model_dir):
    model_path = os.path.join(model_dir, "xgboost-model.bst")
    booster = xgb.Booster()
    booster.load_model(model_path)
    return booster

# Input handler: Parses CSV or JSON input
def input_fn(request_body, request_content_type):
    
    if request_content_type == "application/json":
        payload = json.loads(request_body)
        df = pd.DataFrame([payload])  # expects a single JSON object or dict
        return xgb.DMatrix(df)

    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


# Perform prediction
def predict_fn(input_data, model):
    return model.predict(input_data)


# Output handler
def output_fn(prediction, accept):
    if accept == "application/json":
        return json.dumps(prediction.tolist()), accept
    else:
        raise ValueError(f"Unsupported accept type: {accept}")



