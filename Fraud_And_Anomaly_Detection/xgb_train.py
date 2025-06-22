# xgb_train.py

import argparse
import os
import pandas as pd
import xgboost as xgb
import scipy.sparse as sp
from sklearn.metrics import average_precision_score
import joblib

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--objective", type=str)
    parser.add_argument("--eval_metric", type=str)
    parser.add_argument("--scale_pos_weight", type=float)
    parser.add_argument("--max_depth", type=int)
    parser.add_argument("--subsample", type=float)
    parser.add_argument("--num_round", type=int)
    args = parser.parse_args()

    print(f"\n🔹 Loading training data from: {args.train}")

    # Load sparse matrix and labels
    X_train = sp.load_npz(os.path.join(args.train, "train.npz"))
    y_train = pd.read_csv(os.path.join(args.train, "train_labels.csv")).values.flatten()

    X_val = sp.load_npz(os.path.join(args.train, "val.npz"))
    y_val = pd.read_csv(os.path.join(args.train, "val_labels.csv")).values.flatten()

    print(f"✅ Loaded X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"✅ Loaded X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # Prepare DMatrix for training
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    print("\n🚀 Starting XGBoost training...")
    model = xgb.train(
        params={
            "objective": args.objective,
            "eval_metric": args.eval_metric,
            "scale_pos_weight": args.scale_pos_weight,
            "max_depth": args.max_depth,
            "subsample": args.subsample,
        },
        dtrain=dtrain,
        num_boost_round=args.num_round,
        evals=[(dval, "validation")],
        early_stopping_rounds=10,
        verbose_eval=True
    )

    print("\n📈 Evaluating model...")
    val_preds = model.predict(dval)
    aucpr = average_precision_score(y_val, val_preds)
    print(f"✅ Validation AUCPR: {aucpr:.4f}")

    # Save models
    os.makedirs(args.model_dir, exist_ok=True)

    bst_model_path = os.path.join(args.model_dir, "xgboost-model.bst")
    model.save_model(bst_model_path)
    print(f"📁 Saved XGBoost Booster (.bst) to: {bst_model_path}")

    joblib_model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, joblib_model_path)
    print(f"📁 Saved XGBoost model using joblib to: {joblib_model_path}")
