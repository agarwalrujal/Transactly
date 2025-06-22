import os
import argparse
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Feature lists (match those used in preprocess.py)
USER_EMBEDDINGS = [
    'recency_days_since_last_txn', 'recency_days_since_last_login', 
    'recency_days_since_last_cart_abandonment', 'recency_days_since_last_support_contact',
    'recency_days_since_last_refund', 'frequency_txns_per_week', 
    'frequency_sessions_per_day', 'frequency_logins_per_month',
    'frequency_failed_logins_per_month', 'frequency_searches_per_session',
    'monetary_avg_order_value_30d', 'monetary_avg_order_value_90d',
    'monetary_total_spend_30d', 'monetary_total_spend_90d',
    'monetary_spend_volatility', 'monetary_median_order_value',
    'monetary_high_ticket_orders', 'monetary_micro_txns',
    'monetary_coupon_usage_rate', 'monetary_gift_card_redemption_rate',
    'cat_electronics', 'cat_grocery', 'cat_fashion', 'cat_home_kitchen',
    'cat_books_media', 'cat_health_beauty', 'cat_sports_outdoors',
    'cat_toys_games', 'cat_automotive', 'cat_prime_video', 'cat_prime_music',
    'cat_kindle', 'cat_aws_api', 'cat_aws_storage', 'cat_amazon_fresh',
    'cat_audible', 'cat_pharmacy', 'cat_amazon_pay', 'cat_amazon_photos',
    'temp_hour_00', 'temp_hour_01', 'temp_hour_02',
    'temp_hour_03', 'temp_hour_04', 'temp_hour_05', 'temp_hour_06',
    'temp_hour_07', 'temp_hour_08', 'temp_hour_09', 'temp_hour_10',
    'temp_hour_11', 'temp_hour_12', 'temp_hour_13', 'temp_hour_14',
    'temp_hour_15', 'temp_hour_16', 'temp_hour_17', 'temp_hour_18',
    'temp_hour_19', 'temp_hour_20', 'temp_hour_21', 'temp_hour_22',
    'temp_hour_23', 'temp_day_mon', 'temp_day_tue', 'temp_day_wed',
    'temp_day_thu', 'temp_day_fri', 'temp_day_sat', 'temp_day_sun',
    'device_pct_mobile', 'device_pct_desktop', 'device_pct_tablet',
    'device_distinct_count', 'device_entropy', 'device_pct_android',
    'device_pct_ios', 'device_pct_windows', 'device_pct_macos',
    'device_pct_headless', 'network_pct_vpn', 'network_avg_asn_score',
    'network_pct_tor', 'network_geo_mismatch', 'network_distinct_ips',
    'network_pct_new_ips', 'geo_city_count', 'geo_country_count',
    'geo_avg_distance', 'geo_max_distance_24h', 'geo_home_mismatch',
    'geo_pct_top_cities', 'geo_pct_overseas', 'geo_cross_region_hops',
    'returns_pct_returned', 'returns_avg_return_time',
    'returns_pct_refunds_approved', 'returns_chargebacks',
    'returns_pct_partial_refunds', 'support_tickets',
    'support_resolution_time', 'support_pct_escalated',
    'support_negative_feedback', 'support_pct_reopened',
    'engagement_avg_pages', 'engagement_ctr', 'engagement_pct_video',
    'engagement_pct_audio', 'engagement_pct_search',
    'engagement_search_abandon', 'engagement_wishlist_adds',
    'engagement_cart_adds', 'engagement_pct_buy_again',
    'engagement_pct_social_share', 'behavior_std_amount',
    'behavior_std_time', 'behavior_std_sessions', 'behavior_gini_spend',
    'behavior_temporal_churn', 'behavior_burstiness',
    'behavior_device_stability', 'behavior_address_stability'
]  # Same as before, truncated for brevity
CATEGORICAL = ['country_code', 'payment_method', 'category_id']
REAL_TIME_FEATURES = [
    'amount', 'ip_risk_score', 'country_code', 'device_age_days', 'hour_of_day',
    'is_vpn', 'distance_from_home_km', 'category_id', 'cvv_attempts',
    'session_duration_sec', 'is_new_device', 'payment_method', 'shipping_billing_match'
]

def load_dataset(path):
    df = pd.read_csv(path)

    X_wide = df[REAL_TIME_FEATURES].copy()
    X_deep = df[USER_EMBEDDINGS].copy()
    y = df["fraud_label"].astype(int).values

    for col in CATEGORICAL:
        if col in X_wide.columns:
            X_wide[col] = X_wide[col].astype('category').cat.codes

    scaler = StandardScaler()
    X_deep = scaler.fit_transform(X_deep)

    return (
        X_wide.values.astype(np.float32),
        X_deep.astype(np.float32),
        y.astype(np.float32),
        scaler  # return scaler in case we want to save/use it later
    )

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

def train_model(model, train_loader, val_loader, epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        for wide_x, deep_x, y in train_loader:
            wide_x, deep_x, y = wide_x.to(device), deep_x.to(device), y.to(device).unsqueeze(1)
            preds = model(wide_x, deep_x)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for wide_x, deep_x, y in val_loader:
                wide_x, deep_x = wide_x.to(device), deep_x.to(device)
                preds = model(wide_x, deep_x).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y.numpy())

        score = average_precision_score(all_labels, all_preds)
        print(f"Epoch {epoch+1} | Val AUCPR: {score:.4f}")

    return model

def create_dataloader(Xw, Xd, y, batch_size=32):
    dataset = TensorDataset(torch.tensor(Xw), torch.tensor(Xd), torch.tensor(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR", "./model"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"📂 Training data path: {args.train}")
    print(f"📂 Model output path: {args.model_dir}")

    train_csv = os.path.join(args.train, "train.csv")
    X_wide, X_deep, y, scaler = load_dataset(train_csv)

    print(f"✅ Loaded data: {X_wide.shape[0]} samples")
    print(f"🧠 Real-time feature dim: {X_wide.shape[1]}")
    print(f"🔗 User embedding dim: {X_deep.shape[1]}")

    Xw_train, Xw_val, Xd_train, Xd_val, y_train, y_val = train_test_split(
        X_wide, X_deep, y, test_size=0.2, random_state=42
    )

    train_loader = create_dataloader(Xw_train, Xd_train, y_train, args.batch_size)
    val_loader = create_dataloader(Xw_val, Xd_val, y_val, args.batch_size)

    model = WideAndDeepModel(wide_input_dim=X_wide.shape[1], deep_input_dim=X_deep.shape[1])

    print(f"\n🚀 Training model on {Xw_train.shape[0]} samples...")
    model = train_model(model, train_loader, val_loader, epochs=args.epochs, lr=args.lr)

    # Save model
    # ✅ Ensure model_dir is set correctly to SageMaker's expected path
    model_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    os.makedirs(model_dir, exist_ok=True)
    
    # ✅ Save PyTorch model weights to expected path
    model_path = os.path.join(model_dir, "wd_model.pth")
    torch.save(model.state_dict(), model_path)
    
    # ✅ Save metadata (scaler, feature lists, etc.)
    metadata_path = os.path.join(model_dir, "metadata.pkl")
    joblib.dump({
        "real_time_features": REAL_TIME_FEATURES,
        "user_embeddings": USER_EMBEDDINGS,
        "scaler": scaler
    }, metadata_path)
    
    print(f"\n✅ Model weights saved to: {model_path}")
    print(f"📁 Metadata saved to: {metadata_path}")
