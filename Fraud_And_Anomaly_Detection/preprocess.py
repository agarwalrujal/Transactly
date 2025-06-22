# preprocess.py
import argparse
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import scipy.sparse as sp

# Feature lists
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
]

REAL_TIME_FEATURES = [
    'amount', 'ip_risk_score', 'country_code', 'device_age_days',
    'hour_of_day', 'is_vpn', 'distance_from_home_km', 'category_id',
    'cvv_attempts', 'session_duration_sec', 'is_new_device',
    'payment_method', 'shipping_billing_match'
]

CATEGORICAL = ['country_code', 'payment_method', 'category_id']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str)
    parser.add_argument("--output-data", type=str)
    parser.add_argument("--preprocessor-output", type=str)
    args = parser.parse_args()

    print(f"📥 Input data path: {args.input_data}")
    print(f"📤 Output data path: {args.output_data}")
    print(f"💾 Preprocessor output path: {args.preprocessor_output}")
    
    # Load data
    input_file = os.path.join(args.input_data, "synthetic_fraud_dataset.csv")
    df = pd.read_csv(input_file)
    print(f"✅ Loaded data with shape: {df.shape}")
    
    if 'transaction_timestamp' not in df.columns:
        df['transaction_timestamp'] = pd.date_range(
            start='2023-01-01', periods=len(df), freq='h'
        )
    
    # Time-based split
    df = df.sort_values('transaction_timestamp')
    train = df.iloc[:int(0.7*len(df))]
    val = df.iloc[int(0.7*len(df)):int(0.85*len(df))]
    test = df.iloc[int(0.85*len(df)):]

    print(f"🔧 Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")

    # Preprocessing pipeline
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), [f for f in REAL_TIME_FEATURES if f not in CATEGORICAL]),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse=True), CATEGORICAL)
    ])

    # Fit and transform
    X_train_rt = preprocessor.fit_transform(train[REAL_TIME_FEATURES])
    X_val_rt = preprocessor.transform(val[REAL_TIME_FEATURES])
    X_test_rt = preprocessor.transform(test[REAL_TIME_FEATURES])

    # Save XGBoost-friendly data
    xgb_output = os.path.join(args.output_data, "xgb")
    os.makedirs(xgb_output, exist_ok=True)

    # Save sparse matrices as .npz
    sp.save_npz(os.path.join(xgb_output, "train.npz"), X_train_rt)
    sp.save_npz(os.path.join(xgb_output, "val.npz"), X_val_rt)
    sp.save_npz(os.path.join(xgb_output, "test.npz"), X_test_rt)

    # Save labels
    train['fraud_label'].to_csv(os.path.join(xgb_output, "train_labels.csv"), index=False)
    val['fraud_label'].to_csv(os.path.join(xgb_output, "val_labels.csv"), index=False)
    test['fraud_label'].to_csv(os.path.join(xgb_output, "test_labels.csv"), index=False)

    # Save Wide & Deep dataset
    wd_output = os.path.join(args.output_data, "wd")
    os.makedirs(wd_output, exist_ok=True)

    train[USER_EMBEDDINGS + REAL_TIME_FEATURES + ['fraud_label']]\
        .to_csv(os.path.join(wd_output, "train.csv"), index=False)
    val[USER_EMBEDDINGS + REAL_TIME_FEATURES + ['fraud_label']]\
        .to_csv(os.path.join(wd_output, "val.csv"), index=False)
    test[USER_EMBEDDINGS + REAL_TIME_FEATURES + ['fraud_label']]\
        .to_csv(os.path.join(wd_output, "test.csv"), index=False)

    # Save preprocessor
    os.makedirs(args.preprocessor_output, exist_ok=True)
    joblib.dump(preprocessor, os.path.join(args.preprocessor_output, "preprocessor.joblib"))

    print("✅ Preprocessing completed and saved.")
