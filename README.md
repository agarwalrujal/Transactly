# Transactly
Transactly is a full-stack intelligent payment routing and fraud detection platform that simulates a real-time cross-border payment ecosystem. It leverages AWS services, AI/ML models, and C++/Python microservices to compute the most optimal and compliant routes for financial transactions.


#DYNAMIC ROUTING ENGINE
The Dynamic Routing Engine is a C++-based system that computes the most optimal top-K transaction paths across global payment corridors. It leverages an A*-based search algorithm, integrates real-time ML retry scoring via HTTP from a Python Flask server, and enforces regional compliance filters.

This engine is core to the intelligent path selection in the Transactly platform, simulating how a real-world fintech might route cross-border transactions optimally, securely, and intelligently.


XGBoost Fraud Detection Model:-

Provides a fast, explainable baseline fraud risk score based on structured transaction features. It is the first filter in the multi-model fraud detection pipeline.

Wide & Deep Neural Network (FraudScoreNet):-

Combines memorization (wide features) and generalization (deep features) to detect fraud patterns in both short-term and l Wide Features (payload["wide"])

Features:-
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
]  # Same as before, truncated for brevity
CATEGORICAL = ['country_code', 'payment_method', 'category_id']
REAL_TIME_FEATURES = [
    'amount', 'ip_risk_score', 'country_code', 'device_age_days', 'hour_of_day',
    'is_vpn', 'distance_from_home_km', 'category_id', 'cvv_attempts',
    'session_duration_sec', 'is_new_device', 'payment_method', 'shipping_billing_match'
]

AWS PERSONALISE:-

Amazon Personalize – Next Best Action (NBA)

This project implements a real-time Next Best Action (NBA) system using Amazon Personalize, designed to provide personalized user actions such as product recommendations, marketing offers, or engagement prompts.

User Personalization with Amazon Personalize

This project sets up a real-time recommendation system using the User-Personalization recipe in Amazon Personalize to suggest the most relevant products, content, or actions based on user interaction history.






