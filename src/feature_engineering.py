import numpy as np
import pandas as pd

# 1. Define multiple churn indicators
def create_churn_targets(df):
    df_churn = df.copy()
    # Primary churn definition: &gt;15% month-over-month decline
    df_churn['churn_binary'] = (df_churn['subscriber_change'] < -0.15).astype(int)
    # Severity-based churn categories
    def categorize_churn_severity(change):
        if change < -0.30:
            return 'High'
        elif change < -0.15:
            return 'Medium'
        elif change < -0.05:
            return 'Low'
        else:
            return 'None'
    df_churn['churn_severity'] = df_churn['subscriber_change'].apply(categorize_churn_severity)

    # Sustained churn: decline for 2+ consecutive months

    return df_churn

# 2. Validate churn definitions against business logic
def validate_churn_patterns(df):
    # Analyze churn by operator category
    major_operators = ['JIO', 'AIRTEL', 'VI', 'BSNL']
    df['operator_category'] = df['service_provider'].apply(
    lambda x: 'Major' if any(op in x.upper() for op in major_operators) else 'Regional')
    # Churn analysis by operator category
    churn_by_category = df.groupby('operator_category')['churn_binary'].mean()
    print(f"Churn rates by operator category:\n{churn_by_category}")
    # Geographic churn analysis
    churn_by_circle = df.groupby('circle')['churn_binary'].mean().sort_values(ascending=False)
    print(f"Top 10 circles by churn rate:\n{churn_by_circle.head(10)}")
    return df

# 3. Create temporal features for churn prediction
def create_temporal_features(df):
    df_features = df.copy()
    df_features = df_features.sort_values(['service_provider', 'circle', 'date'])

    # Lag features
    for lag in [1, 3, 6, 12]:
        df_features[f'subscribers_lag_{lag}'] = (
            df_features.groupby(['service_provider', 'circle'])['value']
            .shift(lag)
        )

    # Growth rates
    df_features['mom_growth'] = (
        df_features.groupby(['service_provider', 'circle'])['value']
        .pct_change()
    )

    df_features['yoy_growth'] = (
        df_features.groupby(['service_provider', 'circle'])['value']
        .pct_change(periods=12)
    )

    # Volatility measures (rolling std of growth)
    for window in [3, 6, 12]:
        df_features[f'growth_volatility_{window}'] = (
            df_features.groupby(['service_provider', 'circle'])['mom_growth']
            .rolling(window=window)
            .std()
            .reset_index(level=[0,1], drop=True)
        )

    # Trend coefficients (linear slope over 12 months)
    def calculate_trend(series):
        if len(series.dropna()) < 3:
            return np.nan
        x = np.arange(len(series))
        y = series.values
        valid_indices = ~np.isnan(y)
        if valid_indices.sum() < 3:
            return np.nan
        return np.polyfit(x[valid_indices], y[valid_indices], 1)[0]

    df_features['trend_12m'] = (
        df_features.groupby(['service_provider', 'circle'])['value']
        .rolling(window=12)
        .apply(calculate_trend, raw=False)
        .reset_index(level=[0,1], drop=True)
    )

    return df_features


# 4. Advanced feature engineering: derivatives and seasonal adjustments
def create_advanced_features(df):
    df_advanced = df.copy()
    df_advanced = df_advanced.sort_values(['service_provider', 'circle', 'date'])

    # First derivative (momentum): change in subscribers
    df_advanced['growth_momentum'] = (
        df_advanced.groupby(['service_provider', 'circle'])['value']
        .diff()
    )

    # Second derivative (acceleration): change in momentum
    df_advanced['growth_acceleration'] = (
        df_advanced.groupby(['service_provider', 'circle'])['growth_momentum']
        .diff()
    )

    # Seasonal adjustment
    monthly_seasonal = df_advanced.groupby('month_num')['value'].mean()
    df_advanced['seasonal_factor'] = df_advanced['month_num'].map(monthly_seasonal)
    df_advanced['seasonally_adjusted_value'] = (
        df_advanced['value'] / df_advanced['seasonal_factor']
    )

    return df_advanced

# 5. Competitive landscape features
def create_competitive_features(df):
    df_competitive = df.copy()
    df_competitive = df_competitive.sort_values(['circle', 'date'])

    # Market share features (already calculated)

    # Market ranking: rank providers by market share within each circle-date
    df_competitive['market_rank'] = (
        df_competitive.groupby(['circle', 'date'])['market_share']
        .rank(method='dense', ascending=False)
    )

    # Gap from market leader: difference between current provider and top share
    leader_share = (
        df_competitive.groupby(['circle', 'date'])['market_share']
        .transform('max')
    )
    df_competitive['share_gap_leader'] = leader_share - df_competitive['market_share']

    # Relative performance vs circle average: subscriber count vs peer average
    circle_avg = (
        df_competitive.groupby(['circle', 'date'])['value']
        .transform('mean')
    )
    df_competitive['relative_performance'] = (
        df_competitive['value'] / circle_avg
    )

    return df_competitive

# create business features
def create_business_features(df):
    df_business = df.copy()
    df_business = df_business.sort_values(['service_provider', 'circle', 'date'])

    # Circle classification (requires domain knowledge)
    metro_circles = ['Delhi', 'Mumbai', 'Chennai', 'Kolkata', 'Hyderabad', 'Bangalore']
    df_business['circle_type'] = df_business['circle'].apply(
        lambda x: 'Metro' if x in metro_circles else 'Non-Metro'
    )

    # Connection type analysis
    df_business['is_wireless'] = (df_business['type_of_connection'] == 'wireless').astype(int)

    # Market maturity indicators (total subscribers per circle)
    total_market_size = df_business.groupby('circle')['value'].transform('sum')
    df_business['market_size_category'] = pd.qcut(total_market_size, q=3, labels=['Small','Medium','Large'])

    # Geographic diversity for each operator (number of circles served)
    operator_circle_count = df_business.groupby('service_provider')['circle'].nunique()
    df_business['operator_geographic_diversity'] = df_business['service_provider'].map(operator_circle_count)

    return df_business