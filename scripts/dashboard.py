
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

import streamlit as st
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from src.modeling import load_data

def create_dashboard():
    st.title('Evoastra Telecom Churn Risk Dashboard')
    st.markdown('**Client Confidential** - Market share Churn Prediction System')
    
    # Load and preprocess data
    predictions_df,X,y = load_data('../data/final/telecom_churn_features.csv')
    
    
    # Load the model pipline
    final_model = joblib.load("../models/final_churn_pipeline.pkl")
    # Extract the trained model from your pipeline
    gb_model = final_model.named_steps["gradientboostingclassifier"]
    # Preprocess your test data
    X_test_encoded = final_model.named_steps["onehotencoder"].transform(X)
    X_test_scaled = final_model.named_steps["standardscaler"].transform(X_test_encoded)
    
    # Add a column for churn probability predictions
    predictions_df['churn_probability'] = gb_model.predict_proba(X_test_scaled)[:, 1]
    
    # Sidebar filters
    st.sidebar.header('Filters')
    selected_circles = st.sidebar.multiselect(
    'Select Circles',
    options=predictions_df['circle'].unique(),
    default=predictions_df['circle'].unique()[:5]
    )
    selected_operators = st.sidebar.multiselect(
    'Select Operators',
    options=predictions_df['service_provider'].unique(),
    default=predictions_df['service_provider'].unique()[:5]
    )
    
    # Filter data
    filtered_df = predictions_df[
    (predictions_df['circle'].isin(selected_circles)) &
    (predictions_df['service_provider'].isin(selected_operators))
    ]

    # Main dashboard
    col1, col2, col3 = st.columns(3)
    with col1:
            st.metric('Total Records', len(filtered_df))
    with col2:
        avg_churn_risk = filtered_df['churn_probability'].mean()
        st.metric('Average Churn Risk', f'{avg_churn_risk:.2%}')
    with col3:
        high_risk_count = (filtered_df['churn_probability'] > 0.5).sum()
        st.metric('High Risk Cases', high_risk_count)
    
    # Churn risk heatmap
    st.subheader('Churn Risk by Circle and Operator')
    risk_pivot = filtered_df.pivot_table(
    values='churn_probability',
    index='circle',
    columns='service_provider',
    aggfunc='mean'
    )
    st.dataframe(risk_pivot.style.background_gradient(cmap='Reds').format("{:.2%}"))
    fig, ax = plt.subplots()
    sns.heatmap(risk_pivot, annot=True, fmt=".2%", cmap='Reds', ax=ax)
    plt.title('Average Churn Probability Heatmap')
    # Display in Streamlit
    st.pyplot(fig)
    # Time series of churn risk
    st.subheader('Churn Risk Trends Over Time')
    time_series = filtered_df.groupby('date')['churn_probability'].mean().reset_index()
    st.dataframe(time_series)
    fig2, ax2 = plt.subplots()
    sns.lineplot(data=time_series, x='date', y='churn_probability', ax=ax2)
    plt.title('Average Churn Probability Over Time')
    plt.ylabel('Average Churn Probability')
    plt.xlabel('Date')
    plt.xticks(rotation=45)
    st.pyplot(fig2)




if __name__ == "__main__":
    create_dashboard()