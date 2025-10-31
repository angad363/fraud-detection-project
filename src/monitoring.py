# monitoring.py - Simple monitoring dashboard

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
from datetime import datetime, timedelta
import time

st.set_page_config(page_title="Fraud Detection Monitor", layout="wide")

# Title
st.title("🔍 Real-Time Fraud Detection Monitoring Dashboard")
st.markdown("---")

# API endpoint
API_URL = "http://localhost:8000"

# Check API health
def check_api_health():
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

# Sidebar - API Status
st.sidebar.header("System Status")
api_healthy = check_api_health()

if api_healthy:
    st.sidebar.success("✅ API Online")
else:
    st.sidebar.error("❌ API Offline")
    st.sidebar.warning("Start API with: python app.py")

st.sidebar.markdown("---")
st.sidebar.header("Settings")
refresh_rate = st.sidebar.slider("Refresh Rate (seconds)", 1, 10, 5)
alert_threshold = st.sidebar.slider("Alert Threshold", 0.0, 1.0, 0.7, 0.05)

# Main content
if api_healthy:

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)

    # Simulate real-time metrics (in production, these would come from database)
    with col1:
        st.metric(
            label="Transactions Today",
            value="1,247",
            delta="123 vs yesterday"
        )

    with col2:
        st.metric(
            label="Fraud Detected",
            value="3",
            delta="-2 vs yesterday",
            delta_color="inverse"
        )

    with col3:
        st.metric(
            label="Fraud Rate",
            value="0.24%",
            delta="-0.08%",
            delta_color="inverse"
        )

    with col4:
        st.metric(
            label="Avg Response Time",
            value="45 ms",
            delta="-5 ms"
        )

    st.markdown("---")

    # Live transaction testing
    st.header("🧪 Test Transaction")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Input Transaction")
        amount = st.number_input("Amount ($)", min_value=0.0, value=150.0, step=10.0)
        time_val = st.number_input("Time (seconds)", min_value=0.0, value=406.0)

        # Simplified inputs for demo
        st.info("V1-V28 features are set to example values for demo")

        if st.button("🔍 Check for Fraud", type="primary"):

            # Sample transaction with user inputs
            transaction = {
                "Time": time_val,
                "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
                "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698,
                "V9": 0.363787, "V10": 0.090794, "V11": -0.551600, "V12": -0.617801,
                "V13": -0.991390, "V14": -0.311169, "V15": 1.468177, "V16": -0.470401,
                "V17": 0.207971, "V18": 0.025791, "V19": 0.403993, "V20": 0.251412,
                "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
                "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053,
                "Amount": amount
            }

            try:
                response = requests.post(f"{API_URL}/predict", json=transaction)
                result = response.json()

                with col2:
                    st.subheader("Prediction Result")

                    # Display result
                    fraud_prob = result['fraud_probability']

                    # Gauge chart
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=fraud_prob * 100,
                        title={'text': "Fraud Probability"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkred" if fraud_prob > alert_threshold else "green"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "lightcoral"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': alert_threshold * 100
                            }
                        }
                    ))

                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)

                    # Decision
                    if result['prediction'] == 1:
                        st.error(f"🚨 **FRAUD DETECTED**")
                    else:
                        st.success(f"✅ **LEGITIMATE TRANSACTION**")

                    st.write(f"**Risk Level:** {result['risk_level'].upper()}")
                    st.write(f"**Recommendation:** {result['recommendation']}")

            except Exception as e:
                st.error(f"Error: {e}")

    st.markdown("---")

    # Historical trends (simulated data for demo)
    st.header("📈 Historical Trends")

    # Generate sample data
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    transaction_counts = np.random.randint(800, 1500, 30)
    fraud_counts = np.random.randint(1, 10, 30)

    df_history = pd.DataFrame({
        'Date': dates,
        'Transactions': transaction_counts,
        'Frauds': fraud_counts,
        'Fraud_Rate': (fraud_counts / transaction_counts) * 100
    })

    col1, col2 = st.columns(2)

    with col1:
        # Transaction volume over time
        fig = px.line(df_history, x='Date', y='Transactions',
                      title='Daily Transaction Volume')
        fig.update_traces(line_color='#1f77b4', line_width=2)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Fraud rate over time
        fig = px.line(df_history, x='Date', y='Fraud_Rate',
                      title='Daily Fraud Rate (%)')
        fig.update_traces(line_color='#d62728', line_width=2)
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                      annotation_text="Target Threshold")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # Recent transactions table (simulated)
    st.header("📋 Recent Transactions")

    recent_transactions = pd.DataFrame({
        'Timestamp': pd.date_range(end=datetime.now(), periods=10, freq='min')[::-1],
        'Amount': np.random.uniform(10, 1000, 10).round(2),
        'Fraud_Probability': np.random.uniform(0, 1, 10).round(3),
        'Status': np.random.choice(['Approved', 'Declined', 'Review'], 10, p=[0.7, 0.2, 0.1])
    })

    # Color code by status
    def color_status(val):
        if val == 'Approved':
            return 'background-color: #d4edda'
        elif val == 'Declined':
            return 'background-color: #f8d7da'
        else:
            return 'background-color: #fff3cd'

    styled_df = recent_transactions.style.applymap(color_status, subset=['Status'])
    st.dataframe(styled_df, use_container_width=True)

    # Auto-refresh
    st.caption(f"Dashboard auto-refreshes every {refresh_rate} seconds")
    time.sleep(refresh_rate)
    st.rerun()

else:
    st.error("🚫 API is not running. Please start the API first:")
    st.code("python app.py", language="bash")