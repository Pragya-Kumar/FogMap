import streamlit as tf_stream  # Streamlit for UI
import requests
import pandas as pd
import time

# Page Layout Configuration
tf_stream.set_page_config(
    page_title="FOGMAP: Live Analytics Dashboard",
    page_icon="🌫️",
    layout="wide"
)

# Title & Description
tf_stream.title("🌫️ FOGMAP: Real-Time Fog Detection & Analytics")


# Backend API URL (FastAPI Server Address)
API_URL = "http://127.0.0.1:8000/logs"

# Sidebar Controls
tf_stream.sidebar.header("📊 System Controls")
refresh_rate = tf_stream.sidebar.slider("Refresh Rate (Seconds)", min_value=1, max_value=10, value=3)
auto_refresh = tf_stream.sidebar.checkbox("Auto Refresh Logs", value=True)

# Main Dashboard Engine
def load_dashboard_data():
    try:
        # Fetch records from our FastAPI backend endpoint
        response = requests.get(API_URL, timeout=3)
        if response.status_code == 200:
            return response.json()
    except Exception:
        return []
    return []

# Dynamic Refresh Loop
while True:
    data = load_dashboard_data()
    
    if not data:
        tf_stream.warning("⚠️ Waiting for connection to FastAPI server... Make sure 'python main.py' is running!")
    else:
        # Convert JSON array to Pandas DataFrame for calculations
        df = pd.DataFrame(data)
        
        # 1. Top Metrics Row
        total_logs = len(df)
        smog_incidents = len(df[df['prediction_label'] == 'Smog'])
        avg_confidence = round(df['confidence_score'].mean(), 2) if total_logs > 0 else 0
        
        m1, m2, m3 = tf_stream.columns(3)
        m1.metric(label="Total Scanned Frames", value=total_logs)
        m2.metric(label="🚨 Active Smog Detections", value=smog_incidents, delta=f"{smog_incidents} critical triggers")
        m3.metric(label="🎯 Avg AI Confidence", value=f"{avg_confidence}%")
        
        tf_stream.markdown("---")
        
        # 2. Charts Section (Two Columns)
        c1, c2 = tf_stream.columns(2)
        
        with c1:
            tf_stream.subheader("📈 Distribution of Detections")
            label_counts = df['prediction_label'].value_counts()
            tf_stream.bar_chart(label_counts)
            
        with c2:
            tf_stream.subheader("⚡ System Latency Tracker (ms)")
            # Line chart showing performance timing trend
            tf_stream.line_chart(df['latency_ms'].tail(15))
            
        tf_stream.markdown("---")
        
        # 3. Interactive Historical Data Table
        tf_stream.subheader("📋 Live Log Spreadsheet (Neon Synced)")
        tf_stream.dataframe(df[['id', 'timestamp', 'prediction_label', 'confidence_score', 'visibility_level', 'latency_ms']], use_container_width=True)

    # Break loop immediately if user disables auto-refresh
    if not auto_refresh:
        break
        
    time.sleep(refresh_rate)
    tf_stream.rerun()