# =============================================================
# streamlit_app.py
# =============================================================
# Purpose: Interactive dashboard for air quality monitoring.
#          Visualises PM2.5 forecasts and anomaly detection.
#
# Author: Martin James Ng'ang'a | github.com/M20Jay
# =============================================================

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

API_URL = "https://air-quality-anomaly-detection.onrender.com"

st.set_page_config(
    page_title="Nairobi Air Quality Monitor",
    page_icon="🌍",
    layout="wide"
)

# Header
st.title("🌍 Nairobi Air Quality Monitor")
st.markdown("Real-time PM2.5 forecasting and anomaly detection — powered by ARIMA, Prophet, LSTM and Isolation Forest")

# Check API health
@st.cache_data(ttl=60)
def check_health():
    try:
        response = requests.get(f"{API_URL}/health")
        return response.json()
    except:
        return None

health = check_health()

if health and health['status'] == 'healthy':
    st.success("✅ API Connected — All models healthy")
else:
    st.error("❌ API Unavailable — Please start the API")
    st.stop()

st.divider()

# Two column layout
col1, col2 = st.columns(2)

# ── FORECAST SECTION ──────────────────────────────────────
with col1:
    st.subheader("📈 PM2.5 Forecast")
    steps = st.slider("Forecast hours ahead", 
                      min_value=1, 
                      max_value=168, 
                      value=24)
    
    if st.button("Generate Forecast"):
        with st.spinner("Running ARIMA model..."):
            response = requests.post(
                f"{API_URL}/forecast",
                json={"steps": steps}
            )
            data = response.json()
            
            now = datetime.now()
            timestamps = [now + timedelta(hours=i) 
                         for i in range(1, steps + 1)]
            
            df = pd.DataFrame({
                'datetime': timestamps,
                'pm25': data['forecast']
            })
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df['datetime'],
                y=df['pm25'],
                mode='lines+markers',
                name='PM2.5 Forecast',
                line=dict(color='steelblue', width=2)
            ))
            fig.add_hline(y=15, line_dash="dash",
                         line_color="orange",
                         annotation_text="WHO 24hr limit")
            fig.add_hline(y=5, line_dash="dash",
                         line_color="green",
                         annotation_text="WHO annual limit")
            fig.update_layout(
                title=f"PM2.5 Forecast — Next {steps} Hours",
                xaxis_title="Time",
                yaxis_title="PM2.5 (µg/m³)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(df.set_index('datetime'))

# ── ANOMALY SECTION ───────────────────────────────────────
with col2:
    st.subheader("🚨 Anomaly Detection")
    pm25_input = st.number_input(
        "Enter PM2.5 reading (µg/m³)",
        min_value=0.0,
        max_value=1000.0,
        value=12.0,
        step=0.1
    )
    
    if st.button("Check Reading"):
        with st.spinner("Running Isolation Forest..."):
            response = requests.post(
                f"{API_URL}/anomaly",
                json={"pm25": pm25_input}
            )
            data = response.json()
            
            if data['is_anomaly']:
                st.error(f"🚨 {data['message']}")
            else:
                st.success(f"✅ {data['message']}")
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric("PM2.5", f"{data['pm25']} µg/m³")
            col_b.metric("Risk Level", data['risk_level'])
            col_c.metric("Anomaly Score", 
                        round(data['anomaly_score'], 3))

st.divider()
st.caption("Built by Martin James Ng'ang'a | github.com/M20Jay | Nairobi, Kenya 🇰🇪")