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
import time
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
st.markdown("*Trained on real OpenAQ sensor data from 5 Nairobi locations | Built by [Martin James Ng'ang'a](https://github.com/M20Jay)*")

# Check API health with auto retry
def check_health():
    for attempt in range(3):
        try:
            response = requests.get(
                f"{API_URL}/health",
                timeout=60
            )
            return response.json()
        except:
            if attempt < 2:
                time.sleep(10)
    return None

with st.spinner("⏳ Connecting to API... (first load may take 30 seconds)"):
    health = check_health()

if health and health['status'] in ['healthy', 'degraded']:
    if health['status'] == 'healthy':
        st.success("✅ API Connected — All models healthy")
    else:
        st.warning("⚠️ API Connected — Some models unavailable")
        st.json(health['models'])
else:
    st.warning("⏳ API is warming up. Please refresh the page in 30 seconds.")
    st.info("💡 This is normal for free tier hosting. The server wakes up on first request.")
    st.stop()

st.divider()

# Two column layout
col1, col2 = st.columns(2)

# ── FORECAST SECTION ──────────────────────────────────────
with col1:
    st.subheader("📈 PM2.5 Forecast")
    st.caption("Forecast future PM2.5 levels using ARIMA model")
    steps = st.slider("Forecast hours ahead",
                      min_value=1,
                      max_value=168,
                      value=24)

    if st.button("Generate Forecast", type="primary"):
        with st.spinner("Running ARIMA model..."):
            try:
                response = requests.post(
                    f"{API_URL}/forecast",
                    json={"steps": steps},
                    timeout=60
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
                             annotation_text="WHO 24hr limit (15 µg/m³)")
                fig.add_hline(y=5, line_dash="dash",
                             line_color="green",
                             annotation_text="WHO annual limit (5 µg/m³)")
                fig.update_layout(
                    title=f"PM2.5 Forecast — Next {steps} Hours",
                    xaxis_title="Time",
                    yaxis_title="PM2.5 (µg/m³)",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(df.set_index('datetime'))

            except Exception as e:
                st.error(f"Forecast failed: {str(e)}")

# ── ANOMALY SECTION ───────────────────────────────────────
with col2:
    st.subheader("🚨 Anomaly Detection")
    st.caption("Check if a PM2.5 reading is dangerous using Isolation Forest")

    pm25_input = st.number_input(
        "Enter PM2.5 reading (µg/m³)",
        min_value=0.0,
        max_value=1000.0,
        value=12.0,
        step=0.1
    )

    st.caption("Try: 8.5 (normal) | 55.0 (unhealthy) | 469.23 (hazardous spike detected in Nairobi Feb 2024)")

    if st.button("Check Reading", type="primary"):
        with st.spinner("Running Isolation Forest..."):
            try:
                response = requests.post(
                    f"{API_URL}/anomaly",
                    json={"pm25": pm25_input},
                    timeout=60
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

            except Exception as e:
                st.error(f"Detection failed: {str(e)}")

st.divider()

# Footer
st.markdown("""
**About this system:**
- **Data:** 11,998 real PM2.5 readings from 5 Nairobi locations via OpenAQ API
- **Models:** ARIMA (RMSE: 9.93) · Prophet (RMSE: 22.05) · LSTM PyTorch (RMSE: 19.46) · Isolation Forest
- **Finding:** Nairobi PM2.5 peaks at 4am daily — 93x WHO safe limit recorded Feb 2024
- **API:** [Live Docs](https://air-quality-anomaly-detection.onrender.com/docs) | **GitHub:** [M20Jay](https://github.com/M20Jay)
""")
st.caption("Built by Martin James Ng'ang'a | MLOps Engineer | Nairobi, Kenya 🇰🇪")