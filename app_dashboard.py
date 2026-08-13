import streamlit as st
import pandas as pd
import plotly.express as px
import pydeck as pdk
from sqlalchemy import text
from datetime import datetime, timedelta

from database import engine  # reuses your existing Neon connection

# ────────────────────────────────────────────────────────
# PAGE CONFIG
# ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FOGMAP | Live Fog Monitoring",
    page_icon="🌫️",
    layout="wide",
)
    # Custom CSS for Flashing Warning Banner and System Health Badges
st.markdown(""" 
<style>
    @keyframes flash {
        0% { background-color: #ff4b4b; color: white; }
        50% { background-color: #b30000; color: white; }
        100% { background-color: #ff4b4b; color: white; }
    }
    .critical-flash-box {
        padding: 20px;
        border-radius: 8px;
        text-align: center;
        font-weight: bold;
        font-size: 20px;
        animation: flash 1.5s infinite;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .system-health-card {
        padding: 12px;
        border-radius: 6px;
        font-weight: 600;
        text-align: center;
        margin-bottom: 15px;
    }
    .health-online { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .health-offline { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    /* HIGH SEVERITY: Flashing Red Animation */
    @keyframes flash-red {
        0% { background-color: #ff4b4b; color: white; }
        50% { background-color: #b30000; color: white; }
        100% { background-color: #ff4b4b; color: white; }
    }
    .severity-high {
        padding: 15px; border-radius: 8px; font-weight: bold; text-align: center;
        animation: flash-red 1.5s infinite; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* MEDIUM SEVERITY: Solid Orange Color Box */
    .severity-medium {
        padding: 15px; border-radius: 8px; font-weight: bold; text-align: center;
        background-color: #ffa500; color: black; margin-bottom: 15px;
        border: 2px solid #cc8400; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    /* LOW SEVERITY: Solid Yellow Color Box */
    .severity-low {
        padding: 15px; border-radius: 8px; font-weight: bold; text-align: center;
        background-color: #ffeb3b; color: black; margin-bottom: 15px;
        border: 2px solid #fbc02d; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────
# CAMERA LOCATION → GPS MAPPING
# ────────────────────────────────────────────────────────
CAMERA_COORDS = {
    "Van_Camera_01": {"lat": 21.1938, "lon": 81.9137},  # Raipur, CG
    "Van_Camera_02": {"lat": 21.2514, "lon": 81.6296},
    "Van_Camera_03": {"lat": 21.0500, "lon": 82.0500},
    "Demonstration_Van_01": {"lat": 21.1600, "lon": 81.8500},
    "Test_Bench_Van": {"lat": 21.2300, "lon": 81.9800},
}
DEFAULT_COORD = {"lat": 21.1938, "lon": 81.9137}

# ────────────────────────────────────────────────────────
# DATA LOADING
# ────────────────────────────────────────────────────────
@st.cache_data(ttl=15)
def load_detections(hours: int = 24) -> pd.DataFrame:
    query = text("""
        SELECT id, timestamp, camera_location, prediction_label, 
               confidence_score, visibility_level, alert_sent, latency_ms 
        FROM fog_detections 
        WHERE timestamp >= :cutoff 
        ORDER BY timestamp DESC
    """)
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"cutoff": cutoff})
    return df

@st.cache_data(ttl=15)
def load_active_alerts() -> pd.DataFrame:
    query = text("""
        SELECT a.alert_id, a.alert_level, a.is_resolved,
               d.timestamp, d.camera_location, d.confidence_score, d.visibility_level
        FROM active_alerts a
        JOIN fog_detections d ON a.detection_id = d.id
        WHERE a.is_resolved = false
        ORDER BY d.timestamp DESC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def resolve_alert(alert_id: int):
    with engine.begin() as conn:
        conn.execute(
            text("UPDATE active_alerts SET is_resolved = true WHERE alert_id = :aid"),
            {"aid": alert_id},
        )
    st.cache_data.clear()

# ────────────────────────────────────────────────────────
# SIDEBAR — FILTERS
# ────────────────────────────────────────────────────────
st.sidebar.title("🌫️ FOGMAP Controls")
time_window = st.sidebar.selectbox(
    "Time window",
    options=[6, 12, 24, 48, 168, 720],
    index=4,
    format_func=lambda h: f"Last {h} hours" if h < 720 else "Last 30 days",
)
if st.sidebar.button("🔄 Refresh now"):
    st.cache_data.clear()

st.sidebar.caption("Dashboard auto-refreshes data every 15 seconds.")

# ────────────────────────────────────────────────────────
# LOAD DATA
# ────────────────────────────────────────────────────────
df = load_detections(hours=time_window)
st.sidebar.markdown("---")
st.sidebar.markdown("### 🖥️ Network Infrastructure Status")

if not df.empty:
    # Sabse aakhiri frame validation log kab aaya
    latest_time = pd.to_datetime(df['timestamp'].iloc[0])
    time_diff = (datetime.utcnow() - latest_time).total_seconds() / 60  # calculates delay in minutes
    
    # Critical Threshold Check: Agar camera 5 minute se zyada silent hai -> Offline
    if time_diff <= 5.0:
        st.sidebar.markdown('<div class="system-health-card health-online">🟢 CCTV Node Matrix: Active</div>', unsafe_allow_html=True)
    else:
        st.sidebar.markdown('<div class="system-health-card health-offline">🔴 CCTV Node Matrix: Offline (No Feed)</div>', unsafe_allow_html=True)
else:
    st.sidebar.markdown('<div class="system-health-card health-offline">🔴 CCTV Node Matrix: Disconnected</div>', unsafe_allow_html=True)

st.sidebar.info("Authorized Control Panel. National Highway Traffic Operations Division.")
alerts_df = load_active_alerts()

st.title("🌫️ FOGMAP — Live Fog Monitoring Dashboard")
st.markdown("🌐 **Ministry of Road Transport & Highways (MoRTH) - Disaster Management Panel**")
st.markdown("---")

# # Flash emergency sign if active unresolved alerts exist in database
# active_alert_count = len(alerts_df)
# if active_alert_count > 0:
#     st.markdown(
#         f'<div class="critical-flash-box">'
#         f'⚠️ CRITICAL VISIBILITY HAZARD: {active_alert_count} UNRESOLVED HIGHWAY THREATS LOGGED!<br>'
#         f'<span style="font-size: 14px; font-weight: normal;">'
#         f'Variable Message Signs (VMS) on Route NH-53 have been activated automatically to slow down traffic.</span>'
#         f'</div>',
#         unsafe_allow_html=True
#     )
# Flash emergency sign dynamically according to database severity levels
active_alert_count = len(alerts_df)

if active_alert_count > 0:
    
    latest_alert_level = str(alerts_df['alert_level'].iloc[0]).strip().lower()
    
    # Condition 1: High / Critical Severity (🔴 Red Flashing)
    if latest_alert_level in ['high', 'critical']:
        st.markdown(
            f'<div class="severity-high">'
            f'🚨 CRITICAL VISIBILITY HAZARD: {active_alert_count} HIGH-RISK THREATS ACTIVE!<br>'
            f'<span style="font-size: 14px; font-weight: normal;">Action: Variable Message Signs (VMS) on Route NH-53 initiated automatically.</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
    # Condition 2: Medium Severity (🟠 Solid Orange)
    elif latest_alert_level == 'medium':
        st.markdown(
            f'<div class="severity-medium">'
            f'⚠️ MODERATE VISIBILITY ALERT: {active_alert_count} REGIONAL HAZARDS DETECTED<br>'
            f'<span style="font-size: 14px; font-weight: normal;">Advice: Advisory speed limit signs activated. Drivers urged to use fog lights.</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        
    # Condition 3: Low Severity (🟡 Solid Yellow)
    else: 
        st.markdown(
            f'<div class="severity-low">'
            f'ℹ️ LOW SEVERITY WEATHER NOTE: Haze monitoring layers active ({active_alert_count} nodes logged)<br>'
            f'<span style="font-size: 14px; font-weight: normal;">Status: Routine tracking operational. No structural highway blockades required.</span>'
            f'</div>',
            unsafe_allow_html=True
        )

if df.empty:
    st.warning("No detection records found in the selected time window yet. Run a few predictions through the API to populate data.")
    st.stop()

# ────────────────────────────────────────────────────────
# KPI ROW
# ────────────────────────────────────────────────────────
latest = df.iloc[0]
total_detections = len(df)
smog_count = (df["prediction_label"] == "Smog").sum()
#active_alert_count = len(alerts_df)
avg_latency = round(df["latency_ms"].mean(), 1)

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Latest Status", latest["prediction_label"], f"{latest['confidence_score']}% confidence")
col2.metric("Visibility", latest["visibility_level"] or "—")
col3.metric("Total Detections", total_detections)
col4.metric("Active Alerts", active_alert_count, delta_color="inverse" if active_alert_count > 0 else "normal")
col5.metric("Avg Latency", f"{avg_latency} ms")

st.divider()

# ────────────────────────────────────────────────────────
# ACTIVE ALERTS PANEL
# ────────────────────────────────────────────────────────
st.subheader("🚨 Active Alerts")
if alerts_df.empty:
    st.success("No unresolved alerts right now.")
else:
    for _, row in alerts_df.iterrows():
        level_color = "🔴" if row["alert_level"] == "Critical" else "🟠"
        c1, c2 = st.columns([6, 1])
        c1.write(
            f"{level_color} **{row['alert_level']}** — {row['camera_location']} — "
            f"{row['visibility_level']} visibility ({row['confidence_score']}%) "
            f"at {row['timestamp']}"
        )
        if c2.button("Resolve", key=f"resolve_{row['alert_id']}"):
            resolve_alert(int(row["alert_id"]))
            st.rerun()

st.divider()

# ────────────────────────────────────────────────────────
# TREND CHARTS
# ────────────────────────────────────────────────────────
st.subheader("📈 Visibility & Confidence Trend")
trend_df = df.sort_values("timestamp")
fig_trend = px.line(
    trend_df,
    x="timestamp",
    y="confidence_score",
    color="prediction_label",
    markers=True,
    title="Confidence Score Over Time",
    labels={"confidence_score": "Confidence (%)", "timestamp": "Time"},
)
st.plotly_chart(fig_trend, use_container_width=True)

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🌥️ Fog Frequency Analysis")
    freq_df = df.copy()
    freq_df["date"] = pd.to_datetime(freq_df["timestamp"]).dt.date
    freq_counts = freq_df.groupby(["date", "prediction_label"]).size().reset_index(name="count")
    fig_freq = px.bar(
        freq_counts,
        x="date",
        y="count",
        color="prediction_label",
        barmode="group",
        title="Smog vs Clear Detections per Day",
    )
    st.plotly_chart(fig_freq, use_container_width=True)

with col_b:
    st.subheader("⚠️ Risk Indicator by Visibility Level")
    risk_counts = df["visibility_level"].value_counts().reset_index()
    risk_counts.columns = ["visibility_level", "count"]
    color_map = {"Low": "#d62728", "Moderate": "#ff7f0e", "High": "#2ca02c"}
    fig_risk = px.pie(
        risk_counts,
        names="visibility_level",
        values="count",
        color="visibility_level",
        color_discrete_map=color_map,
        title="Visibility Distribution",
    )
    st.plotly_chart(fig_risk, use_container_width=True)

st.divider()

# ────────────────────────────────────────────────────────
#  FOG HEATMAP (RISK VISUALIZATION)
# ────────────────────────────────────────────────────────
st.subheader("🗺️ Fog Risk Heatmap by Camera Location")

map_df = df.copy()
map_df["lat"] = map_df["camera_location"].map(lambda c: CAMERA_COORDS.get(c, DEFAULT_COORD)["lat"])
map_df["lon"] = map_df["camera_location"].map(lambda c: CAMERA_COORDS.get(c, DEFAULT_COORD)["lon"])

def risk_weight(row):
    if row["prediction_label"] != "Smog":
        return 1
    if row["visibility_level"] == "Low":
        return 10
    if row["visibility_level"] == "Moderate":
        return 5
    return 2

map_df["weight"] = map_df.apply(risk_weight, axis=1)

heatmap_layer = pdk.Layer(
    "HeatmapLayer",
    data=map_df,
    get_position=["lon", "lat"],
    get_weight="weight",
    radiusPixels=60,
)

scatter_layer = pdk.Layer(
    "ScatterplotLayer",
    data=map_df.drop_duplicates("camera_location"),
    get_position=["lon", "lat"],
    get_radius=800,
    get_fill_color=[0, 128, 255, 140],
    pickable=True,
)

view_state = pdk.ViewState(
    latitude=map_df["lat"].mean(),
    longitude=map_df["lon"].mean(),
    zoom=8,
    pitch=0,
)

st.pydeck_chart(pdk.Deck(
    layers=[heatmap_layer, scatter_layer],
    initial_view_state=view_state,
    tooltip={"text": "{camera_location}"},
))
#st.caption("⚠️ Coordinates are currently placeholders. Update CAMERA_COORDS for accurate mapping.")

st.divider()

# ────────────────────────────────────────────────────────
# RECENT LOGS TABLE
# ────────────────────────────────────────────────────────
# Create a downloadable CSV payload from active dataframe
export_df = df.copy()
export_df['timestamp'] = export_df['timestamp'].astype(str)
csv_payload = export_df.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Export Official Incident Log Sheets (CSV Format)",
    data=csv_payload,
    file_name=f"MORTH_FOGMAP_OfficialLog_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
    mime="text/csv",
    key="government_audit_report_downloader"
)
st.subheader("📋 Recent Fog Detection Logs")
display_df = df.copy()
display_df["alert_sent"] = display_df["alert_sent"].map({True: "🔔 Yes", False: "—"})
st.dataframe(
    display_df[["timestamp", "camera_location", "prediction_label", "confidence_score", "visibility_level", "alert_sent", "latency_ms"]],
    use_container_width=True,
    hide_index=True,
)
