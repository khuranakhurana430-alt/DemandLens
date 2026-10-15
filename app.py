#STREAMLIT
import streamlit as st

#PANDAS
import pandas as pd

#NUMPY
import numpy as np

#SAVE
import joblib

#PLOTLY
import plotly.express as px
import plotly.graph_objects as go

# ==================== PAGE SETUP ====================
st.set_page_config(page_title="DemandLens", page_icon="📈", layout="wide")

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
        color: #e2e8f0;
    }
    section[data-testid="stSidebar"] {
        background-color: #1e293b !important;
        border-right: 1px solid #334155;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] p {
        color: #e2e8f0 !important;
    }
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #38bdf8;
        margin-bottom: 5px;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1rem;
        margin-bottom: 25px;
    }
    .dashboard-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #334155;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #1e293b;
        border-radius: 12px;
        padding: 18px;
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-title {
        color: #94a3b8;
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        color: #38bdf8;
        font-size: 1.8rem;
        font-weight: 800;
    }
    .prediction-card {
        background: linear-gradient(135deg, #1e3a8a 0%, #312e81 100%);
        border-radius: 16px;
        padding: 30px;
        text-align: center;
        border: 1px solid #3b82f6;
        box-shadow: 0 8px 32px rgba(59, 130, 246, 0.15);
    }
    .prediction-label {
        color: #94a3b8;
        font-size: 1rem;
        font-weight: 600;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .prediction-value {
        color: #38bdf8;
        font-size: 3.5rem;
        font-weight: 900;
    }
    .info-box {
        background-color: #1e293b;
        border-left: 4px solid #38bdf8;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 20px;
        color: #cbd5e1;
    }
</style>
""", unsafe_allow_html=True)

# ==================== LOAD MODEL ====================
@st.cache_resource
def load_model():
    model = joblib.load('sales_model.pkl')
    le_family = joblib.load('le_family.pkl')
    le_type = joblib.load('le_type.pkl')
    features = joblib.load('features.pkl')
    config = joblib.load('config.pkl')
    return model, le_family, le_type, features, config

model, le_family, le_type, features, config = load_model()

# ==================== LOAD HISTORY & STORES ====================
@st.cache_data
def load_history():
    history = pd.read_csv('history.csv', parse_dates=['date'])
    stores = pd.read_csv('stores.csv')
    return history, stores

history, stores = load_history()

# ==================== SIDEBAR INPUTS ====================
st.sidebar.markdown("### 🔮 Forecast Settings")
st.sidebar.markdown("Choose store, family, and date:")

store_list = sorted(history['store_nbr'].unique())
family_list = sorted(history['family'].unique())

store_nbr = st.sidebar.selectbox("Store Number", store_list)
family = st.sidebar.selectbox("Product Family", family_list)

forecast_date = st.sidebar.date_input("Forecast Date", value=pd.Timestamp('2017-08-16'))
onpromotion = st.sidebar.slider("Items on Promotion", 0, 100, 0)
oil_price = st.sidebar.number_input("Oil Price", value=float(config['oil_last']), format="%.2f")
is_holiday = st.sidebar.selectbox("National Holiday?", [0, 1])

predict_button = st.sidebar.button("📊 Forecast Sales", use_container_width=True)

# ==================== MAIN DASHBOARD ====================
st.markdown('<p class="main-title">DemandLens</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">AI-Powered Sales & Demand Forecasting Dashboard</p>', unsafe_allow_html=True)

# Instructions Box
st.markdown("""
<div class="info-box">
    <b>📖 How to use:</b><br>
    1. Pick a <b>Store</b> and <b>Product Family</b> in the sidebar.<br>
    2. Choose a <b>Forecast Date</b> and adjust promotion settings.<br>
    3. Click <b>"Forecast Sales"</b> to see the predicted demand.
</div>
""", unsafe_allow_html=True)

# ==================== METRIC ROW ====================
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(f'<div class="metric-card"><div class="metric-title">RMSLE</div><div class="metric-value">{config["rmsle"]:.3f}</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown(f'<div class="metric-card"><div class="metric-title">RMSE</div><div class="metric-value">{config["rmse"]:.1f}</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown(f'<div class="metric-card"><div class="metric-title">MAE</div><div class="metric-value">{config["mae"]:.1f}</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown(f'<div class="metric-card"><div class="metric-title">R² Score</div><div class="metric-value">{config["r2"]:.3f}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ==================== PREDICTION ====================
if predict_button:
    try:
        # Get last 30 days for this store-family combo
        filtered = history[
            (history['store_nbr'] == store_nbr) &
            (history['family'] == family)
        ].sort_values('date')

        if len(filtered) < 7:
            st.error(f"Not enough history for this combination. Found {len(filtered)} days.")
            st.stop()

        sales_series = filtered['sales'].values

        # Build features
        lag_1 = sales_series[-1]
        lag_7 = sales_series[-7]
        rolling_mean_7 = np.mean(sales_series[-7:])

        # Get store info
        store_info = stores[stores['store_nbr'] == store_nbr].iloc[0]
        store_type = store_info['type']
        cluster = store_info['cluster']

        family_encoded = le_family.transform([family])[0]
        type_encoded = le_type.transform([store_type])[0]

        forecast_date = pd.to_datetime(forecast_date)

        # Build feature row
        input_data = pd.DataFrame({
            'store_nbr': [store_nbr],
            'family': [family_encoded],
            'onpromotion': [onpromotion],
            'type': [type_encoded],
            'cluster': [cluster],
            'oil_price': [oil_price],
            'is_holiday': [is_holiday],
            'year': [forecast_date.year],
            'month': [forecast_date.month],
            'day': [forecast_date.day],
            'dayofweek': [forecast_date.dayofweek],
            'weekofyear': [forecast_date.isocalendar()[1]],
            'is_weekend': [1 if forecast_date.dayofweek >= 5 else 0],
            'lag_1': [lag_1],
            'lag_7': [lag_7],
            'rolling_mean_7': [rolling_mean_7]
        })

        input_data = input_data[features]

        # Predict
        pred_log = model.predict(input_data)[0]
        prediction = np.expm1(pred_log)
        prediction = max(0, prediction)

        # Show prediction card
        st.markdown("---")
        col_pred, col_info = st.columns([2, 1])

        with col_pred:
            st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-label">Predicted Sales for {family}</div>
                <div class="prediction-label">Store {store_nbr} — {forecast_date.date()}</div>
                <div class="prediction-value">{prediction:,.0f} units</div>
            </div>
            """, unsafe_allow_html=True)

        with col_info:
            st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
            st.markdown("**📋 Forecast Context**")
            st.markdown(f"**City:** {store_info['city']}")
            st.markdown(f"**State:** {store_info['state']}")
            st.markdown(f"**Store Type:** {store_type}")
            st.markdown(f"**Cluster:** {cluster}")
            st.markdown(f"**Promotion:** {onpromotion} items")
            st.markdown(f"**Holiday:** {'Yes' if is_holiday == 1 else 'No'}")
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

    except Exception as e:
        st.error(f"Error: {e}")
else:
    st.info("👈 Set your options in the sidebar and click 'Forecast Sales'.")
    st.markdown("---")

# ==================== CHART 1: SALES TREND ====================
st.markdown('<div class="dashboard-card"><h4 style="color: #38bdf8;">📈 Sales Trend (Last 30 Days)</h4>', unsafe_allow_html=True)

filtered_trend = history[
    (history['store_nbr'] == store_nbr) &
    (history['family'] == family)
].sort_values('date')

if len(filtered_trend) > 0:
    fig_line = px.line(
        filtered_trend, x='date', y='sales',
        markers=True, color_discrete_sequence=['#38bdf8']
    )
    fig_line.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e2e8f0'),
        margin=dict(t=10, b=10, l=10, r=10),
        xaxis=dict(showgrid=False, title='Date'),
        yaxis=dict(showgrid=True, gridcolor='#334155', title='Units Sold'),
        height=320
    )
    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.info("No history available.")

st.markdown('</div>', unsafe_allow_html=True)

# ==================== ROW: TWO CHARTS ====================
st.markdown("<br>", unsafe_allow_html=True)
col_left, col_right = st.columns(2)

# ==================== CHART 2: WEEKLY PATTERN ====================
with col_left:
    st.markdown('<div class="dashboard-card"><h4 style="color: #38bdf8;">📊 Weekly Sales Pattern</h4>', unsafe_allow_html=True)

    if len(filtered_trend) > 0:
        filtered_trend['day_name'] = filtered_trend['date'].dt.day_name()
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_avg = filtered_trend.groupby('day_name')['sales'].mean().reindex(day_order).reset_index()

        fig_bar = px.bar(
            day_avg, x='day_name', y='sales',
            color='sales', color_continuous_scale=['#38bdf8', '#8b5cf6']
        )
        fig_bar.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(showgrid=False, title=''),
            yaxis=dict(showgrid=True, gridcolor='#334155', title='Avg Sales'),
            coloraxis_showscale=False,
            height=320
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    else:
        st.info("No data for this combination.")

    st.markdown('</div>', unsafe_allow_html=True)

# ==================== CHART 3: FAMILY SHARE ====================
with col_right:
    st.markdown('<div class="dashboard-card"><h4 style="color: #38bdf8;">🍩 Top 5 Families in Store</h4>', unsafe_allow_html=True)

    store_data = history[history['store_nbr'] == store_nbr]
    if len(store_data) > 0:
        family_share = store_data.groupby('family')['sales'].sum().sort_values(ascending=False).head(5).reset_index()

        fig_donut = go.Figure(data=[go.Pie(
            labels=family_share['family'],
            values=family_share['sales'],
            hole=0.55,
            marker=dict(colors=['#38bdf8', '#8b5cf6', '#ec4899', '#f59e0b', '#10b981']),
            textinfo='label+percent',
            textfont=dict(color='#e2e8f0', size=11)
        )])
        fig_donut.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#e2e8f0'),
            margin=dict(t=10, b=10, l=10, r=10),
            showlegend=False,
            height=320
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("No data for this store.")

    st.markdown('</div>', unsafe_allow_html=True)

# ==================== CHART 4: FEATURE IMPORTANCE ====================
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="dashboard-card"><h4 style="color: #38bdf8;">🔍 Top 10 Feature Importance</h4>', unsafe_allow_html=True)

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=True).tail(10)

fig_imp = px.bar(
    importance_df, x='Importance', y='Feature', orientation='h',
    color='Importance', color_continuous_scale=['#1e3a8a', '#38bdf8']
)
fig_imp.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#e2e8f0'),
    margin=dict(t=10, b=10, l=10, r=10),
    xaxis=dict(showgrid=True, gridcolor='#334155'),
    yaxis=dict(showgrid=False),
    coloraxis_showscale=False,
    height=380
)
st.plotly_chart(fig_imp, use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)

# ==================== FOOTER ====================
st.markdown("---")
st.markdown("<p style='text-align:center; color:#64748b;'>DemandLens © 2026 | Built with Streamlit & LightGBM</p>", unsafe_allow_html=True)