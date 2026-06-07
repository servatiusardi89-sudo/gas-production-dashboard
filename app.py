import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import streamlit as st  # <-- Import Streamlit

# Set up Streamlit page configuration (Optional but looks great)
st.set_page_config(page_title="Production Dashboard", layout="wide")

# Add a dashboard title
st.title("📊 Asset Performance & Metrics Dashboard")
st.markdown("---")

# Set seed for reproducibility
np.random.seed(42)

# Generate dummy dataset
dates = pd.date_range(
    start='2023-01-01',
    end='2028-03-01',
    freq='MS'
)
n = len(dates)

df_dataset = pd.DataFrame({'Date': dates})
df_dataset['Producing_Wells'] = np.linspace(2, 28, n).round()
df_dataset['Reservoir_Pressure_Psi'] = np.linspace(8500, 7500, n)
df_dataset['Uptime_%'] = np.random.normal(95, 2, n)
df_dataset['Avg_Productivity_Per_Well_MMSCFD'] = np.random.normal(25, 2, n)
df_dataset['Gas_Production_MMSCFD'] = (
    df_dataset['Producing_Wells']
    * df_dataset['Avg_Productivity_Per_Well_MMSCFD']
    * (df_dataset['Uptime_%'] / 100)
)

# --- Streamlit Metric Section ---
# Let's show the current average uptime as a quick KPI card!
avg_uptime = df_dataset['Uptime_%'].mean()
st.metric(label="Average Asset Uptime", value=f"{avg_uptime:.2f} %")

# --- Visualization Code ---

# Create the plot and set figure size
fig, ax = plt.subplots(figsize=(10, 4))

# Plot Uptime over time
ax.plot(
    df_dataset['Date'],
    df_dataset['Uptime_%'],
    marker='o',
    linestyle='-',
    color='#1f77b4',
    markersize=4,
    label='Monthly Uptime'
)

# Add titles and labels
ax.set_title('Uptime % Trend Over Time', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Month-Year', fontsize=12)
ax.set_ylabel('Uptime (%)', fontsize=12)

# Configure grid lines for readability
ax.grid(True, linestyle='--', alpha=0.5)

# Format X-axis to show dates every 6 months to prevent overlapping labels
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

# Rotate date labels for better spacing
plt.xticks(rotation=45)

# Adjust layout to fit everything properly
plt.tight_layout()

# --- Display in Streamlit ---
# Instead of plt.savefig(), we pass the figure directly to Streamlit
st.pyplot(fig)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Gas Production Dashboard", page_icon="📊", layout="wide")
st.title("🛢️ Gas Production & Reservoir Performance Dashboard")
st.markdown("---")

@st.cache_data
def load_data():
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2028-03-01', freq='MS')
    n = len(dates)
    df = pd.DataFrame({'Date': dates})
    df['Producing_Wells'] = np.linspace(2, 28, n).round()
    df['Reservoir_Pressure'] = np.linspace(8500, 7500, n)
    df['Uptime'] = np.random.normal(95, 2, n)
    df['Productivity_Per_Well'] = np.random.normal(25, 2, n)
    df['Gas_Production_MMSCFD'] = (df['Producing_Wells'] * df['Productivity_Per_Well'] * (df['Uptime']/100))
    return df

df_dataset = load_data()

st.sidebar.header("Dashboard Filters")
pressure_filter = st.sidebar.slider(
    "Select Reservoir Pressure Range (PSI)",
    int(df_dataset['Reservoir_Pressure'].min()),
    int(df_dataset['Reservoir_Pressure'].max()),
    (int(df_dataset['Reservoir_Pressure'].min()), int(df_dataset['Reservoir_Pressure'].max()))
)

filtered_df = df_dataset[
    (df_dataset['Reservoir_Pressure'] >= pressure_filter[0]) &
    (df_dataset['Reservoir_Pressure'] <= pressure_filter[1])
]

col1, col2, col3, col4 = st.columns(4)
with col1:
    latest_prod = filtered_df['Gas_Production_MMSCFD'].iloc[-1] if not filtered_df.empty else 0
    st.metric(label="Latest Gas Production", value=f"{latest_prod:.2f} MMSCFD")
with col2:
    active_wells = filtered_df['Producing_Wells'].iloc[-1] if not filtered_df.empty else 0
    st.metric(label="Active Producing Wells", value=f"{int(active_wells)} Wells")
with col3:
    avg_pressure = filtered_df['Reservoir_Pressure'].mean() if not filtered_df.empty else 0
    st.metric(label="Avg Reservoir Pressure", value=f"{avg_pressure:.0f} PSI")
with col4:
    avg_uptime = filtered_df['Uptime'].mean() if not filtered_df.empty else 0
    st.metric(label="Avg System Uptime", value=f"{avg_uptime:.2f} %")

st.markdown("---")
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.subheader("Gas Production Trend Over Time")
    fig_prod = px.line(filtered_df, x='Date', y='Gas_Production_MMSCFD', labels={'Gas_Production_MMSCFD': 'Production (MMSCFD)'}, template="plotly_dark")
    st.plotly_chart(fig_prod, use_container_width=True)
with chart_col2:
    st.subheader("Reservoir Pressure Trend")
    fig_press = px.line(filtered_df, x='Date', y='Reservoir_Pressure', labels={'Reservoir_Pressure': 'Pressure (PSI)'}, template="plotly_dark")
    st.plotly_chart(fig_press, use_container_width=True)

st.markdown("---")
if st.checkbox("Show Raw Dataset"):
    st.subheader("Raw Data View")
    st.dataframe(filtered_df.style.format({'Reservoir_Pressure': '{:.1f}', 'Uptime': '{:.2f}%', 'Productivity_Per_Well': '{:.2f}', 'Gas_Production_MMSCFD': '{:.2f}'}))
