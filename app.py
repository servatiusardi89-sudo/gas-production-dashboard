import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import streamlit as st
from PIL import Image  # <-- Import Pillow to handle image file loading

# Set up Streamlit page configuration
st.set_page_config(page_title="Texas Oil Co. - Asset Dashboard", layout="wide")

# ==========================================
# BRANDING & LOGO INTEGRATION
# ==========================================
# Load the corporate logo into the sidebar
try:
    logo_img = Image.open("logo.jpg")
    st.sidebar.image(logo_img, use_column_width=True)
except FileNotFoundError:
    # Fallback message if the image hasn't been uploaded to GitHub yet
    st.sidebar.warning("⚠️ 'logo.jpg' not found in repo root. Please upload it to display company branding.")

st.sidebar.markdown("<h2 style='text-align: center; color: #004080;'>Texas Oil Co.</h2>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# Add a dashboard title with corporate name
st.title("📊 Texas Oil Co. - Asset Performance Dashboard")
st.markdown("---")

# --- Data Generation (Cached for Performance) ---
@st.cache_data
def load_data():
    np.random.seed(42)
    dates = pd.date_range(start='2023-01-01', end='2028-03-01', freq='MS')
    n = len(dates)

    df = pd.DataFrame({'Date': dates})
    df['Producing_Wells'] = np.linspace(2, 28, n).round()
    df['Reservoir_Pressure_Psi'] = np.linspace(8500, 7500, n)
    df['Uptime_%'] = np.random.normal(95, 2, n)
    df['Avg_Productivity_Per_Well_MMSCFD'] = np.random.normal(25, 2, n)
    df['Gas_Production_MMSCFD'] = (
        df['Producing_Wells']
        * df['Avg_Productivity_Per_Well_MMSCFD']
        * (df['Uptime_%'] / 100)
    )
    return df

df_dataset = load_data()


# --- Heavy 3D Grid & Reservoir Simulation Math (Cached) ---
@st.cache_data
def generate_3d_reservoir_model():
    x_dim, y_dim = 5000, 5000  
    z_start, z_end = 11500, 13500  
    nx, ny, nz = 40, 40, 50 

    x = np.linspace(0, x_dim, nx)
    y = np.linspace(0, y_dim, ny)
    z = np.linspace(z_start, z_end, nz)

    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    P_at_grid_top = 11500 * 0.433  
    res_top = 12000
    res_bottom = 13000

    pressure = P_at_grid_top + (0.433 * (Z - z_start))
    gas_gradient = 0.15 
    gas_mask = (Z >= res_top) & (Z <= res_bottom) & (X > 1000) & (X < 4000) & (Y > 1000) & (Y < 4000)

    P_at_res_top = P_at_grid_top + (0.433 * (res_top - z_start))
    pressure[gas_mask] = P_at_res_top + (gas_gradient * (Z[gas_mask] - res_top))

    overpressure_center = [2500, 2500, 12500]
    distance = np.sqrt((X - overpressure_center[0])**2 + (Y - overpressure_center[1])**2 + (Z - overpressure_center[2])**2)
    pressure += 1200 * np.exp(- (distance / 800)**2)

    well_x, well_y = 2500, 2500
    r_well = np.sqrt((X - well_x)**2 + (Y - well_y)**2)
    r_well = np.where(r_well < 25, 25, r_well) 

    drawdown = (400 / np.log(1500 / 25)) * np.log(1500 / r_well)
    drawdown = np.clip(drawdown, 0, None)
    drawdown[(Z < res_top) | (Z > res_bottom)] = 0 

    pressure -= drawdown
    return X, Y, Z, pressure, res_top, res_bottom


# --- Placeholder ML Model Function ---
def predict_production(producing_wells, reservoir_pressure, uptime):
    avg_productivity_sim = 24.5  
    forecasted_val = producing_wells * avg_productivity_sim * (uptime / 100)
    return forecasted_val


# --- Sidebar Filters ---
st.sidebar.header("📅 Historical Date Filter")

min_date = df_dataset['Date'].min().to_pydatetime()
max_date = df_dataset['Date'].max().to_pydatetime()

selected_date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

# Sidebar Scenario Controls for Forecasting
st.sidebar.markdown("---")
st.sidebar.header("🔮 Q1 2028 Forecast Scenario")
input_wells = st.sidebar.slider("Producing Wells", min_value=1, max_value=50, value=30)
input_pressure = st.sidebar.slider("Reservoir Pressure (Psi)", min_value=5000, max_value=10000, value=7400)
input_uptime = st.sidebar.slider("Uptime %", min_value=70, max_value=100, value=96)


# --- Filter Logic ---
if len(selected_date_range) == 2:
    start_date, end_date = selected_date_range
    filtered_df = df_dataset[
        (df_dataset['Date'] >= pd.to_datetime(start_date)) & 
        (df_dataset['Date'] <= pd.to_datetime(end_date))
    ]
else:
    filtered_df = df_dataset

# --- Top-Level KPI Summary Cards ---
if not filtered_df.empty:
    kpi1, kpi2 = st.columns(2)
    with kpi1:
        avg_uptime = filtered_df['Uptime_%'].mean()
        st.metric(label="Average Asset Uptime", value=f"{avg_uptime:.2f} %")
    with kpi2:
        total_gas = filtered_df['Gas_Production_MMSCFD'].mean()
        st.metric(label="Average Gas Production", value=f"{total_gas:.2f} MMSCFD")
    st.markdown("---")
else:
    st.warning("No data available for the selected date range.")


# --- Dashboard Layout: Tabs ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🕒 Uptime Analysis", 
    "🔥 Gas Production Analysis", 
    "🔮 Production Forecasting & Planning",
    "📅 Project Horizon Timeline",
    "🛢️ 3D Reservoir Model"
])

if not filtered_df.empty:
    
    # ---------------- TAB 1: UPTIME TREND ----------------
    with tab1:
        st.subheader("Asset Uptime Analysis")
        fig1, ax1 = plt.subplots(figsize=(11, 4))
        ax1.plot(filtered_df['Date'], filtered_df['Uptime_%'], marker='o', linestyle='-', color='#1f77b4', markersize=4)
        ax1.set_title('Uptime % Trend Over Time', fontsize=14, fontweight='bold', pad=15)
        ax1.set_xlabel('Month-Year', fontsize=12)
        ax1.set_ylabel('Uptime (%)', fontsize=12)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(filtered_df) // 10)))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax1.get_xticklabels(), rotation=45)
        fig1.tight_layout()
        st.pyplot(fig1)

    # ---------------- TAB 2: GAS PRODUCTION TREND ----------------
    with tab2:
        st.subheader("Gas Production Exploratory Analysis")
        fig2, ax2 = plt.subplots(figsize=(11, 4))
        ax2.plot(filtered_df['Date'], filtered_df['Gas_Production_MMSCFD'], color='#ff7f0e', linewidth=2)
        ax2.set_title('Gas Production Trend Over The Years', fontsize=14, fontweight='bold', pad=15)
        ax2.set_xlabel('Year', fontsize=12)
        ax2.set_ylabel('MMSCFD', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=max(1, len(filtered_df) // 10)))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        plt.setp(ax2.get_xticklabels(), rotation=45)
        fig2.tight_layout()
        st.pyplot(fig2)

    # ---------------- TAB 3: PRODUCTION FORECASTING & PLANNING ----------------
    with tab3:
        st.subheader("What-If Scenario Production Forecasting")
        st.markdown("Adjust the operational parameters in the sidebar to dynamically forecast gas production outputs.")
        
        future_df = pd.DataFrame({
            'Producing_Wells': [input_wells],
            'Reservoir_Pressure_Psi': [input_pressure],
            'Uptime_%': [input_uptime]
        })
        
        forecast_val = predict_production(input_wells, input_pressure, input_uptime)
        
        f_col1, f_col2 = st.columns([1, 1])
        with f_col1:
            st.markdown("**Scenario Input Vector:**")
            st.dataframe(future_df, hide_index=True)
        with f_col2:
            st.metric(label="🚀 Forecasted Q1 2028 Production", value=f"{round(forecast_val, 2)} MMSCFD")
        
        st.markdown("---")
        
        # ------ Well Gap Analysis Section ------
        st.subheader("🛠️ Well Gap & Capacity Planning")
        target_production = st.number_input("Set Target Gas Production Goal (MMSCFD):", min_value=100, max_value=2000, value=700, step=50)
        
        avg_productivity_sim = 24.5  
        simulated_well_capacity = avg_productivity_sim * (input_uptime / 100)
        
        required_wells = int(np.ceil(target_production / simulated_well_capacity))
        current_wells = 20
        additional_wells = max(0, required_wells - current_wells)
        
        gap_col1, gap_col2, gap_col3 = st.columns(3)
        with gap_col1:
            st.metric(label="Current Inventory", value=f"{current_wells} Wells")
        with gap_col2:
            st.metric(label="Total Required Wells", value=f"{required_wells} Wells")
        with gap_col3:
            if additional_wells > 0:
                st.metric(label="🚨 Additional Wells Needed", value=f"{additional_wells} Wells", delta=f"+{additional_wells} Required", delta_color="inverse")
            else:
                st.metric(label="✅ Asset Status", value="Surplus Capacity", delta="No Drilling Needed")
        
        st.markdown("---")

        # ------ Financial Framework: CAPEX Matrix ------
        st.subheader("💰 Capital Expenditure (CAPEX) Estimation")
        fin_col1, fin_col2 = st.columns(2)
        with fin_col1:
            st.markdown("**Budget Unit Assumptions:**")
            cost_per_well = st.number_input("Estimated Cost Per New Well ($ USD):", min_value=500_000, max_value=20_000_000, value=2_500_000, step=250_000, format="%d")
            hub_budget = st.number_input("Central Hub/Infrastructure Overheads ($ USD):", min_value=0, max_value=100_000_000, value=15_000_000, step=1_000_000, format="%d")
            
        with fin_col2:
            well_budget = additional_wells * cost_per_well
            total_capex = well_budget + hub_budget
            st.container(border=True)
            st.markdown("### Budget Allocations")
            st.write(f"**New Drilling Allocation:** ${well_budget:,.0f} USD")
            st.write(f"**Facility/Infrastructure Allocation:** ${hub_budget:,.0f} USD")
            st.metric(label="Estimated Total CAPEX Requirements", value=f"${total_capex:,.0f} USD")

    # ---------------- TAB 4: PROJECT HORIZON TIMELINE ----------------
    with tab4:
        st.subheader("Ganal Block Phase Development & Monetization Roadmap")
        
        quarters = [
            "2023 Q1", "2023 Q2", "2023 Q3", "2023 Q4", "2024 Q1", "2024 Q2", "2024 Q3", "2024 Q4",
            "2025 Q1", "2025 Q2", "2025 Q3", "2025 Q4", "2026 Q1", "2026 Q2", "2026 Q3", "2026 Q4",
            "2027 Q1", "2027 Q2", "2027 Q3", "2027 Q4", "2028 Q1"
        ]
        gas_prod = [0, 0, 0, 0, 80, 120, 180, 220, 280, 350, 350, 350, 450, 450, 550, 550, 650, 650, 650, 700, 705]
        
        df_timeline = pd.DataFrame({"Quarter": quarters, "Gas_MMSCFD": gas_prod})
        df_timeline["Condensate_BPD"] = df_timeline["Gas_MMSCFD"] * 45 

        fig4, t_ax1 = plt.subplots(figsize=(12, 5.5))
        color_gas = "#008080"
        t_ax1.set_xlabel("Project Timeline (Fast-Track Quarterly Horizon)", fontsize=11, labelpad=10)
        t_ax1.set_ylabel("Gas Production (MMSCFD)", color=color_gas, fontsize=11)
        t_ax1.step(df_timeline["Quarter"], df_timeline["Gas_MMSCFD"], where="mid", color=color_gas, linewidth=3, marker="o", label="Fast-Track Gas Profile")
        t_ax1.tick_params(axis="y", labelcolor=color_gas)
        t_ax1.set_xticklabels(df_timeline["Quarter"], rotation=45, ha="right")
        t_ax1.grid(True, linestyle=":", alpha=0.5)
        t_ax1.axhline(y=700, color="r", linestyle="--", alpha=0.6, label="Strategic Target (700 MMSCFD)")

        t_ax2 = t_ax1.twinx()
        color_cond = "#D4AF37"
        t_ax2.set_ylabel("Condensate Production (BPD)", color=color_cond, fontsize=11)
        t_ax2.plot(df_timeline["Quarter"], df_timeline["Condensate_BPD"], color=color_cond, linestyle=":", linewidth=2, label="Condensate Yield")
        t_ax2.tick_params(axis="y", labelcolor=color_cond)

        t_ax1.axvspan(0, 3, color="red", alpha=0.05)  
        t_ax1.text(1.5, 400, "2023:\nCapex Heavy\nEPCI Phase", ha="center", color="darkred")
        t_ax1.axvspan(4, 19, color="green", alpha=0.05) 
        t_ax1.text(12, 100, "2024-2027: Monetization Phase", ha="center", color="green")
        t_ax1.annotate("First Gas (Q1 2024)\n80 MMSCFD", xy=("2024 Q1", 80), xytext=("2023 Q2", 200), arrowprops=dict(arrowstyle="->", color="black", lw=1.5))

        plt.title("Ganal Block Projected Target (Q1 2023 - Q1 2028)", fontsize=13, fontweight="bold", pad=12)
        fig4.tight_layout()
        st.pyplot(fig4)

    # ---------------- TAB 5: 3D RESERVOIR MODEL ----------------
    with tab5:
        st.subheader("Interactive Reservoir Spatial Analysis")
        st.markdown("Examine subsurface structural variations, structural boundaries, and near-wellbore spatial localized drawdown pressures.")

        X, Y, Z, pressure, res_top, res_bottom = generate_3d_reservoir_model()

        fig5 = go.Figure(data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=pressure.flatten(),
            isomin=float(pressure.min()),
            isomax=float(pressure.max()),
            opacity=0.15,
            surface_count=30,
            colorscale='Jet',
            colorbar=dict(title=dict(text="Pressure (psi)", side="right")),
        ))

        fig5.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=pressure.flatten(),
            isomin=float(pressure.max() * 0.92),
            isomax=float(pressure.max()),
            opacity=0.5,
            colorscale='Hot',
            showscale=False
        ))

        fig5.update_layout(
            title=f"3D Pressure Model: Deep Gas Reservoir ({res_top:,} - {res_bottom:,} ft)",
            scene=dict(
                xaxis_title="X (ft)",
                yaxis_title="Y (ft)",
                zaxis_title="Depth (ft)",
                zaxis=dict(autorange="reversed"), 
                camera=dict(eye=dict(x=1.4, y=1.4, z=1.4))
            ),
            margin=dict(l=0, r=0, b=0, t=40),
            height=650 
        )

        st.plotly_chart(fig5, use_container_width=True)
