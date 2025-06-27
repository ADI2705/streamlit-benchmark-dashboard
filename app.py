# --- Unified Streamlit Benchmark Dashboard v2 ---

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import zipfile
import tempfile
from io import BytesIO

st.set_page_config(layout="wide")
st.title("📊 Unified Benchmark Dashboard")

# --- Sidebar Upload ---
uploaded_zip = st.sidebar.file_uploader("📦 Upload Benchmark ZIP (HW + FIO)", type=["zip"])

# --- Chart style picker ---
chart_type = st.sidebar.selectbox("📈 Chart Type", ["Line", "Bar", "Scatter"])

# --- Utility: Plot by type ---
def plot_chart(df, x, y, color, title):
    if chart_type == "Line":
        fig = px.line(df, x=x, y=y, color=color, title=title)
    elif chart_type == "Bar":
        fig = px.bar(df, x=x, y=y, color=color, title=title, barmode='group')
    elif chart_type == "Scatter":
        fig = px.scatter(df, x=x, y=y, color=color, title=title)
    return fig

# --- Metric Summary ---
def metric_summary(df, group_by, value_col):
    return df.groupby(group_by)[value_col].agg(['min', 'mean', 'max']).reset_index()

# --- Function to parse timestamp with comma separator (for temp.csv) ---
def parse_temp_timestamp(timestamp_str):
    """Parse timestamp with comma separator between date and time (temp.csv format)"""
    try:
        # Replace comma with space for proper datetime parsing
        cleaned_timestamp = str(timestamp_str).replace(',', ' ')
        return pd.to_datetime(cleaned_timestamp, format='%Y-%m-%d %H:%M:%S')
    except:
        # Fallback to pandas default parsing
        return pd.to_datetime(timestamp_str, errors='coerce')
    

# --- Load data from zip ---
def extract_and_load_data(zip_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # Load HW data
        hw_data = []
        temp_data = []
        fio_data = None
        for root, dirs, files in os.walk(tmpdir):
            for file in files:
                fpath = os.path.join(root, file)
                if file in ['cpu.csv', 'fan.csv', 'mem.csv']:
                    df = pd.read_csv(fpath)
                    test_type = root.split(os.sep)[-2] if 'HW' in root else 'unknown'
                    df['test_type'] = test_type
                    df['metric_type'] = file.replace('.csv', '')
                    if 'timestamp' in df.columns:
                        df.rename(columns={'timestamp': 'Timestamp'}, inplace=True)
                    if 'Timestamp' in df.columns:
                        # Standard timestamp parsing for cpu/mem/fan files
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                        df.dropna(subset=['Timestamp'], inplace=True)
                    hw_data.append(df)
                elif file == 'temp.csv':
                    df = pd.read_csv(fpath)
                    test_type = root.split(os.sep)[-2] if 'HW' in root else 'unknown'
                    df['test_type'] = test_type
                    if 'timestamp' in df.columns:
                        df.rename(columns={'timestamp': 'Timestamp'}, inplace=True)
                    if 'Timestamp' in df.columns:
                        # Custom timestamp parsing for temp.csv (comma separator)
                        df['Timestamp'] = df['Timestamp'].apply(parse_temp_timestamp)
                        df.dropna(subset=['Timestamp'], inplace=True)
                    temp_data.append(df)
                elif file == 'fio_results.csv':
                    fio_data = pd.read_csv(fpath)

        # HW Melt
        hw_df = pd.DataFrame()
        if hw_data:
            hw_df = pd.concat(hw_data, ignore_index=True)
            id_vars = ['Timestamp', 'test_type', 'metric_type']
            hw_df = hw_df.melt(id_vars=id_vars, var_name='HardwareMetric', value_name='Value')
            hw_df['Value'] = pd.to_numeric(hw_df['Value'], errors='coerce')
            hw_df.dropna(subset=['Value'], inplace=True)

        # TEMP Melt
        temp_df = pd.DataFrame()
        if temp_data:
            temp_df = pd.concat(temp_data, ignore_index=True)
            temp_df = temp_df.melt(id_vars=['Timestamp', 'test_type'],
                                   var_name='TemperatureMetric', value_name='Value')
            temp_df['Value'] = pd.to_numeric(temp_df['Value'], errors='coerce')
            temp_df.dropna(subset=['Value'], inplace=True)

        # FIO Process
        fio_df = pd.DataFrame()
        if fio_data is not None:
            fio_data_grouped = fio_data.groupby(['device', 'test_type', 'iodepth', 'numjobs'])
            fio_df = fio_data_grouped[['bw_mib_s', 'iops', 'latency_us']].mean().reset_index()
            fio_df['iodepth_numjobs'] = fio_df['iodepth'].astype(str) + '_' + fio_df['numjobs'].astype(str)
            fio_df['device_test_type'] = fio_df['device'] + '_' + fio_df['test_type']

        return hw_df, temp_df, fio_df

# --- Main Logic ---
if uploaded_zip:
    hw_df, temp_df, fio_df = extract_and_load_data(uploaded_zip)
    tabs = st.tabs(["🖥 Hardware Metrics", "🌡 Temperature Metrics", "⚙️ FIO Benchmark"])

    # --- HW TAB ---
    with tabs[0]:
        if hw_df.empty:
            st.warning("No hardware metrics found.")
        else:
            st.subheader("Hardware Metrics Viewer")
            col1, col2 = st.columns(2)
            with col1:
                metric_types = st.multiselect("Metric Types", hw_df['metric_type'].unique().tolist(), default=hw_df['metric_type'].unique(), key="hw_metric_types")
            with col2:
                test_types_hw = st.multiselect("Test Types", hw_df['test_type'].unique().tolist(), default=hw_df['test_type'].unique(), key="hw_test_types")
            filtered = hw_df[(hw_df['metric_type'].isin(metric_types)) & (hw_df['test_type'].isin(test_types_hw))]
            for metric in metric_types:
                df_sub = filtered[filtered['metric_type'] == metric]
                if not df_sub.empty:
                    st.markdown(f"### {metric.upper()} Metrics")
                    fig = plot_chart(df_sub, 'Timestamp', 'Value', 'HardwareMetric', f"{metric.upper()} Over Time")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(metric_summary(df_sub, 'HardwareMetric', 'Value'))

    # --- TEMP TAB ---
    with tabs[1]:
        if temp_df.empty:
            st.warning("No temperature metrics found.")
        else:
            st.subheader("Temperature Metrics Viewer")
            col1, col2 = st.columns(2)
            with col1:
                tmetrics = st.multiselect("Temperature Metrics", temp_df['TemperatureMetric'].unique().tolist(), default=temp_df['TemperatureMetric'].unique(), key="temp_metrics")
            with col2:
                test_types_temp = st.multiselect("Test Types", temp_df['test_type'].unique().tolist(), default=temp_df['test_type'].unique(), key="temp_test_types")
            filtered = temp_df[(temp_df['TemperatureMetric'].isin(tmetrics)) & (temp_df['test_type'].isin(test_types_temp))]
            if not filtered.empty:
                fig = plot_chart(filtered, 'Timestamp', 'Value', 'TemperatureMetric', "Temperature Over Time")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(metric_summary(filtered, 'TemperatureMetric', 'Value'))

    # --- FIO TAB ---
    with tabs[2]:
        if fio_df.empty:
            st.warning("No FIO results found.")
        else:
            st.subheader("FIO Benchmark Viewer")
            st.markdown("**Metric:**")
            show_iops = st.checkbox("IOPS", True)
            show_bw = st.checkbox("Bandwidth", True)
            show_lat = st.checkbox("Latency", True)

            devices = st.multiselect("Devices", fio_df['device'].unique().tolist(), default=fio_df['device'].unique(), key="fio_devices")
            tests = st.multiselect("Test Types", fio_df['test_type'].unique().tolist(), default=fio_df['test_type'].unique(), key="fio_test_types")
            filtered = fio_df[(fio_df['device'].isin(devices)) & (fio_df['test_type'].isin(tests))]

            if show_iops:
                fig = plot_chart(filtered, 'iodepth_numjobs', 'iops', 'device_test_type', "IOPS")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(metric_summary(filtered, 'device_test_type', 'iops'))
            if show_bw:
                fig = plot_chart(filtered, 'iodepth_numjobs', 'bw_mib_s', 'device_test_type', "Bandwidth")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(metric_summary(filtered, 'device_test_type', 'bw_mib_s'))
            if show_lat:
                fig = plot_chart(filtered, 'iodepth_numjobs', 'latency_us', 'device_test_type', "Latency")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(metric_summary(filtered, 'device_test_type', 'latency_us'))
else:
    st.info("Upload a ZIP file containing HW metrics (cpu/mem/fan/temp) and fio_results.csv")
