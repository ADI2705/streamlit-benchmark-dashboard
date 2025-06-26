import streamlit as st
import pandas as pd
import plotly.express as px
import os
import zipfile
import tempfile

st.set_page_config(layout="wide")
st.title("Unified Benchmark Dashboard")

st.sidebar.title("Dashboard Selection")
dashboard_selection = st.sidebar.radio("Select Dashboard", ["Hardware Metrics", "Temperature Metrics", "FIO Benchmark Results"])

test_types = ['precondition', 'read', 'write', 'randread', 'randwrite', 'randrw']
hw_files = ['cpu.csv', 'fan.csv', 'mem.csv', 'temp.csv']

# --- Load CPU/MEM/FAN Data ---
@st.cache_data
def load_hw_data(base_path):
    all_data = []
    for test_type in test_types:
        hw_path = os.path.join(base_path, test_type, 'HW')
        if not os.path.isdir(hw_path):
            continue
        for file_name in ['cpu.csv', 'fan.csv', 'mem.csv']:
            file_path = os.path.join(hw_path, file_name)
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    df['test_type'] = test_type
                    df['metric_type'] = file_name.replace('.csv', '')
                    if 'timestamp' in df.columns:
                        df.rename(columns={'timestamp': 'Timestamp'}, inplace=True)
                    all_data.append(df)
                except:
                    continue
    if not all_data:
        return pd.DataFrame()

    df_combined = pd.concat(all_data, ignore_index=True)
    df_combined['Timestamp'] = pd.to_datetime(df_combined['Timestamp'], errors='coerce')
    df_combined.dropna(subset=['Timestamp'], inplace=True)

    id_vars = ['Timestamp', 'test_type', 'metric_type']
    metric_value_vars = {
        'cpu': ['User%', 'System%', 'Idle%'],
        'fan': [col for col in df_combined.columns if 'FAN' in col and 'RPM' in col],
        'mem': ['Total_Memory_MB', 'Used_Memory_MB', 'Free_Memory_MB', 'Shared_Memory_MB', 'Buffer_Cache_MB', 'Available_Memory_MB']
    }

    melted = []
    for mtype in df_combined['metric_type'].unique():
        df = df_combined[df_combined['metric_type'] == mtype].copy()
        value_vars = [col for col in metric_value_vars.get(mtype, []) if col in df.columns]
        if not value_vars:
            value_vars = [col for col in df.columns if col not in id_vars and col.lower() != 'timestamp']
        if value_vars:
            melted.append(df.melt(id_vars=id_vars, value_vars=value_vars,
                                  var_name='HardwareMetric', value_name='Value'))

    if not melted:
        return pd.DataFrame()
    final_df = pd.concat(melted, ignore_index=True)
    final_df['Value'] = pd.to_numeric(final_df['Value'], errors='coerce')
    return final_df.dropna(subset=['Value'])

# --- Load TEMP Data (dedicated) ---
@st.cache_data
def load_temp_data(base_path):
    all_data = []
    for test_type in test_types:
        temp_path = os.path.join(base_path, test_type, 'HW', 'temp.csv')
        if os.path.exists(temp_path):
            try:
                df = pd.read_csv(temp_path)
                df['test_type'] = test_type
                if 'timestamp' in df.columns:
                    df.rename(columns={'timestamp': 'Timestamp'}, inplace=True)
                all_data.append(df)
            except:
                continue
    if not all_data:
        return pd.DataFrame()
    df_combined = pd.concat(all_data, ignore_index=True)
    if 'Timestamp' not in df_combined.columns:
        return pd.DataFrame()
    df_combined['Timestamp'] = pd.to_datetime(df_combined['Timestamp'], errors='coerce')
    df_combined.dropna(subset=['Timestamp'], inplace=True)
    id_vars = ['Timestamp', 'test_type']
    value_vars = [col for col in df_combined.columns if col not in id_vars]
    df_melted = df_combined.melt(id_vars=id_vars, value_vars=value_vars,
                                 var_name='TemperatureMetric', value_name='Value')
    df_melted['Value'] = pd.to_numeric(df_melted['Value'], errors='coerce')
    return df_melted.dropna(subset=['Value'])

# --- Load FIO Data ---
@st.cache_data
def load_fio_data(file):
    try:
        df = pd.read_csv(file)
        grouped = df.groupby(['device', 'test_type', 'iodepth', 'numjobs'])
        df_mean = grouped[['bw_mib_s', 'iops', 'latency_us']].mean().reset_index()
        df_mean['iodepth_numjobs'] = df_mean['iodepth'].astype(str) + '_' + df_mean['numjobs'].astype(str)
        df_mean['device_test_type'] = df_mean['device'].astype(str) + '_' + df_mean['test_type'].astype(str)
        return df_mean
    except:
        return pd.DataFrame()

# === 1. Hardware Metrics ===
if dashboard_selection == "Hardware Metrics":
    uploaded_zip = st.sidebar.file_uploader("Upload HW ZIP (cpu/mem/fan)", type=["zip"])
    if uploaded_zip:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "hw.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            df_hw = load_hw_data(tmpdir)

        if df_hw.empty:
            st.error("No valid hardware data found.")
        else:
            st.sidebar.subheader("Filter Options")
            metric_types = df_hw['metric_type'].unique().tolist()
            selected_metrics = st.sidebar.multiselect("Metric Type", metric_types, default=metric_types)
            selected_tests = st.sidebar.multiselect("Test Type", df_hw['test_type'].unique(), default=df_hw['test_type'].unique())
            filtered = df_hw[df_hw['metric_type'].isin(selected_metrics) & df_hw['test_type'].isin(selected_tests)]
            for metric in selected_metrics:
                df_metric = filtered[filtered['metric_type'] == metric]
                if not df_metric.empty:
                    st.subheader(f"{metric.upper()} Metrics")
                    fig = px.line(df_metric, x="Timestamp", y="Value",
                                  color="HardwareMetric", line_dash="test_type",
                                  title=f"{metric.upper()} Trends")
                    st.plotly_chart(fig, use_container_width=True)

# === 2. Temperature Metrics ===
elif dashboard_selection == "Temperature Metrics":
    uploaded_zip = st.sidebar.file_uploader("Upload HW ZIP (for temp)", type=["zip"])
    if uploaded_zip:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "temp.zip")
            with open(zip_path, "wb") as f:
                f.write(uploaded_zip.getbuffer())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)
            df_temp = load_temp_data(tmpdir)

        if df_temp.empty:
            st.error("No temperature data found.")
        else:
            st.sidebar.subheader("Filter Options")
            test_types_selected = st.sidebar.multiselect("Test Types", df_temp['test_type'].unique(),
                                                         default=df_temp['test_type'].unique())
            metrics_selected = st.sidebar.multiselect("Temperature Metrics", df_temp['TemperatureMetric'].unique(),
                                                      default=df_temp['TemperatureMetric'].unique())
            filtered = df_temp[df_temp['test_type'].isin(test_types_selected) &
                               df_temp['TemperatureMetric'].isin(metrics_selected)]
            if filtered.empty:
                st.warning("No data matches selected filters.")
            else:
                fig = px.line(filtered, x="Timestamp", y="Value",
                              color="TemperatureMetric", line_dash="test_type",
                              title="Temperature Trends")
                st.plotly_chart(fig, use_container_width=True)

# === 3. FIO Benchmark Results ===
elif dashboard_selection == "FIO Benchmark Results":
    uploaded_csv = st.sidebar.file_uploader("Upload fio_results.csv", type=["csv"])
    if uploaded_csv:
        df_fio = load_fio_data(uploaded_csv)
        if df_fio.empty:
            st.error("FIO CSV could not be read.")
        else:
            st.sidebar.subheader("Filter Options")
            show_iops = st.sidebar.checkbox("IOPS", True)
            show_bw = st.sidebar.checkbox("Bandwidth", True)
            show_lat = st.sidebar.checkbox("Latency", True)
            selected_devs = st.sidebar.multiselect("Device", df_fio['device'].unique(), default=df_fio['device'].unique())
            selected_tests = st.sidebar.multiselect("Test Type", df_fio['test_type'].unique(), default=df_fio['test_type'].unique())
            filtered = df_fio[df_fio['device'].isin(selected_devs) & df_fio['test_type'].isin(selected_tests)]

            if filtered.empty:
                st.warning("No data for selected filters.")
            else:
                if show_iops:
                    fig = px.line(filtered, x="iodepth_numjobs", y="iops", color="device_test_type", title="IOPS")
                    st.plotly_chart(fig, use_container_width=True)
                if show_bw:
                    fig = px.line(filtered, x="iodepth_numjobs", y="bw_mib_s", color="device_test_type", title="Bandwidth")
                    st.plotly_chart(fig, use_container_width=True)
                if show_lat:
                    fig = px.line(filtered, x="iodepth_numjobs", y="latency_us", color="device_test_type", title="Latency")
                    st.plotly_chart(fig, use_container_width=True)
