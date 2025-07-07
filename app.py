# --- Unified Streamlit Benchmark Dashboard v3 ---

import streamlit as st
import pandas as pd
import plotly.express as px
import os
import zipfile
import tempfile
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from datetime import datetime
import logging

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# --- Function to load DIMM temperature data with proper header handling ---
def load_dimm_temp_data(fpath, test_type):
    """Load DIMM temperature data with proper header handling"""
    try:
        # Read the file to check its structure
        with open(fpath, 'r') as f:
            lines = f.readlines()
        
        # Check if header is complete
        header_line = lines[0].strip()
        if header_line == "Timestamp":
            # Header is incomplete, need to add column names
            # Count the number of columns in the first data line
            first_data_line = lines[1].strip().split(',')
            num_temp_cols = len(first_data_line) - 1  # Subtract 1 for timestamp
            
            # Create proper header with DIMM sensor names
            dimm_names = ['DIMMA~D Temp', 'DIMME~H Temp']  # Add more if needed
            header = ['Timestamp'] + dimm_names[:num_temp_cols]
            
            # If more columns than predefined names, add generic ones
            if num_temp_cols > len(dimm_names):
                for i in range(len(dimm_names), num_temp_cols):
                    header.append(f'DIMM_{chr(65+i)}~X Temp')
            
            # Read data and assign proper column names
            df = pd.read_csv(fpath, header=None, names=header)
            df = df.iloc[1:]  # Skip the incomplete header row
        else:
            # Header is complete, read normally
            df = pd.read_csv(fpath)
        
        df['test_type'] = test_type
        if 'Timestamp' in df.columns:
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            df.dropna(subset=['Timestamp'], inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Error loading DIMM temperature data: {e}")
        logger.error(f"Error loading DIMM temperature data: {e}")
        return pd.DataFrame()

# --- Load data from zip ---
def extract_and_load_data(zip_file):
    with tempfile.TemporaryDirectory() as tmpdir:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            zip_ref.extractall(tmpdir)

        # Load HW data
        hw_data = []
        temp_data = []
        cpu_temp_data = []
        dimm_temp_data = []
        hba_temp_data = []
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
                elif file == 'cpu_temp.csv':
                    df = pd.read_csv(fpath)
                    test_type = root.split(os.sep)[-2] if 'HW' in root else 'unknown'
                    df['test_type'] = test_type
                    if 'Timestamp' in df.columns:
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                        df.dropna(subset=['Timestamp'], inplace=True)
                    cpu_temp_data.append(df)
                elif file == 'dimm_temp.csv':
                    df = load_dimm_temp_data(fpath, root.split(os.sep)[-2] if 'HW' in root else 'unknown')
                    if not df.empty:
                        dimm_temp_data.append(df)
                elif file == 'hba_temp.csv':
                    df = pd.read_csv(fpath)
                    test_type = root.split(os.sep)[-2] if 'HW' in root else 'unknown'
                    df['test_type'] = test_type
                    if 'Timestamp' in df.columns:
                        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
                        df.dropna(subset=['Timestamp'], inplace=True)
                    hba_temp_data.append(df)
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

        # CPU Temperature Processing
        cpu_temp_df = pd.DataFrame()
        if cpu_temp_data:
            cpu_temp_df = pd.concat(cpu_temp_data, ignore_index=True)
            cpu_temp_df = cpu_temp_df.melt(id_vars=['Timestamp', 'test_type'],
                                           var_name='CPUTempMetric', value_name='Value')
            cpu_temp_df['Value'] = pd.to_numeric(cpu_temp_df['Value'], errors='coerce')
            cpu_temp_df.dropna(subset=['Value'], inplace=True)

        # DIMM Temperature Processing
        dimm_temp_df = pd.DataFrame()
        if dimm_temp_data:
            dimm_temp_df = pd.concat(dimm_temp_data, ignore_index=True)
            dimm_temp_df = dimm_temp_df.melt(id_vars=['Timestamp', 'test_type'],
                                             var_name='DIMMTempMetric', value_name='Value')
            dimm_temp_df['Value'] = pd.to_numeric(dimm_temp_df['Value'], errors='coerce')
            dimm_temp_df.dropna(subset=['Value'], inplace=True)

        # HBA Temperature Processing
        hba_temp_df = pd.DataFrame()
        if hba_temp_data:
            hba_temp_df = pd.concat(hba_temp_data, ignore_index=True)
            hba_temp_df = hba_temp_df.melt(id_vars=['Timestamp', 'test_type'],
                                           var_name='HBATempMetric', value_name='Value')
            hba_temp_df['Value'] = pd.to_numeric(hba_temp_df['Value'], errors='coerce')
            hba_temp_df.dropna(subset=['Value'], inplace=True)

        # FIO Process
        fio_df = pd.DataFrame()
        if fio_data is not None:
            fio_data_grouped = fio_data.groupby(['device', 'test_type', 'iodepth', 'numjobs'])
            fio_df = fio_data_grouped[['bw_mib_s', 'iops', 'latency_us']].mean().reset_index()
            fio_df['iodepth_numjobs'] = fio_df['iodepth'].astype(str) + '_' + fio_df['numjobs'].astype(str)
            fio_df['device_test_type'] = fio_df['device'] + '_' + fio_df['test_type']

        return hw_df, temp_df, cpu_temp_df, dimm_temp_df, hba_temp_df, fio_df

# --- PDF Report Generation ---
def create_pdf_report(hw_df, temp_df, cpu_temp_df, dimm_temp_df, hba_temp_df, fio_df):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=72, leftMargin=72, topMargin=72, bottomMargin=18)
    elements = []

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle('CustomTitle', parent=styles['Heading1'], fontSize=16, spaceAfter=30, alignment=1)
    heading_style = ParagraphStyle('CustomHeading', parent=styles['Heading2'], fontSize=12, spaceAfter=12, textColor=colors.darkblue)
    subheading_style = ParagraphStyle('CustomSubHeading', parent=styles['Heading3'], fontSize=10, spaceAfter=10)

    elements.append(Paragraph("Benchmark Dashboard Report", title_style))
    elements.append(Spacer(1, 12))
    elements.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    elements.append(Spacer(1, 20))

    def create_table_from_df(df, title, max_rows=50):
        if df.empty:
            return []
        table_elements = [Paragraph(title, subheading_style), Spacer(1, 12)]
        display_df = df.head(max_rows)
        data = [display_df.columns.tolist()] + [[str(val) for val in row.tolist()] for _, row in display_df.iterrows()]
        table = Table(data)
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 10),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 1), (-1, -1), 8),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        table_elements.append(table)
        table_elements.append(Spacer(1, 20))
        if len(df) > max_rows:
            table_elements.append(Paragraph(f"Note: Showing first {max_rows} rows of {len(df)} total rows", styles['Italic']))
            table_elements.append(Spacer(1, 20))
        return table_elements

    def add_summary_section(df, name, metric_col, label_col, test_type=None):
        if not df.empty:
            if test_type:
                df = df[df['test_type'] == test_type]
            if not df.empty:
                summary = metric_summary(df, label_col, metric_col)
                elements.extend(create_table_from_df(summary, name))

    # Define test types and their metrics in the specified order
    test_types = ['precondition', 'write', 'read', 'randwrite', 'randread', 'randrw']
    metrics_order = [
        ('Memory Util', hw_df, 'Value', 'HardwareMetric', lambda df: df[df['metric_type'] == 'mem']),
        ('DIMM Temp', dimm_temp_df, 'Value', 'DIMMTempMetric', None),
        ('CPU Util', hw_df, 'Value', 'HardwareMetric', lambda df: df[df['metric_type'] == 'cpu']),
        ('CPU Temp', cpu_temp_df, 'Value', 'CPUTempMetric', None),
        ('Disk Temp', temp_df, 'Value', 'TemperatureMetric', None),
        ('HBA Temp', hba_temp_df, 'Value', 'HBATempMetric', None),
        ('Fan', hw_df, 'Value', 'HardwareMetric', lambda df: df[df['metric_type'] == 'fan']),
        ('BW', fio_df, 'bw_mib_s', 'device_test_type', None),
        ('IOPS', fio_df, 'iops', 'device_test_type', None),
        ('Latency', fio_df, 'latency_us', 'device_test_type', None)
    ]

    for test_type in test_types:
        elements.append(Paragraph(f"Test: {test_type.capitalize()}", heading_style))
        elements.append(Spacer(1, 12))
        for metric_name, df, value_col, label_col, filter_func in metrics_order:
            # Skip FIO metrics for precondition
            if test_type == 'precondition' and metric_name in ['BW', 'IOPS', 'Latency']:
                continue
            filtered_df = filter_func(df) if filter_func else df
            add_summary_section(filtered_df, f"{metric_name} Summary ({test_type.capitalize()})", value_col, label_col, test_type)
        elements.append(Spacer(1, 20))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# --- Main Logic ---
if uploaded_zip:
    hw_df, temp_df, cpu_temp_df, dimm_temp_df, hba_temp_df, fio_df = extract_and_load_data(uploaded_zip)
    
    # Export PDF Button
    st.sidebar.header("Export Options")
    if st.sidebar.button("Generate PDF Report", type="primary"):
        with st.spinner("Generating PDF report..."):
            try:
                pdf = create_pdf_report(hw_df, temp_df, cpu_temp_df, dimm_temp_df, hba_temp_df, fio_df)
                st.sidebar.download_button(
                    "📥 Download PDF",
                    data=pdf,
                    file_name=f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf"
                )
                st.sidebar.success("PDF generated!")
            except Exception as e:
                st.sidebar.error(f"PDF Error: {e}")
                logger.error(f"PDF Error: {e}")

    tabs = st.tabs(["🖥 Hardware Metrics", "🌡 Temperature Metrics", "🌡 CPU Temperature", "🌡 DIMM Temperature", "🌡 HBA Temperature", "⚙️ FIO Benchmark"])

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

    # --- CPU TEMP TAB ---
    with tabs[2]:
        if cpu_temp_df.empty:
            st.warning("No CPU temperature metrics found.")
        else:
            st.subheader("CPU Temperature Metrics Viewer")
            col1, col2 = st.columns(2)
            with col1:
                cpu_tmetrics = st.multiselect("CPU Temperature Metrics", cpu_temp_df['CPUTempMetric'].unique().tolist(), default=cpu_temp_df['CPUTempMetric'].unique(), key="cpu_temp_metrics")
            with col2:
                test_types_cpu_temp = st.multiselect("Test Types", cpu_temp_df['test_type'].unique().tolist(), default=cpu_temp_df['test_type'].unique(), key="cpu_temp_test_types")
            filtered = cpu_temp_df[(cpu_temp_df['CPUTempMetric'].isin(cpu_tmetrics)) & (cpu_temp_df['test_type'].isin(test_types_cpu_temp))]
            if not filtered.empty:
                fig = plot_chart(filtered, 'Timestamp', 'Value', 'CPUTempMetric', "CPU Temperature Over Time")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(metric_summary(filtered, 'CPUTempMetric', 'Value'))

    # --- DIMM TEMP TAB ---
    with tabs[3]:
        if dimm_temp_df.empty:
            st.warning("No DIMM temperature metrics found.")
        else:
            st.subheader("DIMM Temperature Metrics Viewer")
            col1, col2 = st.columns(2)
            with col1:
                dimm_tmetrics = st.multiselect("DIMM Temperature Metrics", dimm_temp_df['DIMMTempMetric'].unique().tolist(), default=dimm_temp_df['DIMMTempMetric'].unique(), key="dimm_temp_metrics")
            with col2:
                test_types_dimm_temp = st.multiselect("Test Types", dimm_temp_df['test_type'].unique().tolist(), default=dimm_temp_df['test_type'].unique(), key="dimm_temp_test_types")
            filtered = dimm_temp_df[(dimm_temp_df['DIMMTempMetric'].isin(dimm_tmetrics)) & (dimm_temp_df['test_type'].isin(test_types_dimm_temp))]
            if not filtered.empty:
                fig = plot_chart(filtered, 'Timestamp', 'Value', 'DIMMTempMetric', "DIMM Temperature Over Time")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(metric_summary(filtered, 'DIMMTempMetric', 'Value'))

    # --- HBA TEMP TAB ---
    with tabs[4]:
        if hba_temp_df.empty:
            st.warning("No HBA temperature metrics found.")
        else:
            st.subheader("HBA Temperature Metrics Viewer")
            col1, col2 = st.columns(2)
            with col1:
                hba_tmetrics = st.multiselect("HBA Temperature Metrics", hba_temp_df['HBATempMetric'].unique().tolist(), default=hba_temp_df['HBATempMetric'].unique(), key="hba_temp_metrics")
            with col2:
                test_types_hba_temp = st.multiselect("Test Types", hba_temp_df['test_type'].unique().tolist(), default=hba_temp_df['test_type'].unique(), key="hba_temp_test_types")
            filtered = hba_temp_df[(hba_temp_df['HBATempMetric'].isin(hba_tmetrics)) & (hba_temp_df['test_type'].isin(test_types_hba_temp))]
            if not filtered.empty:
                fig = plot_chart(filtered, 'Timestamp', 'Value', 'HBATempMetric', "HBA Temperature Over Time")
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(metric_summary(filtered, 'HBATempMetric', 'Value'))

    # --- FIO TAB ---
    with tabs[5]:
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
    st.info("Upload a ZIP file containing HW metrics (cpu/mem/fan/temp/cpu_temp/dimm_temp/hba_temp) and fio_results.csv")
