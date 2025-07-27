import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

# Set Streamlit page configuration
st.set_page_config(
    page_title="Applicant Volumes by Geography",
    layout="wide"
)

# Title
st.title("Volume of Homelessness* by Geography Over Time")

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/DavidGCorrales/Homelessness/master/all_national_vols.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Age band columns
age_band_columns = [
    '16-17 Year Olds', '18-24 Year Olds', '25-34 Year Olds',
    '35-44 Year Olds', '45-54 Year Olds', '55-64 Year Olds',
    '65-74 Year Olds', '75+ Year Olds', 'Age Unknown'
]

# Calculate total
df['Total'] = df[age_band_columns].sum(axis=1)

# Extract subset
df_plot = df[['Geography', 'Quarter', 'Total']].copy()

# Convert Quarter to datetime and clean data
df_plot['QuarterDate'] = pd.to_datetime(df_plot['Quarter'], format='%Y-%m', errors='coerce')
df_plot = df_plot.dropna(subset=['QuarterDate'])

# Create sorted list of unique quarterly dates
quarter_dates = sorted(df_plot['QuarterDate'].unique())

# Use select_slider instead of slider for quarterly granularity
date_range = st.select_slider(
    "Select Quarter Range",
    options=quarter_dates,
    value=(quarter_dates[0], quarter_dates[-1]),
    format_func=lambda x: x.strftime('%Y-%m')
)

# Filter by date range
df_plot = df_plot[(df_plot['QuarterDate'] >= date_range[0]) & (df_plot['QuarterDate'] <= date_range[1])]

# Geography multiselect
geos = df_plot['Geography'].unique()
selected_geos = st.multiselect("Select Geographies", options=geos, default=list(geos))
df_plot = df_plot[df_plot['Geography'].isin(selected_geos)]

# Sort
df_plot = df_plot.sort_values(by=["Geography", "QuarterDate"])

# Plot
st.subheader("Line Chart of Total Applicants by Quarter")
fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df_plot, x="QuarterDate", y="Total", hue="Geography", ax=ax)
ax.set_xlabel("Quarter")
ax.set_ylabel("Volume")
ax.set_title("Volume of Main Applicants Assessed as Owed a Prevention or Relief Duty")
ax.set_xticks(df_plot['QuarterDate'].unique())
ax.set_xticklabels(df_plot['Quarter'].unique(), rotation=45)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
st.pyplot(fig)

# Summary table
st.subheader("Average Monthly Volume by Geography (Filtered Range)")
summary_table = df_plot.groupby('Geography')['Total'].mean().reset_index()
summary_table.columns = ['Geography', 'Average Volume']
st.dataframe(summary_table, use_container_width=True)

# Download button
csv = df_plot[['Geography', 'Quarter', 'Total']].to_csv(index=False)
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name="filtered_national_volumes.csv",
    mime="text/csv"
)
