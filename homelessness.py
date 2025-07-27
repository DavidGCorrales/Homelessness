import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
from datetime import datetime

# Set Streamlit page configuration
st.set_page_config(
    page_title="Applicant Volumes by Geography",
    layout="centered"
)

# Title
# st.markdown(
#     "<h1 style='text-align: center;'>Homelessness in London vs Rest of England</h1>",
#     unsafe_allow_html=True
# )
st.title("Homelessness in England since March 2021")
st.markdown("##### \"Homelessness\" refers to households initially assessed as threatened with homelessness (owed prevention duty) or homeless (owed relief duty)")

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

# # Age band multiselect
# selected_bands = st.multiselect("Select Age Bands", options=age_band_columns, default=age_band_columns)

df['Total'] = df[age_band_columns].sum(axis=1)

# # Calculate Total based on selected age bands
# df['Total'] = df[selected_bands].sum(axis=1)

# Extract subset
df_plot = df[['Geography', 'Quarter', 'Total']].copy()

# Convert Quarter to datetime and clean data
df_plot['QuarterDate'] = pd.to_datetime(df_plot['Quarter'], format='%Y-%m', errors='coerce')
df_plot = df_plot.dropna(subset=['QuarterDate'])

# Create sorted list of unique quarterly dates
quarter_dates = sorted(df_plot['QuarterDate'].unique())

# Use select_slider instead of slider for quarterly granularity
date_range = st.select_slider(
    "Select Date Range",
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

# # Plot
# st.markdown("#### London vs Rest of England")
# fig, ax = plt.subplots(figsize=(12, 6))
# sns.lineplot(data=df_plot, x="QuarterDate", y="Total", hue="Geography", ax=ax)
# ax.set_xlabel("Quarter")
# ax.set_ylabel("Volume")
# ax.set_title("Volume of Main Applicants Assessed as Owed a Prevention or Relief Duty")
# ax.set_xticks(df_plot['QuarterDate'].unique())
# ax.set_xticklabels(df_plot['Quarter'].unique(), rotation=45)
# ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
# st.pyplot(fig)

df_plot['QuarterStr'] = df_plot['QuarterDate'].dt.strftime('%Y-%m')

# Pivot with string-based index
df_volume_chart = df_plot.pivot(index='QuarterStr', columns='Geography', values='Total')

# Optional: Sort index
df_volume_chart = df_volume_chart.sort_index()

st.subheader("London vs Rest of England")
st.line_chart(df_volume_chart)

# Summary table
st.markdown("#### Average Monthly Volume")
summary_table = df_plot.groupby('Geography')['Total'].mean().reset_index()
summary_table.columns = ['Geography', 'Average Volume']
summary_table['Average Volume'] = summary_table['Average Volume'].round(0).astype(int)
st.dataframe(summary_table, use_container_width=True)

######### Add space between sections ##########
st.markdown("<br><br>", unsafe_allow_html=True)
###############################################

# Sort
df = df.sort_values(['Geography', 'Quarter'])

# Calculate baseline from March 2021
baseline = df[df['Quarter'] == '2021-03'].set_index('Geography')['Total']

# Calculate indexed % change
df['Indexed Change (%)'] = df.apply(
    lambda row: (row['Total'] / baseline[row['Geography']] - 1) * 100 if row['Geography'] in baseline else None,
    axis=1
)

# Filtered data
df_indexed = df[['Geography', 'Quarter', 'Total', 'Indexed Change (%)']].dropna()

# Plot
st.subheader("Percentage Change in Homelessness vs Selected Start Date")

# Optional: geography selector
geos = df_indexed['Geography'].unique()
selected_geos = st.multiselect("Select Geographies to Compare", options=geos, default=list(geos))
df_indexed = df_indexed[df_indexed['Geography'].isin(selected_geos)]

# Convert Quarter to datetime and clean data
df_indexed['QuarterDate'] = pd.to_datetime(df_indexed['Quarter'], format='%Y-%m', errors='coerce')
df_indexed = df_indexed.dropna(subset=['QuarterDate'])

# Create sorted list of unique quarterly dates
quarter_dates = sorted(df_indexed['QuarterDate'].unique())

# Use select_slider instead of slider for quarterly granularity
date_range = st.select_slider(
    "Select Date Range ",
    options=quarter_dates,
    value=(quarter_dates[0], quarter_dates[-1]),
    format_func=lambda x: x.strftime('%Y-%m')
)

# Filter by date range
df_filtered = df_indexed[(df_indexed['QuarterDate'] >= date_range[0]) & (df_indexed['QuarterDate'] <= date_range[1])]
df_filtered = df_filtered[df_filtered['Geography'].isin(selected_geos)]

# Step 1: Get the baseline (earliest date in slider)
baseline_date = df_filtered['QuarterDate'].min()

# Step 2: Build baseline by geography
baseline_df = df_filtered[df_filtered['QuarterDate'] == baseline_date][['Geography', 'Total']]
baseline_df = baseline_df.set_index('Geography')['Total']

# Step 3: Calculate % change relative to baseline
df_filtered['Indexed Change (%)'] = df_filtered.apply(
    lambda row: (row['Total'] / baseline_df[row['Geography']] - 1) * 100
    if row['Geography'] in baseline_df else None,
    axis=1
)

# fig, ax = plt.subplots(figsize=(12, 6))
# sns.lineplot(data=df_filtered, x="Quarter", y="Indexed Change (%)", hue="Geography", ax=ax)
# ax.axhline(0, color='gray', linestyle='--')
# ax.set_xlabel("Quarter")
# ax.set_ylabel("Indexed Change (%)")
# ax.set_title("Percentage Change in Volume of Homeless Applicants vs March 2021")
# plt.xticks(rotation=45)
# ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
# st.pyplot(fig)

df_filtered['QuarterStr'] = df_filtered['QuarterDate'].dt.strftime('%Y-%m')

# Ensure QuarterDate is used for consistent x-axis
df_indexed_chart = df_filtered.pivot(index='QuarterStr', columns='Geography', values='Indexed Change (%)')
st.line_chart(df_indexed_chart)

# Optional: data preview
with st.expander("Show Data Table"):
    st.dataframe(df_indexed, use_container_width=True)

######### Add space between sections ##########
st.markdown("<br><br>", unsafe_allow_html=True)
###############################################

melted_df = pd.melt(
    df,
    id_vars=['Geography', 'Quarter'],
    value_vars=age_band_columns,
    var_name='Age Band',
    value_name='Volume'
)

# Group and sum by Age Band and Quarter
n_vols_sum2 = melted_df.groupby(['Quarter', 'Age Band'])['Volume'].sum().reset_index()

# Convert Quarter to datetime for plotting
n_vols_sum2['QuarterDate'] = pd.to_datetime(n_vols_sum2['Quarter'], format='%Y-%m', errors='coerce')

# Create sorted list of unique quarterly dates
quarter_dates = sorted(n_vols_sum2['QuarterDate'].unique())

# Age band selector
age_bands = sorted(n_vols_sum2['Age Band'].unique())
selected_ages = st.multiselect("Select Age Bands to Compare", options=age_bands, default=list(age_bands))
n_vols_sum2 = n_vols_sum2[n_vols_sum2['Age Band'].isin(selected_ages)]

# Use select_slider instead of slider for quarterly granularity
date_range = st.select_slider(
    "Select Date Range    ",
    options=quarter_dates,
    value=(quarter_dates[0], quarter_dates[-1]),
    format_func=lambda x: x.strftime('%Y-%m')
)

# Filter by date range
n_vols_sum2 = n_vols_sum2[(n_vols_sum2['QuarterDate'] >= date_range[0]) & (n_vols_sum2['QuarterDate'] <= date_range[1])]

n_vols_sum2['QuarterStr'] = n_vols_sum2['QuarterDate'].dt.strftime('%Y-%m')

# Pivot so each Age Band is a column
pivot_df = n_vols_sum2.pivot(index='QuarterStr', columns='Age Band', values='Volume')
pivot_df = pivot_df.sort_index()  # Ensure chronological order

# Streamlit plot
st.subheader("Homelessness by Age Band")
st.line_chart(pivot_df)

######### Add space between sections ##########
st.markdown("<br><br>", unsafe_allow_html=True)
###############################################

selected_ages = st.multiselect("Select Age Bands to Compare ", options=age_bands, default=list(age_bands))
n_vols_sum2 = n_vols_sum2[n_vols_sum2['Age Band'].isin(selected_ages)]

# Use select_slider instead of slider for quarterly granularity
date_range = st.select_slider(
    "Select Date Range     ",
    options=quarter_dates,
    value=(quarter_dates[0], quarter_dates[-1]),
    format_func=lambda x: x.strftime('%Y-%m')
)

# Filter by date range
n_vols_sum2 = n_vols_sum2[(n_vols_sum2['QuarterDate'] >= date_range[0]) & (n_vols_sum2['QuarterDate'] <= date_range[1])]

n_vols_sum2['QuarterStr'] = n_vols_sum2['QuarterDate'].dt.strftime('%Y-%m')

df_filtered = n_vols_sum2[n_vols_sum2['Age Band'].isin(age_bands)]

# Step 1: Get the baseline (earliest date in slider)
baseline_date = df_filtered['QuarterDate'].min()

# Step 2: Build baseline by geography
baseline_df = df_filtered[df_filtered['QuarterDate'] == baseline_date][['Age Band', 'Volume']]
baseline_df = baseline_df.set_index('Age Band')['Volume']

# Step 3: Calculate % change relative to baseline
df_filtered['Indexed Change (%)'] = df_filtered.apply(
    lambda row: (row['Volume'] / baseline_df[row['Age Band']] - 1) * 100
    if row['Age Band'] in baseline_df else None,
    axis=1
)

df_filtered['QuarterStr'] = df_filtered['QuarterDate'].dt.strftime('%Y-%m')

# Ensure QuarterDate is used for consistent x-axis
df_indexed_chart = df_filtered.pivot(index='QuarterStr', columns='Age Band', values='Indexed Change (%)')
st.line_chart(df_indexed_chart)

# Optional: data preview
with st.expander("Show Data Table"):
    st.dataframe(df_indexed, use_container_width=True)

# # Download button
# csv = df_plot[['Geography', 'Quarter', 'Total']].to_csv(index=False)
# st.download_button(
#     label="📥 Download Filtered Data as CSV",
#     data=csv,
#     file_name="filtered_national_volumes.csv",
#     mime="text/csv"
# )
