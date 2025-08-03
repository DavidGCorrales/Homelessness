import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.express as px
from datetime import datetime

# Set Streamlit page configuration
st.set_page_config(
    page_title="Applicant Volumes by Geography",
    layout="wide"
)

st.title("Homelessness in England since March 2021")
st.markdown(
    "##### \"Homelessness\" refers to households initially assessed as threatened with homelessness (owed prevention duty) or homeless (owed relief duty).<br><br>"
    "Data sourced from official accredited statistics published on Gov.UK: https://www.gov.uk/government/statistical-data-sets/live-tables-on-homelessness",
    unsafe_allow_html=True
)
# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/DavidGCorrales/Homelessness/master/all_national_vols.csv"
    url_2 = "https://raw.githubusercontent.com/DavidGCorrales/Homelessness/master/all_national_pcnt.csv"
    df = pd.read_csv(url)
    df2 = pd.read_csv(url_2)
    return df, df2

df, df2 = load_data()

# Age band columns
age_band_columns = [
    '16-17 Year Olds', '18-24 Year Olds', '25-34 Year Olds',
    '35-44 Year Olds', '45-54 Year Olds', '55-64 Year Olds',
    '65-74 Year Olds', '75+ Year Olds', 'Age Unknown'
]

# Volume
df['Total'] = df[age_band_columns].sum(axis=1)

# Volume: Extract subset
df_plot = df[['Geography', 'Quarter', 'Total']].copy()

# Volume: Convert Quarter to datetime and clean data
df_plot['QuarterDate'] = pd.to_datetime(df_plot['Quarter'], format='%Y-%m', errors='coerce')
df_plot = df_plot.dropna(subset=['QuarterDate'])

# Volume: Create sorted list of unique quarterly dates
quarter_dates = sorted(df_plot['QuarterDate'].unique())

# Volume: Use select_slider instead of slider for quarterly granularity
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

# Volume: Sort
df_plot = df_plot.sort_values(by=["Geography", "QuarterDate"])

# Volume: Create string-based index for pivot
df_plot['QuarterStr'] = df_plot['QuarterDate'].dt.strftime('%Y-%m')

# # Volume: Pivot with string-based index
# df_volume_chart = df_plot.pivot(index='QuarterStr', columns='Geography', values='Total')
# # Percentage
# df_pcnt_chart = df2_filtered.pivot(index='QuarterStr', columns='Geography', values='Total')

# # Volume: Sort index
# df_volume_chart = df_volume_chart.sort_index()
# # Percentage
# df_pcnt_chart = df_pcnt_chart.sort_index()

# Volume: Create Plotly chart
fig = px.line(df_plot, x="QuarterDate", y="Total", color="Geography")

# Volume: Format x-axis ticks
fig.update_xaxes(
    tickmode='array',
    tickvals=df_plot['QuarterDate'].unique(),  # all unique dates
    tickformat="%b %y",  # e.g., Mar 21
    tickangle=-45
)

fig.update_layout(
    title=dict(
        text="Number of Homeless Households: London vs Rest of England",
        x=0.5,           # Center title horizontally
        xanchor='center' # Anchor title at center
    )  # Center the title horizontally
)

#################################
# Indexed Percentage Change Chart
#################################

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

# Convert Quarter to datetime and clean data
df_indexed['QuarterDate'] = pd.to_datetime(df_indexed['Quarter'], format='%Y-%m', errors='coerce')
df_indexed = df_indexed.dropna(subset=['QuarterDate'])

# Create sorted list of unique quarterly dates
quarter_dates = sorted(df_indexed['QuarterDate'].unique())

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

df_filtered['QuarterStr'] = df_filtered['QuarterDate'].dt.strftime('%Y-%m')

# Create Plotly chart
fig2 = px.line(
    df_filtered,
    x='QuarterStr',
    y='Indexed Change (%)',
    color='Geography'
)

# Format x-axis and center the title
fig2.update_xaxes(
    tickmode='array',
    tickvals=df_filtered['QuarterStr'].unique(),  # all unique dates
    tickformat="%b %y",  # Format as "Apr 21"
    tickangle=-45
)
fig2.update_layout(
    title=dict(
        text="Percentage Change in Homeless Households: London vs Rest of England",
        x=0.5,           # Center title horizontally
        xanchor='center' # Anchor title at center
    )
)

col1, col2 = st.columns(2)
# Volume
with col1:
    st.plotly_chart(fig, use_container_width=True)
# pcnt
with col2:
    st.plotly_chart(fig2, use_container_width=True)

# # Optional: data preview
# with st.expander("Show Data Table"):
#     st.dataframe(df_indexed, use_container_width=True)

######### Add space between sections ##########
st.markdown("<br><br>", unsafe_allow_html=True)
###############################################

# Volume
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
age_bands_sorted = sorted(n_vols_sum2[n_vols_sum2['Age Band'] != 'Age Unknown']['Age Band'].unique())
selected_ages = st.multiselect("Select Age Bands to Compare", options=age_bands_sorted, default=list(age_bands_sorted))
filtered_ages = n_vols_sum2[n_vols_sum2['Age Band'].isin(selected_ages)]

# Use select_slider instead of slider for quarterly granularity
date_range2 = st.select_slider(
    "Select Date Range    ",
    options=quarter_dates,
    value=(quarter_dates[0], quarter_dates[-1]),
    format_func=lambda x: x.strftime('%Y-%m')
)

# Volume: Filter by date range
n_vols_sum_filtered = filtered_ages[(filtered_ages['QuarterDate'] >= date_range2[0]) & (filtered_ages['QuarterDate'] <= date_range2[1])]

# Volume: Create string-based index for pivot
n_vols_sum_filtered['QuarterStr'] = n_vols_sum_filtered['QuarterDate'].dt.strftime('%Y-%m')


# Create Plotly chart
fig = px.line(
    n_vols_sum_filtered,
    x='QuarterStr',
    y='Volume',
    color='Age Band'
)

# Format x-axis and center the title
fig.update_xaxes(
    tickmode='array',
    tickvals=n_vols_sum_filtered['QuarterStr'].unique(),  # all unique dates
    tickformat="%b %y",  # Format as "Apr 21"
    tickangle=-45
)
fig.update_layout(
    title=dict(
        text="Number of Homeless Households by Age Band",
        x=0.5,           # Center title horizontally
        xanchor='center' # Anchor title at center
    )
)

#################################
# Indexed Percentage Change Chart
#################################

# Filter by date range
n_vols_sum_filtered = n_vols_sum_filtered[(n_vols_sum_filtered['QuarterDate'] >= date_range[0]) & (n_vols_sum_filtered['QuarterDate'] <= date_range[1])]

n_vols_sum_filtered['QuarterStr'] = n_vols_sum_filtered['QuarterDate'].dt.strftime('%Y-%m')

pcnt_age_band = n_vols_sum_filtered[n_vols_sum_filtered['Age Band'].isin(age_bands_sorted)]

# Step 1: Get the baseline (earliest date in slider)
baseline_date = pcnt_age_band['QuarterDate'].min()

# Step 2: Build baseline by geography
baseline_df = pcnt_age_band[pcnt_age_band['QuarterDate'] == baseline_date][['Age Band', 'Volume']]
baseline_df = baseline_df.set_index('Age Band')['Volume']

# Step 3: Calculate % change relative to baseline
pcnt_age_band['Indexed Change (%)'] = pcnt_age_band.apply(
    lambda row: (row['Volume'] / baseline_df[row['Age Band']] - 1) * 100
    if row['Age Band'] in baseline_df else None,
    axis=1
)

pcnt_age_band['QuarterStr'] = pcnt_age_band['QuarterDate'].dt.strftime('%Y-%m')

# Create Plotly chart
fig2 = px.line(
    pcnt_age_band,
    x='QuarterStr',
    y='Indexed Change (%)',
    color='Age Band'
)

# Format x-axis and center the title
fig2.update_xaxes(
    tickmode='array',
    tickvals=pcnt_age_band['QuarterStr'].unique(),  # all unique dates
    tickformat="%b %y",  # Format as "Apr 21"
    tickangle=-45
)
fig2.update_layout(
    title=dict(
        text="Percentage Change in Homeless Households by Age Band",
        x=0.5,           # Center title horizontally
        xanchor='center' # Anchor title at center
    )
                  )

# Display in Plotly
col1, col2 = st.columns(2)
# Volume
with col1:
    st.plotly_chart(fig, use_container_width=True)
# pcnt
with col2:
    st.plotly_chart(fig2, use_container_width=True)

# # Streamlit plot
# st.subheader("Homelessness by Age Band")
# st.line_chart(pivot_df)

######### Add space between sections ##########
st.markdown("<br><br>", unsafe_allow_html=True)
###############################################

pcnt_age_band2 = n_vols_sum2.copy()
pcnt_age_band2['Age Band 2'] = np.where(pcnt_age_band2['Age Band'].isin(['16-17 Year Olds', '18-24 Year Olds', '25-34 Year Olds']), "16-34 Year Olds",
np.where(pcnt_age_band2['Age Band'].isin(['35-44 Year Olds', '45-54 Year Olds']), '35-54 Year Olds',
         np.where(pcnt_age_band2['Age Band'].isin(['55-64 Year Olds','65-74 Year Olds','75+ Year Olds']), "55+ Year Olds", 'Age Unknown')))

pcnt_age_band2['QuarterStr'] = pcnt_age_band2['QuarterDate'].dt.strftime('%Y-%m')

pcnt_age_band2 = pcnt_age_band2.groupby(['QuarterStr', 'QuarterDate', 'Age Band 2'])['Volume'].sum().reset_index()

# Create sorted list of unique quarterly dates
quarter_dates = sorted(pcnt_age_band2['QuarterDate'].unique())

# Age band selector
age_bands2 = sorted(pcnt_age_band2[pcnt_age_band2['Age Band 2'] != 'Age Unknown']['Age Band 2'].unique())
selected_ages2 = st.multiselect("Select Age Bands to Compare", options=age_bands2, default=list(age_bands2))
pcnt_age_band2 = pcnt_age_band2[pcnt_age_band2['Age Band 2'].isin(selected_ages2)]

# Use select_slider instead of slider for quarterly granularity
date_range3 = st.select_slider(
    "Select Date Range      ",
    options=quarter_dates,
    value=(quarter_dates[0], quarter_dates[-1]),
    format_func=lambda x: x.strftime('%Y-%m')
)

# Filter by date range
pcnt_age_band2 = pcnt_age_band2[(pcnt_age_band2['QuarterDate'] >= date_range3[0]) & (pcnt_age_band2['QuarterDate'] <= date_range3[1])]

pcnt_age_band2['QuarterStr'] = pcnt_age_band2['QuarterDate'].dt.strftime('%Y-%m')

pcnt_age_band2 = pcnt_age_band2[pcnt_age_band2['Age Band 2'].isin(age_bands2)]

# Step 1: Get the baseline (earliest date in slider)
baseline_date = pcnt_age_band2['QuarterDate'].min()

# Step 2: Build baseline by geography
baseline_df = pcnt_age_band2[pcnt_age_band2['QuarterDate'] == baseline_date][['Age Band 2', 'Volume']]
baseline_df = baseline_df.set_index('Age Band 2')['Volume']

# Step 3: Calculate % change relative to baseline
pcnt_age_band2['Indexed Change (%)'] = pcnt_age_band2.apply(
    lambda row: (row['Volume'] / baseline_df[row['Age Band 2']] - 1) * 100
    if row['Age Band 2'] in baseline_df else None,
    axis=1
)

pcnt_age_band2['QuarterStr'] = pcnt_age_band2['QuarterDate'].dt.strftime('%Y-%m')

# Create Plotly chart
fig = px.line(
    pcnt_age_band2,
    x='QuarterStr',
    y='Volume',
    color='Age Band 2'
)

# Format x-axis and center the title
fig.update_xaxes(
    tickmode='array',
    tickvals=pcnt_age_band2['QuarterStr'].unique(),  # all unique dates
    tickformat="%b %y",  # Format as "Apr 21"
    tickangle=-45
)
fig.update_layout(
    title=dict(
        text="Number of Homeless Households by Grouped Age Bands",
        x=0.5,           # Center title horizontally
        xanchor='center' # Anchor title at center
    )
                  )

# Create Plotly chart
fig2 = px.line(
    pcnt_age_band2,
    x='QuarterStr',
    y='Indexed Change (%)',
    color='Age Band 2'
)

# Format x-axis and center the title
fig2.update_xaxes(
    tickmode='array',
    tickvals=pcnt_age_band2['QuarterStr'].unique(),  # all unique dates
    tickformat="%b %y",  # Format as "Apr 21"
    tickangle=-45
)
fig2.update_layout(
    title=dict(
        text="Percentage Change in Homeless Households by Grouped Age Bands",
        x=0.5,           # Center title horizontally
        xanchor='center' # Anchor title at center
    )
                  )

# Display in Plotly
col1, col2 = st.columns(2)
# Volume
with col1:
    st.plotly_chart(fig, use_container_width=True)
# pcnt
with col2:
    st.plotly_chart(fig2, use_container_width=True)

######### Add space between sections ##########
st.markdown("<br><br>", unsafe_allow_html=True)
############################################### 

# Percent
melted_df2 = pd.melt(
    df2,
    id_vars=['Geography', 'Quarter'],
    value_vars=age_band_columns,
    var_name='Age Band',
    value_name='Percent'
)

melted_df2['Age Band 2'] = np.where(melted_df2['Age Band'].isin(['16-17 Year Olds', '18-24 Year Olds', '25-34 Year Olds']), "16-34 Year Olds",
np.where(melted_df2['Age Band'].isin(['35-44 Year Olds', '45-54 Year Olds']), '35-54 Year Olds',
         np.where(melted_df2['Age Band'].isin(['55-64 Year Olds','65-74 Year Olds','75+ Year Olds']), "55+ Year Olds", 'Age Unknown')))

# Group and sum by Age Band and Quarter
n_vols_sum2 = melted_df2.groupby(['Quarter', 'Age Band 2'])['Percent'].sum().reset_index()

# Convert Quarter to datetime for plotting
n_vols_sum2['QuarterDate'] = pd.to_datetime(n_vols_sum2['Quarter'], format='%Y-%m', errors='coerce')

# Create sorted list of unique quarterly dates
quarter_dates = sorted(n_vols_sum2['QuarterDate'].unique())

# Age band selector
age_bands = sorted(n_vols_sum2[n_vols_sum2['Age Band 2'] != 'Age Unknown']['Age Band 2'].unique())
selected_ages = st.multiselect("Select Age Bands to Compare ", options=age_bands, default=list(age_bands))
n_vols_sum2 = n_vols_sum2[n_vols_sum2['Age Band 2'].isin(selected_ages)]

# Use select_slider instead of slider for quarterly granularity
date_range = st.select_slider(
    "Select Date Range     ",
    options=quarter_dates,
    value=(quarter_dates[0], quarter_dates[-1]),
    format_func=lambda x: x.strftime('%Y-%m')
)

# Volume: Filter by date range
n_vols_sum2 = n_vols_sum2[(n_vols_sum2['QuarterDate'] >= date_range[0]) & (n_vols_sum2['QuarterDate'] <= date_range[1])]

# Volume: Create string-based index for pivot
n_vols_sum2['QuarterStr'] = n_vols_sum2['QuarterDate'].dt.strftime('%Y-%m')

# pcnt
n_pcnt_sum2 = melted_df2.groupby(['Quarter', 'Age Band 2'])['Percent'].sum().reset_index()

# pcnt
n_pcnt_sum2['QuarterDate'] = pd.to_datetime(n_pcnt_sum2['Quarter'], format='%Y-%m', errors='coerce')

n_pcnt_sum2 = n_pcnt_sum2[n_pcnt_sum2['Age Band 2'].isin(selected_ages)]

# pcnt
n_pcnt_sum2 = n_pcnt_sum2[(n_pcnt_sum2['QuarterDate'] >= date_range[0]) & (n_pcnt_sum2['QuarterDate'] <= date_range[1])]

# pcnt
n_pcnt_sum2['QuarterStr'] = n_pcnt_sum2['QuarterDate'].dt.strftime('%Y-%m')

# Create Plotly 100% stacked bar chart
fig = px.bar(
    n_pcnt_sum2,
    x='QuarterStr',
    y='Percent',
    color='Age Band 2'
)

# Format x-axis
fig.update_xaxes(
    tickmode='array',
    tickvals=n_pcnt_sum2['QuarterStr'].unique(),
    tickformat="%b %y",
    tickangle=-45
)

# Layout updates
fig.update_layout(
    title=dict(
        text="Distribution of Homeless Households by Age Band",
        x=0.5,
        xanchor='center'
    ),
    barmode='relative',
    barnorm='percent',  # This normalizes each bar to 100%
    yaxis_title='Percentage'
)

st.plotly_chart(fig, use_container_width=True)

# df_filtered2

# # Download button
# csv = df_plot[['Geography', 'Quarter', 'Total']].to_csv(index=False)
# st.download_button(
#     label="📥 Download Filtered Data as CSV",
#     data=csv,
#     file_name="filtered_national_volumes.csv",
#     mime="text/csv"
# )
