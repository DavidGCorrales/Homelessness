import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# Set Streamlit page configuration
st.set_page_config(
    page_title="Applicant Volumes by Geography",
    layout="wide"
)

# Title
st.title("Volume of main applicants assessed as owed a prevention or relief duty by Geography Over Time")

# Load data
@st.cache_data
def load_data():
    path = "all_national_vols.csv"
    df = pd.read_csv(path)
    return df

df = load_data()

# Define age band columns
age_band_columns = [
    '16-17 Year Olds', '18-24 Year Olds', '25-34 Year Olds',
    '35-44 Year Olds', '45-54 Year Olds', '55-64 Year Olds',
    '65-74 Year Olds', '75+ Year Olds', 'Age Unknown'
]

# Calculate Total
df['Total'] = df[age_band_columns].sum(axis=1)

# Extract required columns
df_plot = df[['Geography', 'Quarter', 'Total']].copy()

# Quarter filter
df_plot['Quarter'] = pd.Categorical(df_plot['Quarter'], ordered=True,
                                     categories=sorted(df_plot['Quarter'].unique()))

quarter_list = list(df_plot['Quarter'].cat.categories)

col1, col2 = st.columns(2)
with col1:
    start_q = st.selectbox("Start Quarter", quarter_list, index=0)
with col2:
    end_q = st.selectbox("End Quarter", quarter_list, index=len(quarter_list) - 1)

# Filter by quarter range
filtered_quarters = quarter_list[
    quarter_list.index(start_q): quarter_list.index(end_q) + 1
]
df_plot = df_plot[df_plot['Quarter'].isin(filtered_quarters)]

# Geography filter
geos = df_plot['Geography'].unique()
selected_geos = st.multiselect("Select Geographies", options=geos, default=list(geos))
df_plot = df_plot[df_plot['Geography'].isin(selected_geos)]

# Sort
df_plot = df_plot.sort_values(by=["Geography", "Quarter"])

# Plotting
st.subheader("Line Chart of Total Applicants by Quarter")

fig, ax = plt.subplots(figsize=(12, 6))
sns.lineplot(data=df_plot, x="Quarter", y="Total", hue="Geography", ax=ax)
ax.set_xlabel("Quarter")
ax.set_ylabel("Volume")
ax.set_title("Volume of Main Applicants Assessed as Owed a Prevention or Relief Duty")
plt.xticks(rotation=45)
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
st.pyplot(fig)

# Summary table
st.subheader("Total Volume by Geography (Filtered Range)")
summary_table = df_plot.groupby('Geography')['Total'].sum().reset_index()
summary_table.columns = ['Geography', 'Total Volume']
st.dataframe(summary_table, use_container_width=True)

# Download filtered data
csv = df_plot.to_csv(index=False)
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name="filtered_national_volumes.csv",
    mime="text/csv"
)
