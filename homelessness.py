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
st.title("Volume of Main Applicants by Geography Over Time")

# Load the data
@st.cache_data
def load_data():
    path = "all_national_vols.csv"  # Ensure this file is in the same GitHub repo
    df = pd.read_csv(path)
    return df

# Load and process data
df = load_data()

# Define age band columns
age_band_columns = [
    '16-17 Year Olds', '18-24 Year Olds', '25-34 Year Olds', 
    '35-44 Year Olds', '45-54 Year Olds', '55-64 Year Olds', 
    '65-74 Year Olds', '75+ Year Olds', 'Age Unknown'
]

# Compute total volume
df['Total'] = df[age_band_columns].sum(axis=1)

# Filter only required columns
df_plot = df[['Geography', 'Quarter', 'Total']]

# Allow user to filter by geography
geos = df_plot['Geography'].unique()
selected_geos = st.multiselect("Select Geographies", options=geos, default=list(geos))
df_plot = df_plot[df_plot['Geography'].isin(selected_geos)]

# Sort by Quarter for correct plotting
df_plot = df_plot.sort_values(by="Quarter")

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
