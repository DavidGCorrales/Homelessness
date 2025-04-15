#!/usr/bin/env python
# coding: utf-8

# ### Homelessness by Age Group in the UK - Jan to Sep 2024

# In[1]:


# import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import BytesIO
import re
from datetime import datetime
from IPython.display import display

import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

pd.set_option('display.max_columns', None)

# !pip install odfpy


# #### London vs England; Age Band; Last 3 Quarters

# In[2]:


def process_quarter_data(url):
    """
    Function to process the data for a specific URL containing the Excel data.

    Args:
    - url: str, the URL of the Excel file.

    Returns:
    - national_vols: processed DataFrame for national volumes.
    - national_pcnt: processed DataFrame for national percentages.
    """
    
    def extract_quarter_from_url(url):
        """
        Extracts the first consecutive 6 digits (YYYYMM) from the URL.

        Args:
        - url: str, the URL of the Excel file.

        Returns:
        - quarter: str, the 6-digit date (YYYYMM).
        """
        # Extract the filename from the URL
        filename = url.split('/')[-1]

        # Use regex to find the first consecutive 6 digits
        match = re.search(r'\d{6}', filename)

        if match:
            return match.group(0)  # Return the first match of 6 digits
        else:
            return None  # Return None if no match is found
    
    quarter = extract_quarter_from_url(url)
    quarter = datetime.strptime(quarter, "%Y%m").strftime("%Y-%m")
    
    if url.endswith(".xlsx"):
        engine = "openpyxl"
    elif url.endswith(".ods"):
        engine = "odf"
    else:
        raise ValueError("Unsupported file format. Only .xlsx and .ods are supported.")
    
    # Download the file
    response = requests.get(url)
    file_content = BytesIO(response.content)
    
    xl = pd.ExcelFile(file_content, engine=engine)
    
    # Check which sheet exists
    sheet_name = "A6" if "A6" in xl.sheet_names else "A6_"
    
    # Read the Excel file into a DataFrame
    df = pd.read_excel(file_content, engine=engine, sheet_name=sheet_name)  

    ### Extract the required rows and columns ###
    
    # National: London vs Rest of England #
    national = df.iloc[3:6, 0:24]
    # Regional Split: North East / North West etc. #
    regional = df.iloc[7:16, 0:24]
    # City Split #
    city = df.iloc[17:350, 0:24]
    
    national = national.drop(national.columns[[0, 2, 3]], axis=1)
    regional = regional.drop(regional.columns[[0, 2, 3]], axis=1)
    city = city.drop(city.columns[[0, 2, 3]], axis=1)

    # National Volumes DataFrame
    national_vols = national[['Unnamed: 1', 'Unnamed: 6', 'Unnamed: 8', 'Unnamed: 10', 'Unnamed: 12',
                             'Unnamed: 14', 'Unnamed: 16', 'Unnamed: 18', 'Unnamed: 20', 'Unnamed: 22']]
    # Regional Volumes DataFrame
    regional_vols = regional[['Unnamed: 1', 'Unnamed: 6', 'Unnamed: 8', 'Unnamed: 10', 'Unnamed: 12',
                             'Unnamed: 14', 'Unnamed: 16', 'Unnamed: 18', 'Unnamed: 20', 'Unnamed: 22']]
    # City Volumes DataFrame
    city_vols = city[['Unnamed: 1', 'Unnamed: 6', 'Unnamed: 8', 'Unnamed: 10', 'Unnamed: 12',
                             'Unnamed: 14', 'Unnamed: 16', 'Unnamed: 18', 'Unnamed: 20', 'Unnamed: 22']]
    
    vol_rename = {
        'Unnamed: 1': 'Geography',
        'Unnamed: 6': '16-17 Year Olds',
        'Unnamed: 8': '18-24 Year Olds',
        'Unnamed: 10': '25-34 Year Olds',
        'Unnamed: 12': '35-44 Year Olds',    
        'Unnamed: 14': '45-54 Year Olds',
        'Unnamed: 16': '55-64 Year Olds',
        'Unnamed: 18': '65-74 Year Olds',
        'Unnamed: 20': '75+ Year Olds',
        'Unnamed: 22': 'Age Unknown',
    }

    national_vols = national_vols.rename(columns=vol_rename).reset_index(drop=True)
    regional_vols = regional_vols.rename(columns=vol_rename).reset_index(drop=True)
    city_vols = city_vols.rename(columns=vol_rename).reset_index(drop=True)
    
    # Add 'Quarter' column to national_vols
    national_vols['Quarter'] = quarter
    # Add 'Quarter' column to regionall_vols
    regional_vols['Quarter'] = quarter
    # Add 'Quarter' column to city_vols
    city_vols['Quarter'] = quarter

    # National Percentages DataFrame
    national_pcnt = national[['Unnamed: 1', 'Unnamed: 7', 'Unnamed: 9', 'Unnamed: 11', 'Unnamed: 13',
                              'Unnamed: 15', 'Unnamed: 17', 'Unnamed: 19', 'Unnamed: 21', 'Unnamed: 23']]

    # Regional Percentages DataFrame
    regional_pcnt = regional[['Unnamed: 1', 'Unnamed: 7', 'Unnamed: 9', 'Unnamed: 11', 'Unnamed: 13',
                              'Unnamed: 15', 'Unnamed: 17', 'Unnamed: 19', 'Unnamed: 21', 'Unnamed: 23']]
    
    # City Percentages DataFrame
    city_pcnt = city[['Unnamed: 1', 'Unnamed: 7', 'Unnamed: 9', 'Unnamed: 11', 'Unnamed: 13',
                              'Unnamed: 15', 'Unnamed: 17', 'Unnamed: 19', 'Unnamed: 21', 'Unnamed: 23']]                          
    pcnt_rename = {
        'Unnamed: 1': 'Geography',
        'Unnamed: 7': '16-17 Year Olds',
        'Unnamed: 9': '18-24 Year Olds',
        'Unnamed: 11': '25-34 Year Olds',
        'Unnamed: 13': '35-44 Year Olds',    
        'Unnamed: 15': '45-54 Year Olds',
        'Unnamed: 17': '55-64 Year Olds',
        'Unnamed: 19': '65-74 Year Olds',
        'Unnamed: 21': '75+ Year Olds',
        'Unnamed: 23': 'Age Unknown',
    }

    national_pcnt = national_pcnt.rename(columns=pcnt_rename).reset_index(drop=True)
    regional_pcnt = regional_pcnt.rename(columns=pcnt_rename).reset_index(drop=True)
    city_pcnt = city_pcnt.rename(columns=pcnt_rename).reset_index(drop=True)
    
    # Add 'Quarter' column to national_pcnt
    national_pcnt['Quarter'] = quarter
    # Add 'Quarter' column to regional_pcnt
    regional_pcnt['Quarter'] = quarter
    # Add 'Quarter' column to city_pcnt
    city_pcnt['Quarter'] = quarter
    
    return national_vols, national_pcnt, regional_vols, regional_pcnt, city_vols, city_pcnt


# In[3]:


# List of URLs for different quarters
urls = [
    "https://assets.publishing.service.gov.uk/media/67bdd57b89b4a58925ac6d17/Detailed_LA_202409.xlsx",          # Sep 2024
    "https://assets.publishing.service.gov.uk/media/67bdd5255d6b30e896ac6d1e/Detailed_LA_202406_revised.xlsx",  # Jun 2024
    "https://assets.publishing.service.gov.uk/media/6747279da72d7eb7f348c04c/Detailed_LA_202403.xlsx",          # Mar 2024
    "https://assets.publishing.service.gov.uk/media/66b331c5a3c2a28abb50ddfd/Detailed_LA_202312_Revised_No_Dropdowns.ods",
    "https://assets.publishing.service.gov.uk/media/662f6dd89e82181baa98a8f5/Detailed_LA_202309_revised.ods",
    "https://assets.publishing.service.gov.uk/media/65df3f24cf7eb16adff57f55/Detailed_LA_202306_revised.ods",
    "https://assets.publishing.service.gov.uk/media/65804b3f83ba38000de1b77a/Detailed_LA_202303_Revised.ods",
    "https://assets.publishing.service.gov.uk/media/64be88f41e10bf000e17cd3a/Detailed_LA_202212_revised.ods",
    "https://assets.publishing.service.gov.uk/media/645a5900479612000cc29274/V2_Detailed_LA_202209_revised.ods",
    "https://assets.publishing.service.gov.uk/media/6400d30c8fa8f527ff6b42db/Detailed_LA_202206_revised_Updated.ods",
    "https://assets.publishing.service.gov.uk/media/637e39e0d3bf7f153b8b321c/Detailed_LA_202203_revised.ods",
    "https://assets.publishing.service.gov.uk/media/62e14c208fa8f564a21dcd8a/Detailed_LA_202112_revised.ods",
    "https://assets.publishing.service.gov.uk/media/62694963d3bf7f0e7121ae4b/DetailedLA_202109_revised.ods",
    "https://assets.publishing.service.gov.uk/media/676433a63229e84d9bbde8f3/DetailedLA_202106_-_Revised_fixed.ods",
    "https://assets.publishing.service.gov.uk/media/61796748e90e0719833464ff/DetailedLA_202103_Revised.ods",
    
#     Odd Looking Data (Data much lower across the board pre Mar-2021 - maybe reporting changed?)
#     "https://assets.publishing.service.gov.uk/media/60fae728d3bf7f04599e2170/DetailedLA_202012_Revised_updated.ods",
#     "https://assets.publishing.service.gov.uk/media/60803c88e90e076aab2a0531/DetailedLA_202009_Revised.ods",
#     "https://assets.publishing.service.gov.uk/media/65e6e8372f2b3be8b07cd78e/Detailed_LA_202006_Revised_Fixed.ods",
    
#     Different Spreadsheet Layout
#     "https://assets.publishing.service.gov.uk/media/65e6e8207bc329020bb8c266/Detailed_LA_202003_Revised_Fixed.ods",
#     "https://assets.publishing.service.gov.uk/media/5f749aaad3bf7f2868b3ffa2/DetailedLA_201912_revised.ods",
#     "https://assets.publishing.service.gov.uk/media/5ec53218e90e0754d1dedf18/DetailedLA_201909_revised.xlsx",
#     "https://assets.publishing.service.gov.uk/media/5e7239c1e90e070ad0e62a28/DetailedLA_201906_revised.xlsx"
]

# Loop through the URLs and apply the function
all_national_vols_1 = []
all_national_pcnt_1 = []

all_regional_vols_1 = []
all_regional_pcnt_1 = []

all_city_vols_1 = []
all_city_pcnt_1 = []

for url in urls:
    national_vols, national_pcnt, regional_vols, regional_pcnt, city_vols, city_pcnt = process_quarter_data(url)
    all_national_vols_1.append(national_vols)
    all_national_pcnt_1.append(national_pcnt)
    all_regional_vols_1.append(regional_vols)
    all_regional_pcnt_1.append(regional_pcnt)
    all_city_vols_1.append(city_vols)
    all_city_pcnt_1.append(city_pcnt)


# In[64]:


all_national_vols = pd.concat(all_national_vols_1)
all_national_pcnt = pd.concat(all_national_pcnt_1)
display(all_national_vols),
display(all_national_pcnt)


# In[65]:


age_band_columns = ['16-17 Year Olds', '18-24 Year Olds', '25-34 Year Olds', 
                    '35-44 Year Olds', '45-54 Year Olds', '55-64 Year Olds', 
                    '65-74 Year Olds', '75+ Year Olds', 'Age Unknown']

all_national_vols['Total'] = all_national_vols[age_band_columns].sum(axis=1)

all_national_vols_x = all_national_vols[['Geography', 'Quarter', 'Total']]
# [all_national_vols['Geography'] != "ENGLAND"]
all_national_vols_x


# In[66]:


def plot_age_band_proportion(df, x, y, hue, title="_________"):
    """
    Creates a bar plot grouped by Geography and Quarter.

    Parameters:
    df (DataFrame): The DataFrame containing 'Quarter', 'Average', and 'Age Band' columns.
    title (str): The title of the plot (default: "Proportion by Age Band and Quarter").
    """
    
    # Set up the plotting style
    sns.set(style="whitegrid")

    # Create the figure
    plt.figure(figsize=(10, 6))

    # Create the bar plot
    sns.lineplot(x=x, y=y, hue=hue, data=df, err_style=None)

    # Add labels and title
    plt.title(title)
    plt.xlabel('Quarter')
    plt.ylabel('Average Proportion')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)

    # Move the legend outside the plot
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    # Show the plot
    plt.show()


# In[67]:


plot_age_band_proportion(all_national_vols_x.sort_values(by="Quarter"), x='Quarter', y='Total', hue='Geography')


# In[8]:


melted_df = pd.melt(all_national_vols, id_vars=['Geography', 'Quarter'], 
                    value_vars=['16-17 Year Olds', '18-24 Year Olds', '25-34 Year Olds', 
                                '35-44 Year Olds', '45-54 Year Olds', '55-64 Year Olds',
                                '65-74 Year Olds', '75+ Year Olds', 'Age Unknown'], 
                    var_name='Age Band', value_name='Volume')
melted_df


# In[9]:


# Step 1: Ensure data is sorted
all_national_vols = all_national_vols.sort_values(['Geography', 'Quarter'])

# Step 2: Get the first quarter's total for each Geography
baseline = all_national_vols[all_national_vols['Quarter'] == '2021-03'].set_index('Geography')['Total']

# Step 3: Calculate indexed percentage change
all_national_vols['Indexed Change (%)'] = all_national_vols.apply(
    lambda row: (row['Total'] / baseline[row['Geography']] - 1) * 100 if row['Geography'] in baseline else None,
    axis=1
)

# Step 4: Display result
all_national_vols_indexed = all_national_vols[['Geography', 'Quarter', 'Total', 'Indexed Change (%)']]
all_national_vols_indexed


# In[10]:


plot_age_band_proportion(all_national_vols_indexed, x='Quarter',  y='Indexed Change (%)', hue='Geography')


# In[11]:


n_vols_sum2 = melted_df.groupby(['Age Band', 'Quarter'])['Volume'].sum().reset_index()
n_vols_sum2


# In[12]:


plot_age_band_proportion(n_vols_sum2, x='Quarter', y='Volume', hue='Age Band')


# In[13]:


# Step 1: Ensure data is sorted
all_national_vols = n_vols_sum2.sort_values(['Age Band', 'Quarter'])

# Step 2: Get the first quarter's total for each Geography
baseline = all_national_vols[all_national_vols['Quarter'] == '2021-03'].set_index('Age Band')['Volume']

# Step 3: Calculate indexed percentage change
all_national_vols['Indexed Change (%)'] = all_national_vols.apply(
    lambda row: (row['Volume'] / baseline[row['Age Band']] - 1) * 100 if row['Age Band'] in baseline else None,
    axis=1
)

# Step 4: Display result
all_national_vols_indexed = all_national_vols[['Age Band', 'Quarter', 'Volume', 'Indexed Change (%)']][all_national_vols['Age Band'] != "Age Unknown"]
all_national_vols_indexed


# In[14]:


plot_age_band_proportion(all_national_vols_indexed, x='Quarter', y='Indexed Change (%)', hue='Age Band')


# In[15]:


all_national_vols_indexed['Age Band 2'] = np.where(all_national_vols_indexed['Age Band'].isin(['16-17 Year Olds', '18-24 Year Olds', '25-34 Year Olds']), "16-34 Year Olds", 
np.where(all_national_vols_indexed['Age Band'].isin(['35-44 Year Olds', '45-54 Year Olds']), '35-54 Year Olds',
         np.where(all_national_vols_indexed['Age Band'].isin(['55-64 Year Olds','65-74 Year Olds','75+ Year Olds']), "55+ Year Olds", 'Age Unknown')))

all_national_vols_indexed


# In[16]:


plot_age_band_proportion(all_national_vols_indexed, x='Quarter', y='Indexed Change (%)', hue='Age Band 2')


# In[17]:


all_national_pcnt


# In[18]:


melted_df = pd.melt(all_national_pcnt, id_vars=['Geography', 'Quarter'], 
                    value_vars=['16-17 Year Olds', '18-24 Year Olds', '25-34 Year Olds', 
                                '35-44 Year Olds', '45-54 Year Olds', '55-64 Year Olds',
                                '65-74 Year Olds', '75+ Year Olds', 'Age Unknown'], 
                    var_name='Age Band', value_name='Average')
melted_df


# In[19]:


n_pcnt_mean = melted_df.groupby(['Age Band', 'Quarter'])['Average'].mean().reset_index()
n_pcnt_mean


# In[20]:


plot_age_band_proportion(n_pcnt_mean, x='Quarter', y='Average', hue='Age Band')


# ## Regional Split

# In[21]:


all_regional_vols = pd.concat(all_regional_vols_1)
all_regional_pcnt = pd.concat(all_regional_pcnt_1)
display(all_regional_vols),
display(all_regional_pcnt)


# In[22]:


all_regional_vols['Total'] = all_regional_vols[age_band_columns].sum(axis=1)

all_regional_vols[['Geography', 'Quarter', 'Total']]


# In[23]:


plot_age_band_proportion(all_regional_vols.sort_values(by="Quarter"), x='Quarter', y='Total', hue='Geography')


# In[24]:


Region = "North East"
plot_age_band_proportion(all_regional_vols[all_regional_vols['Geography'] == Region].sort_values(by="Quarter"), x='Quarter', y='Total', hue='Geography')


# In[25]:


# Step 1: Ensure data is sorted
all_regional_vols = all_regional_vols.sort_values(['Geography', 'Quarter'])

# # Step 2: Get the first quarter's total for each Geography
baseline = all_regional_vols[all_regional_vols['Quarter'] == '2021-03'].set_index('Geography')['Total']

# # Step 3: Calculate indexed percentage change
all_regional_vols['Indexed Change (%)'] = all_regional_vols.apply(
    lambda row: (row['Total'] / baseline[row['Geography']] - 1) * 100 if row['Geography'] in baseline else None,
    axis=1
)

# # Step 4: Display result
all_regional_vols_indexed = all_regional_vols[['Geography', 'Quarter', 'Total', 'Indexed Change (%)']].sort_values(['Geography', 'Quarter'])
all_regional_vols_indexed


# In[26]:


plot_age_band_proportion(all_regional_vols_indexed, x='Quarter', y='Indexed Change (%)', hue='Geography')


# In[27]:


all_regional_vols_indexed['Geography2'] = np.where(all_regional_vols_indexed['Geography'].isin(['East Midlands', 'West Midlands']), "The Midlands",
                                                   np.where(all_regional_vols_indexed['Geography'].isin(['East of England', 'London']), "The East",
                                                            np.where(all_regional_vols_indexed['Geography'].isin(['North West', 'North East', 'Yorkshire and The Humber']), "The North", "The South"
                                                                    )))
all_regional_vols_indexed


# In[28]:


plot_age_band_proportion(all_regional_vols_indexed, x='Quarter', y='Indexed Change (%)', hue='Geography2')


# In[29]:


melted_df = pd.melt(all_regional_vols, id_vars=['Geography', 'Quarter'], 
                    value_vars=['16-17 Year Olds', '18-24 Year Olds', '25-34 Year Olds', 
                                '35-44 Year Olds', '45-54 Year Olds', '55-64 Year Olds',
                                '65-74 Year Olds', '75+ Year Olds', 'Age Unknown'], 
                    var_name='Age Band', value_name='Average')
melted_df = melted_df.sort_values(by='Quarter')
melted_df


# In[30]:


melted_df['Age Band 2'] = np.where(melted_df['Age Band'].isin(['16-17 Year Olds', '18-24 Year Olds', '25-34 Year Olds']), "16-34 Year Olds", 
np.where(melted_df['Age Band'].isin(['35-44 Year Olds', '45-54 Year Olds']), "35-54 Year Olds", 
         np.where(melted_df['Age Band'].isin(['55-64 Year Olds','65-74 Year Olds','75+ Year Olds']), "55+ Year Olds", 'Age Unknown')))

melted_df = melted_df.groupby(['Geography', 'Quarter', 'Age Band 2'])['Average'].sum().reset_index()
melted_df


# In[31]:


# Step 1: Ensure data is sorted
melted_df = melted_df.sort_values(['Geography', 'Quarter', 'Age Band 2'])

# # Step 2: Get the first quarter's total for each Geography
baseline = (
    melted_df[melted_df['Quarter'] == '2021-03']
    .groupby(['Geography', 'Age Band 2'])['Average']
    .sum()
)
baseline

# # Step 3: Calculate indexed percentage change
melted_df['Indexed Change (%)'] = melted_df.apply(
    lambda row: (
        ((row['Average'] / baseline.get((row['Geography'], row['Age Band 2']), 1)) - 1) * 100
        if (row['Geography'], row['Age Band 2']) in baseline.index and baseline.get((row['Geography'], row['Age Band 2'])) != 0
        else None
    ),
    axis=1
)

# # Step 4: Display result
melted_df_indexed = melted_df[['Geography', 'Quarter', 'Age Band 2', 'Average', 'Indexed Change (%)']].sort_values(['Geography', 'Quarter', 'Age Band 2'])
melted_df_indexed


# In[32]:


regions = melted_df_indexed['Geography'].unique()

# Loop through each region and create a plot
for Region in regions:
    plot_age_band_proportion(melted_df_indexed[(melted_df_indexed['Geography'] == Region) & (melted_df['Age Band 2'] != "Age Unknown")], x='Quarter', y='Indexed Change (%)', hue='Age Band 2', title=f"Proportion by Age Band and Quarter for {Region}")


# In[33]:


all_city_vols = pd.concat(all_city_vols_1)
all_city_pcnt = pd.concat(all_city_pcnt_1)
display(all_city_vols),
display(all_city_pcnt)


# In[34]:


city_filter = pd.crosstab(all_city_pcnt['Geography'], all_city_pcnt['Quarter'])
city_filter['total_months'] = city_filter.sum(axis=1)
city_filter = city_filter.sort_values(by='total_months', ascending=True).reset_index()
city_filter


# In[35]:


city_filter = city_filter[city_filter['total_months'] > 13]
# city_filter = city_filter.iloc[1:].reset_index(drop=True)
city_filter


# In[36]:


city_filter['geo_length'] = city_filter['Geography'].str.len()
city_filter_ = city_filter.sort_values(by='geo_length', ascending=False)
city_filter_ = city_filter[city_filter['geo_length'] < 40]
city_filter_


# In[37]:


final_city = all_city_pcnt[all_city_pcnt['Geography'].isin(city_filter_['Geography'])]
# final_city.sort_values(by='16-17 Year Olds', ascending=False)
final_city.loc[:,'16-17 Year Olds'] = pd.to_numeric(final_city['16-17 Year Olds'], errors='coerce')
final_city.loc[:,'75+ Year Olds'] = pd.to_numeric(final_city['75+ Year Olds'], errors='coerce')
# final_city = final_city.sort_values(by='75+ Year Olds', ascending=False).head(20)
final_city


# In[38]:


melted_df = pd.melt(final_city, id_vars=['Geography', 'Quarter'], 
                    value_vars=['16-17 Year Olds', '18-24 Year Olds', '25-34 Year Olds', 
                                '35-44 Year Olds', '45-54 Year Olds', '55-64 Year Olds',
                                '65-74 Year Olds', '75+ Year Olds', 'Age Unknown'], 
                    var_name='Age Band', value_name='Average')
melted_df = melted_df.sort_values(by='Quarter')
melted_df


# In[39]:


melted_df['Age Band 2'] = np.where(melted_df['Age Band'].isin(['16-17 Year Olds', '18-24 Year Olds', '25-34 Year Olds']), "16-34 Year Olds", 
np.where(melted_df['Age Band'].isin(['35-44 Year Olds', '45-54 Year Olds']), "35-54 Year Olds", 
         np.where(melted_df['Age Band'].isin(['55-64 Year Olds','65-74 Year Olds','75+ Year Olds']), "55+ Year Olds", 'Age Unknown')))

melted_df = melted_df.groupby(['Geography', 'Quarter', 'Age Band 2'])['Average'].sum().reset_index()
melted_df


# In[40]:


# Step 1: Ensure data is sorted
melted_df = melted_df.sort_values(['Geography', 'Quarter', 'Age Band 2'])
melted_df['Average'] = pd.to_numeric(melted_df['Average'], errors='coerce')

# # Step 2: Get the first quarter's total for each Geography
baseline = (
    melted_df[melted_df['Quarter'] == '2021-03']
    .groupby(['Geography', 'Age Band 2'])['Average']
    .mean()
)
baseline

# # Step 3: Calculate indexed percentage change
melted_df['Indexed Change (%)'] = melted_df.apply(
    lambda row: (
        ((row['Average'] / baseline.get((row['Geography'], row['Age Band 2']), 1)) - 1) * 100
        if (row['Geography'], row['Age Band 2']) in baseline.index and baseline.get((row['Geography'], row['Age Band 2'])) != 0
        else None
    ),
    axis=1
)

# # Step 4: Display result
melted_df_indexed = melted_df[['Geography', 'Quarter', 'Age Band 2', 'Average', 'Indexed Change (%)']].sort_values(['Geography', 'Quarter', 'Age Band 2'])
melted_df_indexed


# In[41]:


top5_increase = melted_df_indexed[melted_df_indexed['Quarter'] == "2024-09"].sort_values(by='Indexed Change (%)', ascending=False).head(5)
top5_increase


# In[42]:


top5_increase_ = melted_df_indexed[melted_df_indexed['Geography'].isin(top5_increase['Geography'])]
top5_increase_


# In[43]:


plot_age_band_proportion(top5_increase_, x='Quarter',  y='Indexed Change (%)', hue='Age Band 2')


# In[44]:


regions = top5_increase_['Geography'].unique()

# Loop through each region and create a plot
for Region in regions:
    plot_age_band_proportion(top5_increase_[(top5_increase_['Geography'] == Region) & (top5_increase_['Age Band 2'] != "Age Unknown")], x='Quarter', y='Indexed Change (%)', hue='Age Band 2', title=f"Proportion by Age Band and Quarter for {Region}")


# In[45]:


final_city = all_city_vols[all_city_pcnt['Geography'].isin(city_filter_['Geography'])]
# final_city.sort_values(by='16-17 Year Olds', ascending=False)
final_city.loc[:,'16-17 Year Olds'] = pd.to_numeric(final_city['16-17 Year Olds'], errors='coerce')
final_city.loc[:,'75+ Year Olds'] = pd.to_numeric(final_city['75+ Year Olds'], errors='coerce')
# final_city = final_city.sort_values(by='75+ Year Olds', ascending=False).head(20)
final_city


# In[46]:


melted_df = pd.melt(final_city, id_vars=['Geography', 'Quarter'], 
                    value_vars=['16-17 Year Olds', '18-24 Year Olds', '25-34 Year Olds', 
                                '35-44 Year Olds', '45-54 Year Olds', '55-64 Year Olds',
                                '65-74 Year Olds', '75+ Year Olds', 'Age Unknown'], 
                    var_name='Age Band', value_name='Total')
melted_df = melted_df.sort_values(by='Quarter')
melted_df


# In[47]:


melted_df['Age Band 2'] = np.where(melted_df['Age Band'].isin(['16-17 Year Olds', '18-24 Year Olds', '25-34 Year Olds']), "16-34 Year Olds", 
np.where(melted_df['Age Band'].isin(['35-44 Year Olds', '45-54 Year Olds']), "35-54 Year Olds", 
         np.where(melted_df['Age Band'].isin(['55-64 Year Olds','65-74 Year Olds','75+ Year Olds']), "55+ Year Olds", 'Age Unknown')))

melted_df = melted_df.groupby(['Geography', 'Quarter', 'Age Band 2'])['Total'].sum().reset_index()
melted_df


# In[48]:


pd.set_option('display.max_rows', 20)
# Step 1: Ensure data is sorted
melted_df = melted_df.sort_values(['Geography', 'Quarter', 'Age Band 2'])
melted_df['Total'] = pd.to_numeric(melted_df['Total'], errors='coerce')

# # Step 2: Get the first quarter's total for each Geography
baseline = (
    melted_df[melted_df['Quarter'] == '2021-03']
    .groupby(['Geography', 'Age Band 2'])['Total']
    .sum()
)
baseline

# # Step 3: Calculate indexed percentage change
melted_df['Indexed Change (%)'] = melted_df.apply(
    lambda row: (
        ((row['Total'] / baseline.get((row['Geography'], row['Age Band 2']), 1)) - 1) * 100
        if (row['Geography'], row['Age Band 2']) in baseline.index and baseline.get((row['Geography'], row['Age Band 2'])) != 0
        else None
    ),
    axis=1
)

# # Step 4: Display result
melted_df_indexed = melted_df[['Geography', 'Quarter', 'Age Band 2', 'Total', 'Indexed Change (%)']].sort_values(['Geography', 'Quarter', 'Age Band 2'])
melted_df_indexed


# In[49]:


melted_df_indexed = melted_df_indexed[~(melted_df_indexed['Indexed Change (%)'].isna()) & ~(melted_df_indexed['Age Band 2'] == "Age Unknown")]
melted_df_indexed


# In[50]:


top5_increase = melted_df_indexed[(melted_df_indexed['Quarter'] == "2024-09") & (melted_df_indexed['Total'] > 49)].sort_values(by='Indexed Change (%)', ascending=False).head(10)
top5_increase


# In[51]:


top5_increase_ = melted_df_indexed[melted_df_indexed['Geography'].isin(top5_increase['Geography'])]
top5_increase_


# In[52]:


regions = top5_increase_['Geography'].unique()

# Loop through each region and create a plot
for Region in regions:
    plot_age_band_proportion(top5_increase_[(top5_increase_['Geography'] == Region) & (top5_increase_['Age Band 2'] != "Age Unknown")], x='Quarter', y='Indexed Change (%)', hue='Age Band 2', title=f"Proportion by Age Band and Quarter for {Region}")


# In[53]:


city_quarter = pd.melt(final_city, id_vars=['Geography', 'Quarter'], 
                    value_vars=['16-17 Year Olds', '18-24 Year Olds', '25-34 Year Olds', 
                                '35-44 Year Olds', '45-54 Year Olds', '55-64 Year Olds',
                                '65-74 Year Olds', '75+ Year Olds', 'Age Unknown'], 
                    var_name='Age Band', value_name='Total')
city_quarter = city_quarter.sort_values(by='Quarter')
city_quarter


# In[54]:


city_quarter = city_quarter.groupby(['Geography', 'Quarter'])['Total'].sum().reset_index()
city_quarter


# In[55]:


pd.set_option('display.max_rows', 20)
# Step 1: Ensure data is sorted
city_quarter = city_quarter.sort_values(['Geography', 'Quarter']).copy()
city_quarter['Total'] = pd.to_numeric(city_quarter['Total'], errors='coerce')

# # Step 2: Get the first quarter's total for each Geography
baseline = (
    city_quarter[city_quarter['Quarter'] == '2021-03']
    .groupby(['Geography'])['Total']
    .sum()
)
baseline

# # Step 3: Calculate indexed percentage change
city_quarter['Indexed Change (%)'] = city_quarter.apply(
    lambda row: (
        ((row['Total'] / baseline.get(row['Geography'], 1)) - 1) * 100
        if row['Geography'] in baseline.index and baseline[row['Geography']] != 0
        else None
    ),
    axis=1
)

# # Step 4: Display result
city_quarter = city_quarter[['Geography', 'Quarter', 'Total', 'Indexed Change (%)']].sort_values(['Geography', 'Quarter'])
city_quarter


# In[56]:


top5_increase = city_quarter[(city_quarter['Quarter'] == "2024-09") & (melted_df_indexed['Total'] > 0)].sort_values(by='Indexed Change (%)', ascending=True).head(10)
top5_increase


# In[57]:


top5_increase_ = city_quarter[city_quarter['Geography'].isin(top5_increase['Geography'])]
top5_increase_


# In[58]:


regions = top5_increase_['Geography'].unique()

# Loop through each region and create a plot
for Region in regions:
    plot_age_band_proportion(top5_increase_[(top5_increase_['Geography'] == Region)], x='Quarter', y='Indexed Change (%)', hue=None, title=f"Proportion by Quarter for {Region}")


# In[ ]:





# In[ ]:





# In[ ]:




