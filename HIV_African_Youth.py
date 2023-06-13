#!/usr/bin/env python
# coding: utf-8

# # Install/Import Necessary Packages

# In[ ]:


# import subprocess

# # Install pandas using pip
# subprocess.run(['pip', 'install', '-q', 'pandas'])

# !pip install -q pandas
# !pip install -q streamlit
# !pip install -q matplotlib
# !pip install -q seaborn
# !pip install -q plotly
# !pip install -q geopandas


# In[ ]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import locale
import geopandas as gpd
import numpy as np
import altair as alt


# # Import CSV Files & Preprocess

# ## Deaths by Cause

# In[ ]:


# Import the CSV file
death_cause = pd.read_csv("Annual Number of Deaths by Cause.csv")

# Remove unwanted parts from column names
new_columns = []
for column in death_cause.columns:
    # Remove "Deaths - " from column names
    new_column = column.replace("Deaths - ", "")
    # Remove " - Sex:" and everything after it
    index = new_column.find(" - Sex:")
    if index != -1:
        new_column = new_column[:index]
    new_columns.append(new_column)

# Assign the new column names to the DataFrame
death_cause.columns = new_columns

# Drop the specified columns
columns_to_drop = ["Number of executions (Amnesty International)", "Code"]
death_cause = death_cause.drop(columns=columns_to_drop)

# Rename the 'Entity' column to 'Country'
death_cause = death_cause.rename(columns={"Entity": "Country"})


# ## HIV Death Rates

# In[ ]:


# Read the CSV file
death_rates = pd.read_csv("HIV Death Rates.csv")

# Preprocess the data
death_rates['Death Rates per 100,000 People'] = pd.to_numeric(death_rates['Death Rates per 100,000 People'], errors='coerce')  # Convert Death Rates to numeric, replacing invalid values with NaN
bins = [0, 5, 10, 50, 100, 500, float('inf')]
death_rates['Color Category'] = pd.cut(death_rates['Death Rates per 100,000 People'], bins=bins, labels=[1, 2, 3, 4, 5, 6], right=False)

labels = ['<5', '5 - 10', '10 - 50', '50 - 100', '100 - 500', '>500']
death_rates['Death Rate Category'] = pd.cut(death_rates['Death Rates per 100,000 People'], bins=bins, labels=labels, right=False)


# ## HIV Infection Rate

# In[ ]:


# Read the CSV file
child_inf = pd.read_csv("HIV Infections (Child).csv")
child_inf_rest = pd.read_csv("HIV Infections (Child-Rest of the World).csv")

# Read the CSV file
adol_inf = pd.read_csv("HIV Infections (Adolescents).csv")
adol_inf_rest = pd.read_csv("HIV Infections (Adolescents-Rest of the World).csv")


# ## HIV by Gender

# In[ ]:


# Read the CSV file
gender_HIV = pd.read_csv('Gender Share of HIV.csv', dtype={'Year': int})

# Filter the DataFrame to include only African countries
african_countries = [
    'Algeria', 'Angola', 'Benin', 'Botswana', 'Burkina Faso', 'Burundi', 'Cabo Verde', 'Cameroon',
    'Central African Republic', 'Chad', 'Comoros', 'Congo', 'Democratic Republic of the Congo',
    'Djibouti', 'Egypt', 'Equatorial Guinea', 'Eritrea', 'Eswatini', 'Ethiopia', 'Gabon', 'Gambia',
    'Ghana', 'Guinea', 'Guinea-Bissau', 'Ivory Coast', 'Kenya', 'Lesotho', 'Liberia', 'Libya',
    'Madagascar', 'Malawi', 'Mali', 'Mauritania', 'Mauritius', 'Morocco', 'Mozambique', 'Namibia',
    'Niger', 'Nigeria', 'Rwanda', 'São Tomé and Príncipe', 'Senegal', 'Seychelles', 'Sierra Leone',
    'Somalia', 'South Africa', 'South Sudan', 'Sudan', 'Tanzania', 'Togo', 'Tunisia', 'Uganda',
    'Zambia', 'Zimbabwe'
]

gender_HIV_african = gender_HIV[gender_HIV['Country'].isin(african_countries)]


# ## Mother to Son Transmission Prevention

# In[ ]:


PMTCT = pd.read_csv('PMTCT.csv', dtype={'Year': int})


# ## HIV Prevention Knowledge

# In[ ]:


prev_know = pd.read_csv('HIV Prevention Awareness.csv')

prev_know_african = prev_know[prev_know['Entity'].isin(african_countries)]


# ## Condom Use

# In[ ]:


cond_know = pd.read_csv('Condom Use 2016.csv')

cond_know_african = cond_know[cond_know['Entity'].isin(african_countries)]

cond_know_african_sorted = cond_know_african.sort_values(by='Condom use at last high-risk sex - All sexes Adults (15-49)')


# ## Mother to Child Tranmission

# In[ ]:


# Step 1: Read the CSV file as a pandas DataFrame
mom_child = pd.read_csv('Mother Child Transmission.csv')


# # Streamlit WebApp

# In[1]:


# Set page config to wide layout
st.set_page_config(layout="wide")

# Define the tabs
tabs = ["Home", "Death by HIV", "Demographic Prevalence", "Children HIV Factors", "Youth HIV Factors"]

# Create the tabs
selected_tab = st.sidebar.radio("Select Tab", tabs)

# Home tab content
if selected_tab == "Home":

    # Set the page title and the header
    st.title("ABOUT HIV")
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Write a brief about HIV
    st.markdown("<h2>What is HIV?</h2>", unsafe_allow_html=True)
    st.markdown("""
    - HIV (Human Immunodeficiency Virus) is a virus that attacks the immune system
    - There is currently no effective cure
    - However, an effective HIV treatment exists that allows HIV-infected people to live long, healthy lives and protect their partners
    """)

    st.markdown("<h2>Where did HIV come from?</h2>", unsafe_allow_html=True)
    st.write("""
    - HIV is believed to have originated from a type of chimpanzee in Central Africa
    - The virus was transmitted to humans through hunting and consumption of meat""")

    st.markdown("<h2>What are the Symptoms?</h2>", unsafe_allow_html=True)
    st.write("""
    - Symptoms of HIV vary but can include flu-like symptoms such as fever, fatigue, swollen lymph nodes, and sore throat. As the infection progresses, it can lead to more severe symptoms and opportunistic infections. \n
    - Some people experience no symptoms. \n
    - The only reliable method to determine if someone has HIV is to undergo testing.""")

    st.markdown("<h2>What are the Modes of Transmisstion?</h2>", unsafe_allow_html=True)
    st.write("""
    - HIV is primarily transmitted through sexual contact
    - HIV can be transmitted from a mother to her child during pregnancy, childbirth, or breastfeeding
    - Non-sexual transmission of HIV can occur through the sharing of injection equipment, such as needles""")

# Add content for Death by HIV tab
elif selected_tab == "Death by HIV":
    st.title("HIV/AIDS:  One of the most deadly infectious diseases")

    # Add some space
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Get unique years from the dataset
    years = death_cause['Year'].unique()

    # Select year using a slider
    year = st.slider("Select Year", min_value=int(min(years)), max_value=int(max(years)), step=1)

    # Filter the dataset based on the selected year and exclude countries with "WHO" in their name
    filtered_df = death_cause[(death_cause['Year'] == year) & (~death_cause['Country'].str.contains("WHO"))]

    # Calculate the total deaths for each cause
    total_deaths = filtered_df.iloc[:, 2:].sum()

    # Sort causes based on total deaths in descending order
    total_deaths = total_deaths.sort_values(ascending=True)

    # Create the horizontal bar plot
    fig, ax = plt.subplots(figsize=(30, 18))
    ax.barh(total_deaths.index, total_deaths.values, color='red')
    ax.set_yticklabels(total_deaths.index, fontsize=20)
    ax.set_title(f'Total Deaths by Cause ({year})', fontsize=20)

    # Display number of deaths to the right of each bar
    for i, v in enumerate(total_deaths.values):
        if v >= 1_000_000:
            formatted_deaths = f"{v/1_000_000:.2f} million"  # Convert to millions without decimal points
        else:
            formatted_deaths = "{:,.0f}".format(v)  # Format with commas for thousands
        ax.text(v, i, formatted_deaths, va='center', color='black', fontsize=20)

    # Remove X axis
    ax.xaxis.set_visible(False)

    # Adjust layout
    plt.tight_layout()

    # Display the plot
    st.pyplot(fig)

    # Add some space
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Select the year using a slider
    min_year = int(death_rates['Year'].min())
    max_year = int(death_rates['Year'].max())
    selected_year = st.slider("Select Year", min_value=min_year, max_value=max_year, value=max_year)

    # Filter the data based on the selected year
    filtered_data = death_rates[death_rates['Year'] == selected_year]

    # Define the color categories and colors
    color_categories = ['<5', '5 and 10', '10 and 50', '50 and 100', '100 and 500', '>500']
    colors = ['rgb(255, 255, 204)', 'rgb(255, 237, 160)', 'rgb(254, 217, 118)',
              'rgb(253, 141, 60)', 'rgb(252, 78, 42)', 'rgb(227, 26, 28)']

    fig = px.choropleth(filtered_data, locations='Code', color='Death Rate Category', hover_name='Country',
                        projection='natural earth', title=f'HIV Death Rates - Year {selected_year}',
                        hover_data=['Death Rates per 100,000 People'], color_discrete_sequence=colors,
                        category_orders={'Color Category': color_categories},
                        labels={'Color Category': 'Death Rate Category'}
                       )
    fig.update_geos(showcountries=True, countrycolor="gray", showcoastlines=False)

    st.plotly_chart(fig, use_container_width=True)

# Add content for the tab as needed
elif selected_tab == "Demographic Prevalence":
    st.title("Youth and Women at Risk: High HIV Prevalence in Africa")

    # Add some space
    st.markdown("<br><br>", unsafe_allow_html=True)

    year_min = gender_HIV['Year'].min()
    year_max = gender_HIV['Year'].max()

    year = st.slider('Select Year', min_value=int(year_min), max_value=int(year_max))
    filtered_data = gender_HIV_african[gender_HIV_african['Year'] == year]

    # Sort the data by the share of women in descending order
    filtered_data = filtered_data.sort_values(by='Share of Women living with HIV (%)', ascending=False)

    fig, ax = plt.subplots(figsize=(15, 12))
    sns.set(style='darkgrid')

    # Plot the bar chart for women
    sns.barplot(x=filtered_data['Share of Women living with HIV (%)'], y=filtered_data['Country'],
                color='pink', label='Women')

    # Plot the bar chart for men
    sns.barplot(x=-filtered_data['Share of Men living with HIV (%)'], y=filtered_data['Country'],
                color='skyblue', label='Men')

    # Add labels and title
    ax.set_xlabel('Percentage')
    ax.set_ylabel('Country')
    ax.set_title(f'Gender Share of HIV in African Countries - {year}')

    # Format the x-axis tick labels as absolute values
    xticks_abs = np.abs(ax.get_xticks()).astype(int)
    ax.set_xticklabels(xticks_abs)

    # Remove the left spine and move the y-axis labels to the right
    ax.spines['left'].set_visible(False)
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position('right')

    # Set the legend
    ax.legend()

    # Add text annotations to each bar
    for i, (women_share, men_share, country) in enumerate(zip(filtered_data['Share of Women living with HIV (%)'],
                                                              filtered_data['Share of Men living with HIV (%)'],
                                                              filtered_data['Country'])):
        # Calculate the midpoint of each bar
        bar_midpoint = (women_share + men_share) / 2

        # Display the share as text at the midpoint
        ax.text(bar_midpoint, i, f'{women_share:.1f}%', ha='center', va='center', color='black', fontsize=10)

    st.pyplot(fig)

    # Add some space
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Group the data by year and calculate the sum of OBS_VALUE
    grouped_data1 = adol_inf.groupby("TIME_PERIOD:Time period")["OBS_VALUE:Observation Value"].sum()
    grouped_data2 = child_inf.groupby("TIME_PERIOD:Time period")["OBS_VALUE:Observation Value"].sum()
    grouped_data3 = adol_inf_rest.groupby("TIME_PERIOD:Time period")["OBS_VALUE:Observation Value"].sum()
    grouped_data4 = child_inf_rest.groupby("TIME_PERIOD:Time period")["OBS_VALUE:Observation Value"].sum()

    # Extract the years and summed rates
    # Extract the years and summed rates
    years1 = np.array(grouped_data1.index)
    summed_rates1 = np.array(grouped_data1.values)
    years2 = np.array(grouped_data2.index)
    summed_rates2 = np.array(grouped_data2.values)
    years3 = np.array(grouped_data3.index)
    summed_rates3 = np.array(grouped_data3.values)
    years4 = np.array(grouped_data4.index)
    summed_rates4 = np.array(grouped_data4.values)

    # Set a larger figure size
    plt.figure(figsize=(15, 8))

    # Create the plot
    plt.plot(years1, summed_rates1*100, label="Adolescents aged 10-19 (Africa)")
    plt.plot(years2, summed_rates2*100, label="Children aged 0-10 (Africa)")
    plt.plot(years3, summed_rates3*100, label="Adolescents aged 10-19 (Rest of The World)")
    plt.plot(years4, summed_rates4*100, label="Children aged 0-10 (Rest of The World)")
    plt.xlabel("Year")
    plt.ylabel("")
    plt.title("New HIV Infection per 100,000 Uninfected People")
    plt.legend()

    # Set the x-axis ticks to display every year
    years1 = years1.astype(int).tolist()
    plt.xticks(years1)

    # Display the plot in Streamlit
    st.pyplot(plt)


# Add content for the tab as needed
elif selected_tab == "Children HIV Factors":
    st.title("Unraveling the Causes: HIV in Children across Africa")

    # Add some space
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Plot the data
    mom_child.sort_values('TIME_PERIOD:Time period', inplace=True)
    fig, ax = plt.subplots()
    ax.plot(mom_child["TIME_PERIOD:Time period"], mom_child["OBS_VALUE:Observation Value"], color='red')
    ax.set_xlabel("Year", fontsize=6)
    ax.set_ylabel("Rate Per 100 HIV-Infected Women", fontsize=6)
    ax.set_title("Mother-to-child HIV transmission rate in Africa", fontsize=6)
    plt.xticks(range(min(mom_child['TIME_PERIOD:Time period']), max(mom_child['TIME_PERIOD:Time period']) + 1), fontsize=6)
    plt.yticks(fontsize=6)  # Set fontsize to 8 for y-axis tick labels

    # Display the plot in Streamlit
    plt.figure(figsize=(20, 10))
    st.pyplot(fig)

    # Add some space
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Set a larger figure size
    plt.figure(figsize=(15, 8))

    # Create a line plot
    plt.plot(PMTCT['Year'], PMTCT['New HIV Infections averted due to PMTCT'], color='green', label='New HIV Infections averted due to PMTCT')
    plt.plot(PMTCT['Year'], PMTCT['New HIV Infections - children (aged 0-14) '], color='red', label='New HIV Infections - children')

    # Customize the plot
    plt.xlabel('Year')
    plt.title('Prevention of mother-to-child transmission (PMTCT) Around the World')
    plt.legend()

    # Set the x-axis tick frequency to show every year
    plt.xticks(np.arange(min(PMTCT['Year']), max(PMTCT['Year'])+1, 1))

    # Display the plot using Streamlit
    st.pyplot(plt)

# Add content for the tab as needed
elif selected_tab == "Youth HIV Factors":
    st.title("Unraveling the Causes: HIV in Adolescents across Africa")

    # Add some space
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Sort data by knowledge
    prev_know_african_sorted = prev_know_african.sort_values(by='Knowledge about HIV prevention in young people - All sexes Young people (15-24) Yes to all questions')

    # Define color scheme
    color_scale = alt.Scale(scheme='redblue')

    # Create a vertical bar chart with color encoding
    chart = alt.Chart(prev_know_african_sorted).mark_bar().encode(
        y=alt.Y('Knowledge about HIV prevention in young people - All sexes Young people (15-24) Yes to all questions', title='Knowledge'),
        x=alt.X('Entity', sort='-y', title='Country'),
        color=alt.Color('Knowledge about HIV prevention in young people - All sexes Young people (15-24) Yes to all questions:Q', title=' ', scale=color_scale)
    )

    # Set chart options
    chart = chart.properties(width=800, height=500,
                            title=alt.TitleParams(text='Knowledge about HIV Prevention in Young People by Country', fontSize=12)
)

    # Render the chart using Streamlit
    st.altair_chart(chart, use_container_width=True)

    # Add some space
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Create a vertical bar chart with color encoding
    chart = alt.Chart(cond_know_african_sorted).mark_bar().encode(
        y=alt.Y('Condom use at last high-risk sex - All sexes Adults (15-49)', title='Percent (%)'),
        x=alt.X('Entity', sort='-y', title='Country'),
        color=alt.Color('Condom use at last high-risk sex - All sexes Adults (15-49)', title=' ', scale=color_scale)
    )

    # Set chart options
    chart = chart.properties(width=800, height=500,
                             title=alt.TitleParams(text='Share of People Aged 15-49 Who Used a Condom Among During Intercourse With a Non-Marital, Non-Cohabiting Sexual Partner', fontSize=12)
)

    # Render the chart using Streamlit
    st.altair_chart(chart, use_container_width=True)

