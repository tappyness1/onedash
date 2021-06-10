import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import pandas as pd
import numpy as np
from src.arcs import generate_arc
from src.preprocess import get_last_known_bounty, get_latest_age, get_main_crew
from configparser import ConfigParser, ExtendedInterpolation
import warnings
warnings.filterwarnings("ignore")


pl_config = ConfigParser(interpolation=ExtendedInterpolation())
pl_config.read('cfg/cfg.ini')

end_chap = pl_config['SCRAPER'].getint('end_chap') + 1
char_link_fp = pl_config['SCRAPER'].get('char_link_fp')
chap_appearance_fp = pl_config['SCRAPER'].get('chap_appearance_fp')
char_details_fp = pl_config['SCRAPER'].get('char_details_fp')


# appearance_df['Chapter'] = appearance_df['Chapter'].ffill()
# df['Arc Name'] = df['Arc Name'].ffill()
all_dims = ['Chapter', 'Appearance', 'Arc', 'Character', 'Appearance Notes']

# preprocess to add arc 
arcs = generate_arc(end_chap)

def arc_col(row):
    """function to generate arc per row for appearance df
    """
    for key in arcs:
        if row['Chapter'] in arcs[key]:
            return key
    return "None"

appearance_df = pd.read_csv(chap_appearance_fp)
char_details_df = pd.read_csv(char_details_fp)


appearance_df['Appearance'] = appearance_df['Character'].str.split("(",expand=True)[0]
appearance_df['Appearance Notes'] = appearance_df['Character'].str.split("(",expand=True)[1]
appearance_df['Appearance Notes'] = appearance_df['Appearance Notes'].str.replace(")", "", regex = True)
appearance_df['Arc'] = appearance_df.apply(arc_col, axis =1) 


char_details_df['last_bounty'] = char_details_df['bounty'].apply(get_last_known_bounty)
char_details_df['latest_age'] = char_details_df['age'].apply(get_latest_age)
char_details_df['latest_age']= char_details_df['latest_age'].fillna(value=np.nan)
char_details_df['main_crew'] = char_details_df['affiliation'].apply(get_main_crew)

df_age_bounty = char_details_df.dropna(subset=['latest_age', 'last_bounty'])
df_age_bounty['latest_age'] = df_age_bounty['latest_age'].astype('int')

@st.cache(suppress_st_warning=True)
def main():
    # Select Plot Option
    st.sidebar.markdown("## Select Mode of Analysis")
    char_appearance = st.sidebar.checkbox('Character Appearance by arc', value = True)  
    latest_bounty = st.sidebar.checkbox('Latest Bounty', value = False)
    latest_age_to_bounty = st.sidebar.checkbox('Latest Bounty by age', value = False)
    age_to_bounty_by_crew = st.sidebar.checkbox('Latest Bounty grouped by crew', value = False)

    fig_app_by_arc = px.histogram(appearance_df, 
                                x='Appearance', 
                                color = 'Arc', 
                                barmode='group',
                                labels={
                                    "Appearance": "Name",
                                    "counts": "Counts"
                                },
                                title="Count of Character Appearance by Arc")

    fig_latest_bounty = px.histogram(char_details_df, 
                                    x="last_bounty", 
                                    nbins =  00, 
                                    title="Bounty histogram")
    fig_latest_age_to_bounty = px.scatter(x = df_age_bounty['latest_age'], 
                                        y=df_age_bounty['last_bounty'], 
                                        color = df_age_bounty['Name'],
                                        labels={
                                            "latest_age": "Age",
                                            "last_bounty": "Latest Bounty",
                                            "Name": "Name"
                                            },
                                        title = "Bounty by Age")
    fig_latest_age_to_bounty.update_xaxes(tickangle=0)

    fig_age_to_bounty_by_crew = px.scatter(x = df_age_bounty['latest_age'], 
                                        y=df_age_bounty['last_bounty'], 
                                        color = df_age_bounty['main_crew'],
                                        labels={
                                            "latest_age": "Age",
                                            "last_bounty": "Latest Bounty",
                                            "main_crew": "Crew"
                                            },
                                        title = "Bounty by Age grouped by Crew")
    fig_age_to_bounty_by_crew.update_xaxes(tickangle=0)

    if char_appearance:
        st.plotly_chart(fig_app_by_arc)

    if latest_bounty:
        st.plotly_chart(fig_latest_bounty)

    if latest_age_to_bounty:
        st.plotly_chart(fig_latest_age_to_bounty)

    if age_to_bounty_by_crew:
        st.plotly_chart(fig_age_to_bounty_by_crew)

if __name__ == "__main__":
    main()