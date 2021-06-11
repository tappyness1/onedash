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
age_bounty_fp = pl_config['SCRAPER'].get('age_bounty_fp')

# all_dims = ['Chapter', 'Appearance', 'Arc', 'Character', 'Appearance Notes']

appearance_df = pd.read_csv(chap_appearance_fp)
char_details_df = pd.read_csv(char_details_fp)
df_age_bounty = pd.read_csv(age_bounty_fp)

st. set_page_config(layout="wide")

# Select Plot Option
st.sidebar.markdown("## Select Mode of Analysis")
char_appearance = st.sidebar.checkbox('Top 20 Character Appearance by arc', value = True)
char_appearance_sunburst = st.sidebar.checkbox('Top 10 Character Appearance by arc (Sunburst)', value = True)
latest_bounty = st.sidebar.checkbox('Latest Bounty', value = False)
latest_age_to_bounty = st.sidebar.checkbox('Latest Bounty by age', value = False)
age_to_bounty_by_crew = st.sidebar.checkbox('Latest Bounty grouped by crew', value = False)

fig_app_by_arc = px.histogram(appearance_df[appearance_df['Appearance'].isin(appearance_df['Appearance'].value_counts().head(20).index.tolist())], 
                            x='Appearance', 
                            color = 'Arc', 
                            barmode='group',
                            labels={
                                "Appearance": "Name",
                                "counts": "Counts"
                            },
                            title="Count of Character Appearance by Arc")

fig_app_by_arc_sunburst = px.sunburst(appearance_df[appearance_df['Appearance'].isin(appearance_df['Appearance'].value_counts().head(10).index.tolist())], 
                             path = ['Appearance', 'Arc'],
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
    st.plotly_chart(fig_app_by_arc,use_container_width=True)

if char_appearance_sunburst:
    st.plotly_chart(fig_app_by_arc_sunburst,use_container_width=True)

if latest_bounty:
    st.plotly_chart(fig_latest_bounty,use_container_width=True)

if latest_age_to_bounty:
    st.plotly_chart(fig_latest_age_to_bounty,use_container_width=True)

if age_to_bounty_by_crew:
    st.plotly_chart(fig_age_to_bounty_by_crew, use_container_width=True)
