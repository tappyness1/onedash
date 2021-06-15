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

# all_dims = ['Chapter', 'Appearance', 'Arc', 'Character', 'Appearance Notes']

pl_config = ConfigParser(interpolation=ExtendedInterpolation())
pl_config.read('cfg/cfg.ini')

end_chap = pl_config['SCRAPER'].getint('end_chap') + 1
char_link_fp = pl_config['SCRAPER'].get('char_link_fp')
chap_appearance_fp = pl_config['SCRAPER'].get('chap_appearance_fp')
char_details_fp = pl_config['SCRAPER'].get('char_details_fp')
age_bounty_fp = pl_config['SCRAPER'].get('age_bounty_fp')

st.set_page_config(page_title='One Dash', layout = 'wide', initial_sidebar_state = 'auto')

@st.cache(suppress_st_warning=True)
def generate_df():
    appearance_df = pd.read_csv(chap_appearance_fp)
    char_details_df = pd.read_csv(char_details_fp)
    df_age_bounty = pd.read_csv(age_bounty_fp)
    return appearance_df, char_details_df, df_age_bounty

@st.cache(suppress_st_warning=True)
def fig_app_by_arc(appearance_df, height):
    fig_app_by_arc = px.histogram(appearance_df[appearance_df['Appearance'].isin(appearance_df['Appearance'].value_counts().head(20).index.tolist())], 
                                x='Appearance', 
                                color = 'Arc', 
                                barmode='group',
                                labels={
                                    "Appearance": "Name",
                                    "counts": "Counts"
                                },
                                height = height
                                )

    fig_app_by_arc.update_layout(
                                xaxis_title="Name",
                                yaxis_title="",
                                )
    return fig_app_by_arc

@st.cache(suppress_st_warning=True)
def fig_app_by_arc_sunburst(appearance_df):
    fig_app_by_arc_sunburst = px.sunburst(appearance_df[appearance_df['Appearance'].isin(appearance_df['Appearance'].value_counts().head(10).index.tolist())], 
                                path = ['Appearance', 'Arc'],
                                width = 800,
                                height = 800)
    return fig_app_by_arc_sunburst

@st.cache(suppress_st_warning=True)
def fig_latest_bounty(char_details_df, height):
    fig_latest_bounty = px.bar(char_details_df[char_details_df['last_bounty'] > 0].head(50), 
                            x = 'Name', 
                            y = 'last_bounty', 
                            height = height, 
                            log_y = True)
    fig_latest_bounty.update_layout(
                                xaxis_title="Name",
                                yaxis_title="Last Bounty",
                                xaxis={'categoryorder':'total descending'}
                                )
    return fig_latest_bounty

@st.cache(suppress_st_warning=True)
def fig_latest_bounty_dist(char_details_df, height):
    fig_latest_bounty_dist = px.histogram(char_details_df, 
                                    x="last_bounty", 
                                    nbins =  00,
                                    height = height)

    fig_latest_bounty_dist.update_layout(
                                xaxis_title="Bounty Group",
                                yaxis_title="",
                                )
    return fig_latest_bounty_dist

@st.cache(suppress_st_warning=True)
def fig_latest_age_to_bounty(df_age_bounty,height):
    fig_latest_age_to_bounty = px.scatter(x = df_age_bounty['latest_age'], 
                                        y=df_age_bounty['last_bounty'], 
                                        color = df_age_bounty['Name'],
                                        labels={
                                            "latest_age": "Age",
                                            "last_bounty": "Latest Bounty",
                                            "Name": "Name"
                                            },
                                        height = height)
    fig_latest_age_to_bounty.update_xaxes(tickangle=0)
    fig_latest_age_to_bounty.update_layout(
                                            xaxis_title="Age",
                                            yaxis_title="Bounty Amount",
                                            )
    return fig_latest_age_to_bounty

@st.cache(suppress_st_warning=True)
def fig_age_to_bounty_by_crew(df_age_bounty, height):
    fig_age_to_bounty_by_crew = px.scatter(x = df_age_bounty['latest_age'], 
                                        y=df_age_bounty['last_bounty'], 
                                        color = df_age_bounty['main_crew'],
                                        labels={
                                            "latest_age": "Age",
                                            "last_bounty": "Latest Bounty",
                                            "main_crew": "Crew"
                                            },
                                            height = height)
    fig_age_to_bounty_by_crew.update_xaxes(tickangle=0)
    fig_age_to_bounty_by_crew.update_layout(
                                            xaxis_title="Age",
                                            yaxis_title="Bounty Amount",
                                            )
    return fig_age_to_bounty_by_crew

# @st.cache(suppress_st_warning=True, persist = True)
def main():
    appearance_df, char_details_df, df_age_bounty = generate_df()
    # st.set_page_config(layout="wide")
    height = 650

    st.markdown(""" <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style> """, 
                unsafe_allow_html=True
                )

    # Select Plot Option
    st.sidebar.markdown("## Character Appearance in each chapter")
    char_appearance = st.sidebar.checkbox('Top 20 Character Appearance', value = True)
    char_appearance_sunburst = st.sidebar.checkbox('Top 10 Character Appearance (Sunburst)', value = True)

    st.sidebar.markdown("## Bounty")
    char_bounty = st.sidebar.checkbox('Bounties (Descending order)', value = True)
    latest_bounty = st.sidebar.checkbox('Bounty Distribution', value = False)
    latest_age_to_bounty = st.sidebar.checkbox('Latest Bounty by age', value = False)
    age_to_bounty_by_crew = st.sidebar.checkbox('Latest Bounty grouped by crew', value = False)

    if char_appearance:
        st.write("## Top 20 Character Appearance")
        st.plotly_chart(fig_app_by_arc(appearance_df, height),use_container_width=True)

    if char_appearance_sunburst:
        st.write("## Top 10 Character Appearance!")
        st.write("### Click on the name to expand on their info!")
        st.plotly_chart(fig_app_by_arc_sunburst(appearance_df),use_container_width=True)

    if char_bounty:
        st.write("## Top 50 Latest bounty (log scaled)")
        st.plotly_chart(fig_latest_bounty(char_details_df, height),use_container_width=True)

    if latest_bounty:
        st.write("## Bounty histogram")
        st.plotly_chart(fig_latest_bounty_dist(char_details_df, height),use_container_width=True)

    if latest_age_to_bounty:
        st.write("## Bounty by Age")
        st.plotly_chart(fig_latest_age_to_bounty(df_age_bounty,height),use_container_width=True)

    if age_to_bounty_by_crew:
        st.write("## Bounty by Age grouped by Crew")
        st.plotly_chart(fig_age_to_bounty_by_crew(df_age_bounty, height), use_container_width=True)

if __name__ == "__main__":
    main()