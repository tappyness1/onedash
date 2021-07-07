import pandas as pd
import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
# import seaborn as sns
# import matplotlib.pyplot as plt
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

def generate_df():
    appearance_df = pd.read_csv(chap_appearance_fp)
    char_details_df = pd.read_csv(char_details_fp)
    df_age_bounty = pd.read_csv(age_bounty_fp)
    return appearance_df, char_details_df, df_age_bounty

all_dims = ['Chapter', 'Appearance', 'Arc', 'Character', 'Appearance Notes']

appearance_df, char_details_df, df_age_bounty = generate_df()

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

def fig_app_by_arc_sunburst(appearance_df):
    fig_app_by_arc_sunburst = px.sunburst(appearance_df[appearance_df['Appearance'].isin(appearance_df['Appearance'].value_counts().head(10).index.tolist())], 
                                path = ['Appearance', 'Arc'],
                                width = 800,
                                height = 800)
    return fig_app_by_arc_sunburst

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

height = 650

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div(
    children = [
                html.H1(children='One Dash'),
                html.H2(children='''
                    Top 20 Character Appearance
                '''),
                dcc.Graph(id="histo_app_by_arc", 
                          figure=fig_app_by_arc(appearance_df, height)),
                html.H2(children='''
                    Top 10 Character Appearance!
                '''),
                html.H3(children='''
                    Click on the name to expand on their info!
                '''),
                dcc.Graph(id="fig_app_by_arc_sunburst",
                          figure=fig_app_by_arc_sunburst(appearance_df)),
                html.H2(children='''
                    Top 50 Latest bounty (log scaled)
                '''),
                dcc.Graph(id="histo_latest_bounty", 
                          figure=fig_latest_bounty(char_details_df, height)),
                html.H2(children='''
                    Bounty histogram
                '''),
                dcc.Graph(id = "fig_latest_bounty_dist",
                         figure = fig_latest_bounty_dist(char_details_df, height)),
                html.H2(children='''
                    Bounty by Age
                '''),
                dcc.Graph(id="scatter_latest_age_to_bounty", 
                          figure=fig_latest_age_to_bounty(df_age_bounty,height)),
                html.H2(children='''
                    Bounty by Age grouped by Crew
                '''),
                dcc.Graph(id="scatter_age_bounty_crew", 
                          figure=fig_age_to_bounty_by_crew(df_age_bounty, height)),
                ]
                    )

if __name__ == '__main__':
    app.run_server(debug=True)