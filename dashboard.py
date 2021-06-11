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

appearance_df = pd.read_csv(chap_appearance_fp)
char_details_df = pd.read_csv(char_details_fp)
df_age_bounty = pd.read_csv(age_bounty_fp)

all_dims = ['Chapter', 'Appearance', 'Arc', 'Character', 'Appearance Notes']

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

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    children = [
                html.H1(children='One Dash'),
                dcc.Graph(id="histo_app_by_arc", 
                          figure=fig_app_by_arc),
                dcc.Graph(id="histo_latest_bounty", 
                          figure=fig_latest_bounty),
                dcc.Graph(id="scatter_latest_age_to_bounty", 
                          figure=fig_latest_age_to_bounty),
                dcc.Graph(id="scatter_age_bounty_crew", 
                          figure=fig_age_to_bounty_by_crew),
                ]
                    )

if __name__ == '__main__':
    app.run_server(debug=True)