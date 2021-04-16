import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from src.arcs import generate_arc

appearance_df = pd.read_csv("data/onedash_chap_appearance.csv")
appearance_df['Chapter'] = appearance_df['Chapter'].ffill()
# df['Arc Name'] = df['Arc Name'].ffill()
all_dims = ['Chapter', 'Appearance', 'Arc', 'Character', 'Appearance Notes']
# preprocess to add arc 
arcs = generate_arc(1010 + 1)
appearance_df['Character'] = appearance_df['Appearance'].str.split("(",expand=True)[0]
appearance_df['Appearance Notes'] = appearance_df['Appearance'].str.split("(",expand=True)[1]
appearance_df['Appearance Notes'] = appearance_df['Appearance Notes'].str.replace(")", "", regex = True)
def arc_col(row):
    for key in arcs:
        if row['Chapter'] in arcs[key]:
            return key
    return "None"
appearance_df['Arc'] = appearance_df.apply(arc_col, axis =1) 
# print (appearance_df.head())

# fig_char = px.histogram(appearance_df, x='Chapter', barmode='group')
fig_char = px.histogram(appearance_df, x='Character', color = 'Arc', barmode='group')

# fig_ability = px.histogram(df_abilities, x='Arc Name', color="Ability Name", barmode='group')

app = dash.Dash(__name__)

app.layout = html.Div(
    children = [
                html.H1(children='One Dash'),
                dcc.Graph(id="histo", figure=fig_char),
                # dcc.Graph(id="histo_abilities", figure = fig_ability),
                ]
                    )

if __name__ == '__main__':
    app.run_server(debug=True)