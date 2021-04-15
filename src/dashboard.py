import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel("data/One Piece Database.xlsx", sheet_name = "Chapter Appearances", engine = 'openpyxl')
df['Chapter Number'] = df['Chapter Number'].ffill()
df['Arc Name'] = df['Arc Name'].ffill()
all_dims = ['Chapter Number', 'Character', 'Arc Name']
# cplot = sns.countplot(x = 'Character', data = df)
# fig = px.histogram(df, x='Character', color="Arc Name", barmode='group')
# fig.show()
df_abilities = pd.read_excel("data/One Piece Database.xlsx", sheet_name = "Abilities History", engine = 'openpyxl')
df_abilities['Chapter'] = df_abilities['Chapter'].ffill()
df_abilities['Arc Name'] = df_abilities['Arc Name'].ffill()
abilities_dims = ['Chapter', 'Ability Name', 'Arc Name']

fig_char = px.histogram(df, x='Arc Name', color="Character", barmode='group')
fig_ability = px.histogram(df_abilities, x='Arc Name', color="Ability Name", barmode='group')

app = dash.Dash(__name__)

app.layout = html.Div(
    children = [
                html.H1(children='One Dash'),
                dcc.Graph(id="histo", figure=fig_char),
                dcc.Graph(id="histo_abilities", figure = fig_ability),
                ]
                    )

app.run_server(debug=True)