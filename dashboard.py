import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
# import seaborn as sns
# import matplotlib.pyplot as plt
from arcs import generate_arc
from configparser import ConfigParser, ExtendedInterpolation

pl_config = ConfigParser(interpolation=ExtendedInterpolation())
pl_config.read('cfg/cfg.ini')

end_chap = pl_config['SCRAPER'].getint('end_chap') + 1
char_link_fp = pl_config['SCRAPER'].get('char_link_fp')
chap_appearance_fp = pl_config['SCRAPER'].get('chap_appearance_fp')
char_details_fp = pl_config['SCRAPER'].get('char_details_fp')

appearance_df = pd.read_csv(chap_appearance_fp)
# appearance_df['Chapter'] = appearance_df['Chapter'].ffill()
# df['Arc Name'] = df['Arc Name'].ffill()
all_dims = ['Chapter', 'Appearance', 'Arc', 'Character', 'Appearance Notes']
# preprocess to add arc 
arcs = generate_arc(end_chap)
# print(arcs)
# appearance_df['Character'] = appearance_df['Appearance'].str.split((",expand=True)[0]
appearance_df['Appearance'] = appearance_df['Character'].str.split("(",expand=True)[0]
appearance_df['Appearance Notes'] = appearance_df['Character'].str.split("(",expand=True)[1]
appearance_df['Appearance Notes'] = appearance_df['Appearance Notes'].str.replace(")", "", regex = True)
def arc_col(row):
    for key in arcs:
        if row['Chapter'] in arcs[key]:
            return key
    return "None"
appearance_df['Arc'] = appearance_df.apply(arc_col, axis =1) 

# print (appearance_df.head())

# fig_char = px.histogram(appearance_df, x='Chapter', barmode='group')
fig_char = px.histogram(appearance_df, x='Appearance', color = 'Arc', barmode='group')

# fig_ability = px.histogram(df_abilities, x='Arc Name', color="Ability Name", barmode='group')

app = dash.Dash(__name__)
server = app.server

app.layout = html.Div(
    children = [
                html.H1(children='One Dash'),
                dcc.Graph(id="histo", figure=fig_char),
                # dcc.Graph(id="histo_abilities", figure = fig_ability),
                ]
                    )

if __name__ == '__main__':
    app.run_server(debug=True)