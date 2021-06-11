import re
import numpy as np
from src.arcs import generate_arc
import warnings
import pandas as pd
from configparser import ConfigParser, ExtendedInterpolation

warnings.filterwarnings("ignore")

def get_last_known_bounty(row):
    """get latest bounty for each character row
    """
    if type(row) == float:
        return row
    elif type(row) == str:
        x = re.sub(r"\[.*?\]", " ", row)
        x = x.split(" ")
        ret = ''.join([n for n in x[0] if n.isdigit()])
        if len(ret) ==0:
            return np.nan
        return int(ret)

def get_latest_age(row):
    if type(row) == str:
        x = re.sub(r"\[.*?\]", " ", row)
        x = re.sub(r"\(.*?\)", " ", x)
        x = x.replace(";", "")
        x = x.split(" ")
        
        ret = ' '.join([n for n in x if n.isdigit()])
        ret = ret.split(" ")
        newret = []
        for i in ret:
            try:
                newret.append(int(i))
            except:
                newret.append(i)

        return (max(newret))

def get_main_crew(row):
    if type(row) == str:
        x = re.sub(r"\[.*?\]", " ", row)
        x = re.sub(r"\(.*?\)", " ", x)
        x = x.split(";")
        # x = x.split("")
        return x[0]

class cleaner:
    def __init__(self, config_path = 'cfg/cfg.ini'):

        pl_config = ConfigParser(interpolation=ExtendedInterpolation())
        pl_config.read(config_path)

        self.end_chap = pl_config['SCRAPER'].getint('end_chap') + 1
        self.char_link_fp = pl_config['SCRAPER'].get('char_link_fp')
        self.chap_appearance_fp = pl_config['SCRAPER'].get('chap_appearance_fp')
        self.char_details_fp = pl_config['SCRAPER'].get('char_details_fp')
        self.age_bounty_fp = pl_config['SCRAPER'].get('age_bounty_fp')
        self.arcs = generate_arc(self.end_chap)
    
    def arc_col(self,row):
        """function to generate arc per row for appearance df
        """
        for key in self.arcs:
            if row['Chapter'] in self.arcs[key]:
                return key
        return "None"
    
    def preprocess_data(self):
        # preprocess to add arc 
        appearance_df = pd.read_csv(self.chap_appearance_fp)
        # appearance_df['Chapter'] = appearance_df['Chapter'].ffill()
        # df['Arc Name'] = df['Arc Name'].ffill()
        
        appearance_df['Appearance'] = appearance_df['Character'].str.split("(",expand=True)[0]
        appearance_df['Appearance Notes'] = appearance_df['Character'].str.split("(",expand=True)[1]
        appearance_df['Appearance Notes'] = appearance_df['Appearance Notes'].str.replace(")", "", regex = True)
        appearance_df['Arc'] = appearance_df.apply(self.arc_col, axis =1) 

        char_details_df = pd.read_csv(self.char_details_fp)
        char_details_df['last_bounty'] = char_details_df['bounty'].apply(get_last_known_bounty)
        char_details_df['latest_age'] = char_details_df['age'].apply(get_latest_age)
        char_details_df['latest_age']= char_details_df['latest_age'].fillna(value=np.nan)
        char_details_df['main_crew'] = char_details_df['affiliation'].apply(get_main_crew)
        df_age_bounty = char_details_df.dropna(subset=['latest_age', 'last_bounty'])
        df_age_bounty['latest_age'] = df_age_bounty['latest_age'].astype('int')

        appearance_df.to_csv(self.chap_appearance_fp, index = False)
        char_details_df.to_csv(self.char_details_fp, index = False)
        df_age_bounty.to_csv(self.age_bounty_fp, index = False)

if __name__ == '__main__':
    cleaner = cleaner()
    cleaner.preprocess_data()