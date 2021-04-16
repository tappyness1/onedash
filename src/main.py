from src.scrape_char_details import scrape_char_details
from src.scraper_chap_appearance import scrape_chap_appearances
from src.scraper_char_list import scrape_char_links
from configparser import ConfigParser, ExtendedInterpolation
import argparse

import pandas as pd
import os.path

def main_scraper(path_to_config): 
    pl_config = ConfigParser(interpolation=ExtendedInterpolation())
    pl_config.read(path_to_config)

    end_chap = pl_config['SCRAPER'].getint('end_chap') + 1
    char_link_fp = pl_config['SCRAPER'].get('char_link_fp')
    chap_appearance_fp = pl_config['SCRAPER'].get('chap_appearance_fp')
    char_details_fp = pl_config['SCRAPER'].get('char_details_fp')

    if pl_config['SCRAPER'].getboolean('char_link'):
        print ("scraping char links")
        char_dict = {}
        scrape_char_links(char_dict, end_chap = end_chap)
        df = pd.DataFrame.from_dict(char_dict, orient = 'index',columns=['Link'])
        df = df.reset_index()
        df.to_csv(char_link_fp, index = False)

    if pl_config['SCRAPER'].getboolean('chap_appearance'):
        print ("scraping char appearance")
        if os.path.exists(chap_appearance_fp):
            df = pd.read_csv(chap_appearance_fp)
        else:
            df = None
        newdf = scrape_chap_appearances(df = df, end_chap = end_chap)
        newdf.to_csv(chap_appearance_fp, index=False)

    if pl_config['SCRAPER'].getboolean('char_details'):
        print ("scraping character details")
        char_link_df = pd.read_csv(char_link_fp)
        scrape_char_details(char_link_df, save_file_name = char_details_fp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_config', default='cfg/cfg.ini', help='path to config file')
    args = parser.parse_args()
    main_scraper(path_to_config = args.path_to_config)
