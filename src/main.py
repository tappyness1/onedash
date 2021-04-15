from src.scrape_char_details import scrape_char_details
from src.scraper_chap_appearance import scrape_chap_appearances
from src.scraper_char_list import scrape_char_links

import argparse

import pandas as pd


def main_scraper(end_chap): 
    char_dict = {}
    update_char_dict = scrape_char_links(char_dict, end_chap = end_chap)
    df = pd.DataFrame.from_dict(char_dict, orient = 'index',columns=['Link'])
    df = df.reset_index()
    df.to_csv("data/char_link.csv")

    df = pd.read_csv("data/onedash_chap_appearance.csv")
    newdf = scrape_chap_appearances(df = df, end_chap = end_chap)
    newdf.to_csv("data/onedash_chap_appearance.csv", index=False)

    char_link_df = pd.read_csv('data/char_link.csv')
    scrape_char_details(char_link_df, save_file_name = "data/char_details.csv")

# dbname = r"data/OPdash.db"
# create_connection(dbname)
# char_link_df = pd.read_csv('data/char_link.csv')
# scrape_char_details(char_link_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--end_chap', default=1010, help='end of chapter')
    args = parser.parse_args()
    main_scraper(end_chap = args.end_chap)
