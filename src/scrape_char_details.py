import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
from sqlite3 import Error


def create_connection(db_file):
    """ create a database connection to a SQLite database """
    conn = None
    conn = sqlite3.connect(db_file)
    if conn:
        conn.close()

def scrape_char_details(char_link_df, save_file_name):
    char_links = char_link_df['Link'].tolist()
    df = pd.DataFrame()    
    for char_link in char_links:
        try:
            URL = f'https://onepiece.fandom.com{char_link}'
            page = requests.get(URL)
            soup = BeautifulSoup(page.content, 'html.parser')
            table = soup.find('aside', {'role': 'region'} )
            
            name = table.find("h2", {"data-source": "name"}).text
            char_det_dict = {"Name": name}
            det_list = ['first','affiliation', 'occupation','residence', 'epithet','status', 'age', 'bounty', 'dfname']
            for det in det_list:
                if table.find("div", {"data-source": det}) is not None:
                    text_value = table.find("div", {"data-source": det}).find("div", {"class": "pi-data-value pi-font"}).text
                    if text_value is not None:
                        char_det_dict[det] = text_value 
                    else:
                        char_det_dict[det] = [i.get("title") for i in table.find("div", {"data-source": det}).find("div").find_all("a")]
            df = df.append(char_det_dict, ignore_index=True)
        except:
            print(char_link)
            continue
    df.to_csv(save_file_name, index=False)
    # print (char_det_dict)
        
if __name__ == '__main__':
    # dbname = r"data/OPdash.db"
    # create_connection(dbname)
    char_link_df = pd.read_csv('data/char_link.csv')
    scrape_char_details(char_link_df, save_file_name = "data/char_details.csv")
    