import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_char_links(char_dict, start_chap = 1, end_chap =5000, continue_last = True):
    # if continue_last:
    #     curr_chapts = df['Chapter'].tolist()
    # else: curr_chapts = []
    for i in range(start_chap, end_chap):
        # if i in curr_chapts:
        #     continue
        # else:
        if i % 100 == 0:
            print (i)
        # char_list = []
        URL = f'https://onepiece.fandom.com/wiki/Chapter_{i}'
        page = requests.get(URL)

        soup = BeautifulSoup(page.content, 'html.parser')

        table = soup.find('table', class_='CharTable')
        
        for elem in table.find_all('li'):
            try:
                # char_list.append(elem.text)
                if elem.find('a').get('title') in char_dict:
                    continue
                else:
                    char_dict[elem.find('a').get('title')] = elem.find('a').get('href')
            except :
                continue
    return char_dict