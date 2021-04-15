import requests
from bs4 import BeautifulSoup
import pandas as pd

def scrape_chap_appearances(df, start_chap = 1, end_chap =5000, continue_last = True):
    if continue_last:
        curr_chapts = df['Chapter'].tolist()
    else: curr_chapts = []
    for i in range(start_chap, end_chap + 1):
        if i in curr_chapts:
            continue
        else:
            if i % 100 == 0:
                print (i)
            # char_list = []
            URL = f'https://onepiece.fandom.com/wiki/Chapter_{i}'
            page = requests.get(URL)

            soup = BeautifulSoup(page.content, 'html.parser')

            table = soup.find('table', class_='CharTable')

            for elem in table.find_all('li'):
                # char_list.append(elem.text)
                df = df.append({'Chapter': i, 'Appearance': elem.text}, ignore_index=True)
    return df
            # appearance_dict[i] = char_list

# if __name__ == '__main__':
#     df = pd.read_csv("data/onedash_chap_appearance.csv")
#     newdf = scrape_chap_appearances(df = df, end_chap = 1006)
#     newdf.to_csv("data/onedash_chap_appearance.csv", index=False)