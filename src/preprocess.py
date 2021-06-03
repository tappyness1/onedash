import re
import numpy as np

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