from datasets import Dataset
from huggingface_hub import login
import os
import argparse

print (os.environ.get('HF_TOKEN'))
login(token = os.environ.get('HF_TOKEN'))

# Create a Dataset object from your CSV file
dataset = Dataset.from_csv('./data/age_bounty.csv')
char_details_dataset = Dataset.from_csv('./data/char_details.csv')
char_link_dataset = Dataset.from_csv('./data/char_link.csv')
chap_appearance_dataset = Dataset.from_csv('./data/onedash_chap_appearance.csv')

dataset.push_to_hub("tappyness1/one_dash")
char_details_dataset.push_to_hub("tappyness1/one_dash")
char_link_dataset.push_to_hub("tappyness1/one_dash")
chap_appearance_dataset.push_to_hub("tappyness1/one_dash")