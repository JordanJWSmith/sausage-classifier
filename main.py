# from web_scrape import filepath_exists, generate_dataset
# from train import run_training
# import argparse
# import torch
# import torch.nn as nn
# import torch.optim as optim
import requests
from bs4 import BeautifulSoup
import urllib.request
import random


# from model import CNNModel
#
#
# if __name__ == '__main__':
#
#     if not filepath_exists():
#         print('filepath does not exist')
#         generate_dataset()
#
#     run_training()

# print('Please run train.py')

queries = ['sausage']
user_agent = 'Mozilla/5.0 (Linux; Android 10; SM-A205U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.5249.126 Mobile Safari/537.36'

for query in queries:
    print(f'downloading {query} images')
    google_url = f"https://www.google.com/search?q={query}&sxsrf=ALeKk03xBalIZi7BAzyIRw8R4_KrIEYONg:1620885765119&" \
          f"source=lnms&tbm=isch&sa=X&ved=2ahUKEwjv44CC_sXwAhUZyjgGHSgdAQ8Q_AUoAXoECAEQAw&cshid=1620885828054361"
    google_page = requests.get(google_url)
    google_soup = BeautifulSoup(google_page.content, 'html.parser')
    image_tags = google_soup.find_all('img')

    ss_url = f"https://www.shutterstock.com/search?searchterm={query}"
    ss_page = requests.get(ss_url, headers={'User-Agent': user_agent})
    ss_soup = BeautifulSoup(ss_page.content, 'html.parser')
    image_tags += ss_soup.find_all('img')

    random.shuffle(image_tags)

    img_errors = 0

    print('attempting to source', len(image_tags), 'images')
    for i, link in enumerate(image_tags):
        if link['src'][-4:] != '.gif':
            try:
                urllib.request.urlretrieve(
                    link['src'],
                    f"input/test/shutterstock_{query.replace(' ', '_')}_{i}.jpg"
                )
            except Exception as e:
                img_errors += 1
                # print('Could not source', i, link)
                # print('error:', e)
                # print()
            # print(link['src'])

    if img_errors:
        print('Count not source ', img_errors, query, 'images')
