# inspo: https://python.plainenglish.io/how-to-scrape-images-using-beautifulsoup4-in-python-e7a4ddb904b8

import requests
from bs4 import BeautifulSoup
import os
import urllib.request
import json


def input_filepath_exists():
    return os.path.exists('input/train/') and os.path.exists('input/valid/')


def build_filepath(class_name):
    train_path = f'input/train/{class_name}/'
    valid_path = f'input/valid/{class_name}/'
    output_path = 'outputs/'
    for path in [train_path, valid_path, output_path]:
        if not os.path.exists(path):
            print(f'Building filepath {path}')
            os.makedirs(path)


def scrape_images(class_name, queries):
    build_filepath(class_name)
    for query in queries:
        print(f'downloading {query} images')
        url = f"https://www.google.com/search?q={query}&sxsrf=ALeKk03xBalIZi7BAzyIRw8R4_KrIEYONg:1620885765119&" \
              f"source=lnms&tbm=isch&sa=X&ved=2ahUKEwjv44CC_sXwAhUZyjgGHSgdAQ8Q_AUoAXoECAEQAw&cshid=1620885828054361"
        page = requests.get(url)

        soup = BeautifulSoup(page.content, 'html.parser')
        image_tags = soup.find_all('img')

        train_split = int(len(image_tags) * 0.8)

        error_counter = 0

        for i, link in enumerate(image_tags):
            # test script
            if i > 0:
                try:
                    if i <= train_split - 1:
                        urllib.request.urlretrieve(
                            link['src'],
                            f"input/train/{class_name}/{query.replace(' ', '_')}_{i}.jpg"
                            )
                    else:
                        urllib.request.urlretrieve(
                            link['src'],
                            f"input/valid/{class_name}/{query.replace(' ', '_')}_{i}.jpg"
                            )
                except:
                    error_counter += 1

        print(f'Saved {len(image_tags) - error_counter} {query} images')


def generate_dataset():
    # Add image queries to image_queries.json
    # Each query currently generates ~20 images

    with open('image_queries.json', 'r') as f:
        queries_json = json.load(f)

    for class_name, queries in queries_json.items():
        scrape_images(class_name, queries)
