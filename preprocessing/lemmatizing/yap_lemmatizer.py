import pandas as pd
import os
import sys
import datetime

# specify the directory that contains the YAP wrapper, download from here:
# https://github.com/amit-shkolnik/YAP-Wrapper
BASE_YAP_DIR = '/mnt/local/mikehash/YAP-Wrapper-master'

sys.path.insert(0, BASE_YAP_DIR)
from yap_api import *


def create_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)


def call_yap(yap, ip, filename):
    time_start = datetime.datetime.now()

    name = filename.replace('.txt', '')
    text_file = open(os.path.join(text_dir, filename), "r", encoding= 'utf8')
    text = text_file.read()

    tokenized_text, segmented_text, lemmas, dep_tree, md_lattice, ma_lattice = yap.run(text, ip)

    lemmatizer_file = open(os.path.join(lemmatize_dir, filename), "w", encoding= 'utf8')
    lemmatizer_file.write(lemmas)
    lemmatizer_file.close()

    time_end = datetime.datetime.now()

    text_file.close()

    # return len(text.split(" ")), len(lemmas.split(" ")), (time_end - time_start).seconds

def run_yap_lemmatizer(text_dir, lemmatize_dir, ip):
    create_dir(lemmatize_dir)

    yap = YapApi()
    done_text = list(os.listdir(lemmatize_dir))

    for filename in os.listdir(text_dir):
        print(filename)
        if not filename.endswith('.txt') or filename in done_text:
            continue
        call_yap(yap, ip, filename)


if __name__ == '__main__':
    base_dir = "/mnt/local/mikehash/Data"
    text_dir = os.path.join(base_dir, "NevoTxt")
    lemmatize_dir = os.path.join(base_dir, r"LemmatizedText")

    # specify the ip where the yap API is deployed
    ip = 'localhost:8000'

    run_yap_lemmatizer(text_dir, lemmatize_dir, ip)


