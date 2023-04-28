import os
import time
from gensim.corpora import Dictionary
from gensim.models import LdaModel
import logging
import pandas as pd
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def split_text(text):
    return text.split(" ")


def remove_rare_and_common_words(docs, no_below, no_above):
    """using gensim Dictionary object in order to filter common + rare words"""
    dictionary = Dictionary(docs)
    dictionary.filter_extremes(no_below=no_below, no_above=no_above)
    return dictionary


def create_bag_of_words(dictionary, docs):
    """creating bag of words out of filtered Dictionary gensim object"""
    corpus = [dictionary.doc2bow(doc) for doc in docs]
    return corpus


def run_lda_pipeline(base_dir, lemmatize_dir, results_dir):
    docs = []
    for filename in os.listdir(lemmatize_dir):
        text_file = open(os.path.join(lemmatize_dir, filename), "r",
                         encoding='utf8')
        text = text_file.read()
        text_file.close()
        splitted_text = split_text(text)
        docs.append(splitted_text)

    dictionary = remove_rare_and_common_words(docs, 100, 0.5)
    corpus = create_bag_of_words(dictionary, docs)

    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    # to modify hyper-parameters
    num_topics = 13
    chunksize = 2000
    passes = 30
    iterations = 400
    eval_every = None
    alpha = 0.6
    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        chunksize=chunksize,
        alpha=alpha,
        eta='auto',
        iterations=iterations,
        num_topics=num_topics,
        passes=passes,
        eval_every=eval_every
    )
    save_dir = os.path.join(results_dir, f'{num_topics}_topics_{round(time.time(),0)}_{alpha}_alpha.xlsx')
    pd.DataFrame(model.print_topics(num_topics=-1)).to_excel(save_dir)


if __name__ == "__main__":
    base_dir = r"C:\Users\noabi\PycharmProjects\University"
    lemmatize_dir = os.path.join(base_dir, r"TrankitLemmatizedTextTest")
    results_dir = os.path.join(base_dir, r"TrankitLdaResults")
    run_lda_pipeline(base_dir, lemmatize_dir, results_dir)
