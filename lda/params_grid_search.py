from lda import *
import itertools
import numpy as np
import datetime
import os
import pandas as pd
import time


def alpha_grid(min_alpha=0.0, max_alpha=1.0, step=0.1):
    return np.arange(min_alpha, max_alpha, step)


def passes_grid(min_passes=5, max_passes=40, step=5):
    return np.arange(min_passes, max_passes, step)


def grid_params_lists(alpha_grid, passes_grid):
    return list(itertools.product(alpha_grid, passes_grid))


def run_model_with_grid_search(results_dir, docs,
                               filenames_lst,
                               corpus, dictionary, params_list):
    model_ind = 1
    models_df = []
    for params in params_list:
        params_dict = {"num_topics": 15,
                       "chunksize": 2000,
                       "alpha": params[0],
                       "passes": params[1],
                       "iterations": 400,
                       "eval_every": None,
                       }
        print(
            f"model num {model_ind} with params {params_dict} start: {datetime.datetime.now()}\n")
        bow, save_model_dir, score, identifier, = run_lda_pipeline(results_dir,
                                                                   docs,
                                                                   filenames_lst,
                                                                   dictionary,
                                                                   corpus,
                                                                   params_dict)
        params_dict["cohernce_score"] = score
        params_dict["identifier"] = identifier
        models_df.append(params_dict)
        print(f"model num {model_ind} finished at: {datetime.datetime.now()}")
        model_ind += 1

    pd.DataFrame(models_df).to_csv(
        os.path.join(results_dir, f"{int(time.time())}_grid_search.csv"))


if __name__ == "__main__":
    base_dir = "/mnt/local/mikehash/Data"
    lemmatize_dir = os.path.join(base_dir, r"LemmatizedText")
    results_dir = os.path.join(base_dir, r"YapLdaResults")
    alphas = alpha_grid(0.1, 1)
    passes = passes_grid(20, 30)
    params_list = grid_params_lists(alphas, passes)
    print("creating docs")
    docs, filenames_lst = create_docs(lemmatize_dir)
    print("creating dictionary object")
    dictionary = create_dictionary(docs, 200, 0.5)
    print("creating bag of words")
    corpus = create_bag_of_words(dictionary, docs)
    print("running models")
    run_model_with_grid_search(results_dir, docs,
                               filenames_lst, corpus, dictionary, params_list)
