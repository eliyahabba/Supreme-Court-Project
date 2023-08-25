from lda import *
import itertools
import numpy as np
import datetime
import os
import pandas as pd
import time
import sys
import json


def alpha_grid(min_alpha=0.0, max_alpha=1.0, step=0.1):
    return np.arange(min_alpha, max_alpha, step)


def passes_grid(min_passes=5, max_passes=40, step=5):
    return np.arange(min_passes, max_passes, step)


def topics_grid(min_topics=10, max_topics=60, step=5):
    return np.arange(min_topics, max_topics, step)


def grid_params_lists(alpha_grid, eta_grid, passes_grid, topics_grid, no_below_dict,
                      no_above_dict, chunksize, iterations):
    return list(
        itertools.product(alpha_grid, eta_grid, passes_grid, topics_grid, no_below_dict,
                          no_above_dict, chunksize, iterations))


def run_model_with_grid_search(results_dir, docs,
                               filenames_lst,
                               corpus, dictionary, params_list):
    model_ind = 1
    models_df = []
    print(f"number of models to run: {len(params_list)}")
    for params in params_list:
        params_dict = {"alpha": params[0],
                       "eta" : params[1],
                       "passes": params[2],
                       "num_topics": params[3],
                       "no_below_int": params[4],
                       "no_above_percent": params[5],
                       "chunksize": params[6],
                       "iterations": params[7],
                       "eval_every": None,
                       }
        print(
            f"model num {model_ind} with params {params_dict} start: {datetime.datetime.now()}\n")
        bow, save_model_dir, scores, identifier, = run_lda_pipeline(
            results_dir,
            docs,
            filenames_lst,
            dictionary,
            corpus,
            params_dict)
        params_dict["cv_score"] = scores["cv"]
        params_dict["u_mass_score"] = scores["u_mass"]
        params_dict["c_uci_score"] = scores["c_uci"]
        params_dict["identifier"] = identifier
        models_df.append(params_dict)
        print(f"model num {model_ind} finished at: {datetime.datetime.now()}")
        model_ind += 1

    pd.DataFrame(models_df).to_csv(
        os.path.join(results_dir, f"{int(time.time())}_grid_search.csv"))


if __name__ == "__main__":
    convert_to_float_keys = ["alpha_min", "alpha_max", "no_above_p"]
    convert_to_int_keys = ["passes_min", "passes_max", "topics_min",
                           "topics_max", "no_below_int", "chunksize",
                           "iterations"]
    arguments = sys.argv
    data = None
    if len(arguments)>1:
        data = json.loads(arguments[1])
        # converting keys
        for key, val in data.items():
            if key in convert_to_float_keys:
                data[key] = float(val)
            elif key in convert_to_int_keys:
                data[key] = int(val)
    if data:
        alphas = alpha_grid(data["alpha_min"], data["alpha_max"])
        passes = passes_grid(data["passes_min"], data["passes_max"])
        topics = topics_grid(data["topics_min"], data["topics_max"])
    else:
        no_below_int = int(input("no below int"))
        no_above_p = float(input("no above p"))
        chunksize = int(input("chunksize"))
        iterations = int(input("iterations"))
        alpha_min = float(input("alpha min"))
        alpha_max = float(input("alpha max"))
        eta_min = float(input("eta min"))
        eta_max = float(input("eta max"))
        passes_min = int(input("passes min"))
        passes_max = int(input("passes max"))
        topics_min = int(input("topics min"))
        topics_max = int(input("topics max"))
        alpha_auto = input("is alpha auto? Y or N")
        eta_auto = input("is eta auto? Y or N")

        if alpha_auto == "Y":
            alphas = ["auto"]
        else:
            alphas = alpha_grid(alpha_min, alpha_max)
        if eta_auto == "Y":
            etas = ["auto"]
        else:
            etas = alpha_grid(eta_min, eta_max)

        passes = passes_grid(passes_min, passes_max)
        topics = topics_grid(topics_min, topics_max)
        data = {"no_below_int": no_below_int, "no_above_p": no_above_p,
                "chunksize": chunksize, "iterations": iterations}

    lemmatize_dir = input("insert full lemmatized file dir")
    results_dir = input("insert full results dir (already exist)")

    dict_params = [data["no_below_int"], data["no_above_p"]]
    chunksize_params = [data["chunksize"]]
    iterations_params = [data["iterations"]]

    params_list = grid_params_lists(alphas, etas, passes, topics,
                                    [data["no_below_int"]],
                                    [data["no_above_p"]], [data["chunksize"]],
                                    [data["iterations"]])

    print("creating docs")
    docs, filenames_lst = create_docs(lemmatize_dir)
    print("creating dictionary object")
    dictionary = create_dictionary(docs, data["no_below_int"],
                                   data["no_above_p"])
    print("creating bag of words")
    corpus = create_bag_of_words(dictionary, docs)
    print("running models")
    run_model_with_grid_search(results_dir, docs,
                               filenames_lst, corpus, dictionary, params_list)
