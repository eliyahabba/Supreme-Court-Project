from lda.lda import *
import itertools
import numpy as np
import datetime


def alpha_grid(min_alpha=0, max_alpha=1, step=0.1):
    return np.arange(min_alpha, max_alpha, step)


def passes_grid(min_passes=5, max_passes=40, step=5):
    return np.arrange(min_passes, max_passes, step)


def grid_params_lists(alpha_grid, passes_grid):
    return list(itertools.product(alpha_grid, passes_grid))


def run_model_with_grid_search(base_dir, lemmatize_dir, results_dir,
                               params_list):
    model_ind = 1
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
        run_lda_pipeline(base_dir, lemmatize_dir, results_dir, params_dict)
        print(f"model num {model_ind} finished at: {datetime.datetime.now()}")
        model_ind += 1


if __name__ == "__main__":
    base_dir = r"C:\Users\noabi\PycharmProjects\University"
    lemmatize_dir = os.path.join(base_dir, r"LdaReadyText")
    results_dir = os.path.join(base_dir, r"YapLdaResults")
    alphas = alpha_grid()
    passes = passes_grid()
    params = grid_params_lists(alphas, passes)
    run_model_with_grid_search(base_dir, lemmatize_dir, results_dir, params)
