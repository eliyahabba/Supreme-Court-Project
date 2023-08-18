import os
import datetime


def file_logger(start_time, end_time, filename, file_ind):
    print(
        f"file {file_ind} \n processed {filename}\n start time {start_time} end_time {end_time}")
    return


def filter_length(text, min_words):
    splitted_text = text.split(" ")
    return len(splitted_text) >= min_words


def save_file(results_dir, lemmatized_dir, filename, file_ind):
    """saving lemmatized text for the current filename in dir"""
    start_time = datetime.datetime.now()
    lemmatized_file = open(os.path.join(lemmatized_dir, filename), "r",
                encoding='utf8')
    lemmatized_text = lemmatized_file.read()
    lemmatized_file.close()
    file = open(os.path.join(results_dir, filename), "w",
                encoding='utf8')
    file.write(lemmatized_text)
    file.close()
    end_time = datetime.datetime.now()
    file_logger(start_time, end_time, filename, file_ind)
    return


def run(source_dir, lemmatized_dir, results_dir, min_words):
    done_text = list(os.listdir(results_dir))
    print(f"starting pipeline at {datetime.datetime.now()}")
    file_ind = 1
    for filename in os.listdir(source_dir):
        if filename in done_text:
            continue
        print(filename)
        text_file = open(os.path.join(source_dir, filename), "r",
                         encoding='utf8')
        text = text_file.read()
        text_file.close()
        if filter_length(text, min_words):
            print(f"{file_ind} file passed filter, saving")
            # we want to save a lemmatized version
            save_file(results_dir, lemmatized_dir, filename, file_ind)
        file_ind += 1
    print(f"ending pipeline at {datetime.datetime.now()}")
    return


def create_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return


if __name__ == "__main__":
    source_dir = input("insert source files path")
    lemmatized_dir = input("insert lemmatized files path")
    results_dir = input("insert results files path")
    min_words = int(input("insert minimum value of words"))
    create_dir(results_dir)
    run(source_dir, lemmatized_dir, results_dir, min_words)
