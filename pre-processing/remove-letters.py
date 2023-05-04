import os
import datetime


def file_logger(start_time, end_time, filename, file_ind):
    print(
        f"file {file_ind} \n processed {filename}\n start time {start_time} end_time {end_time}")
    return


def remove_letters(text):
    splitted_text = text.split(" ")
    return " ".join(word for word in splitted_text if len(word) > 1)


def save_file(results_dir, filename, text, file_ind):
    """saving pre-processed text for the current filename in dir"""
    start_time = datetime.datetime.now()
    file = open(os.path.join(results_dir, filename), "w",
                encoding='utf8')
    file.write(text)
    file.close()
    end_time = datetime.datetime.now()
    file_logger(start_time, end_time, filename, file_ind)
    return


def run(source_dir, results_dir):
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
        text = remove_letters(text)
        save_file(results_dir, filename, text, file_ind)
        file_ind += 1
    print(f"ending pipeline at {datetime.datetime.now()}")
    return


def create_dir(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return


if __name__ == "__main__":
    base_dir = r"C:\Users\noabi\PycharmProjects\University"
    source_dir = os.path.join(base_dir, r"LemmatizedText")
    results_dir = os.path.join(base_dir, r"LdaReadyText")
    create_dir(results_dir)
    run(source_dir, results_dir)
