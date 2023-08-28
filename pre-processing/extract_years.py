import os
import re
import pandas as pd
import time


def find_years(text):
    # Regular expression patterns for different date formats
    patterns = [
        #         r'\b(\d{4})\b',          # yyyy format
        r'\b\d{1,2}/\d{1,2}/(\d{2}|\d{4})\b',
        r'\b\d{1,2}\.\d{1,2}\.(\d{2}|\d{4})\b']
    years = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            year = match
            if len(year) == 2:
                if 0 <= int(year) <= 23:
                    year = '20' + year
                else:
                    year = '19' + year
            years.append(year)

    if not years:
        matches = re.findall(r'\b(\d{4})\b', text)
        years.extend(matches)
    return years


def display_text(data, i, col):
    print(data[col].iloc[i].replace('\n\n', ''))
    print(data['Years'].iloc[i])


def max_year(years):
    years = [int(year) for year in years.split(', ') if year != '' and 1900 <= int(year) <= 2023]
    if not years:
        return '', 0
    return str(max(years)), len(set(years))


def run(directory_path):
    data = {'Text': [], 'filename': [], 'Years': []}
    for filename in os.listdir(directory_path):
        #     if filename.endswith('.txt'):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r', encoding='utf8') as file:
            try:
                text = file.read()
                years = find_years(text)
                data['filename'].append(filename)
                #data['Text'].append(text[:200]) # sample
                data['Years'].append(', '.join(years))
            except:
                print(file_path)
    return pd.DataFrame(data)


if __name__ == "__main__":
    source_dir = input("insert source files path to extract years from")
    results_dir = input("insert results files path")
    df = run(source_dir)
    results_path_name = os.path.join(results_dir, f"extract_years_{int(time.time())}.xlsx")
    df.to_excel(results_path_name)
