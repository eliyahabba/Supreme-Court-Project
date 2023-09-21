import re
import os
import pandas as pd

PSAK_REGEX = r'פ\s*ס\s*ק\s*-*\s*ד\s*י\s*ן\s*\n'


def extract_middle_content(text):
    if text is None:
        return ""
    try:
        pattern = re.compile(fr'{PSAK_REGEX}([\s\S]*)[\s-]*ניתן[\s-]*היום', re.DOTALL)
        match = pattern.search(text)
        if match:
            extracted_text = match.group(1).strip()
            extracted_percentage = len(extracted_text) / len(text) * 100
            if extracted_percentage == 0:
                return text
            return extracted_text
    except:
        return text

    try:
        # If the previous pattern fails, extract from the last 'פסק-דין' to the end
        pattern = re.compile(fr'.*{PSAK_REGEX}([\s\S]*)', re.DOTALL)
        match = pattern.search(text)
        if match:
            extracted_text = match.group(1).strip()
            extracted_percentage = len(extracted_text) / len(text) * 100
            if extracted_percentage == 0:
                return text
            return extracted_text
    except:
        return text

    try:
        # If there's no 'פסק-דין', extract from the beginning to 'ניתן היום'
        pattern = re.compile(r'^([\s\S]*)(ניתן[\s-]*היום)', re.DOTALL)
        match = pattern.search(text)
        if match:
            extracted_text = match.group(1).strip()
            extracted_percentage = len(extracted_text) / len(text) * 100
            if extracted_percentage == 0:
                return text
            return extracted_text
    except:
        return text

    # If there's no 'פסק-דין', extract from the beginning
    return text


def get_documents_content(directory_path):
    data = []
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        with open(file_path, 'r') as file:
            try:
                data.append(file.read())
            except:
                print(file_path)
    df = pd.DataFrame(data, columns=['text'])
    df['extracted_content'] = df['text'].apply(lambda x: extract_middle_content(x))
    return df


if __name__ == '__main__':
    directory_path = "/mnt/local/mikehash/Data/Nevo/NevoVerdicts"
    df = get_documents_content(directory_path)
