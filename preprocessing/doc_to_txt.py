import os
import win32com.client
import shutil

FILE_PATH = r"C:\Users\Micha\Documents\HUJI\FinalProject\Nevo"
OUTPUT_FILE_PATH = r"C:\Users\Micha\Documents\HUJI\FinalProject\NevoTxt"
PROBLEM_OUTPUT_PATH = r"C:\Users\Micha\Documents\HUJI\FinalProject\NevoProblem"


def convert_doc_to_text(docfile):
    try:
        doc = win32com.client.GetObject(os.path.join(FILE_PATH, docfile))
        text = doc.Range().Text
        # print(len(text.split(" ")))

        with open(os.path.join(OUTPUT_FILE_PATH, os.path.splitext(docfile)[0] + ".txt"), "wb") as f:
            f.write(text.encode("utf-8"))
    except:
        shutil.copy(os.path.join(FILE_PATH, docfile), os.path.join(PROBLEM_OUTPUT_PATH, docfile))


for i, docfile in enumerate(os.listdir(FILE_PATH)):
    print(i)
    convert_doc_to_text(docfile)
