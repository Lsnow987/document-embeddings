import os
import pandas as pd


class Document:
    def __init__(self, title, doc_num, paragraphs):
        self.title = title
        self.doc_num = doc_num
        self.paragraphs = paragraphs


class Paragraph:
    def __init__(self, paragraph, paragraph_num, doc_num, length):  # need this second docnum
        self.paragraph_num = paragraph_num
        self.doc_num = doc_num
        self.paragraph = paragraph
        self.length = length

    def to_dict(self):
        return {
            'paragraph_num': self.paragraph_num,
            'doc_num': self.doc_num,
            'paragraph': self.paragraph,
            'length': self.length,
        }


def read_document(full_text, doc_name):
    # pgraphs = pd.DataFrame(columns=['paragraph_num', 'doc_num', 'paragraph', 'length'])
    text_array = full_text.split("@")
    # print(text_array)
    paragraph_count = 1
    title = ""
    paragraph_list = list()
    for paragraph in text_array:
        if paragraph_count == 1:
            title = paragraph.split("\r\n", 1)[:1]
            paragraph = paragraph.split("\r\n", 1)[1:]
        # if len(paragraph) > 5:
        p = read_paragraph(paragraph, paragraph_count, doc_name)
        paragraph_list.append(p)
        paragraph_count += 1
    docu = Document(title, doc_name, paragraph_list)
    temp = pd.concat([pd.Series(s.to_dict()) for s in paragraph_list], axis=1)
    informat = [docu, temp]
    return informat


def read_paragraph(paragraph, paragraph_count, doc_name):
    length = len(paragraph)
    p1 = Paragraph(paragraph, paragraph_count, doc_name, length)
    return p1

count = 1
arr = os.listdir("/home/jacob/code/responaProjectReccomender/Data")
all_documents = list()
full_graph = pd.DataFrame(columns=['paragraph_num', 'doc_num', 'paragraph', 'length'])
for x in arr:
    f = open("/home/jacob/code/responaProjectReccomender/Data/" + x, "rb")
    text = f.read()
    text = text.decode("cp1255", errors="ignore")
    info = read_document(text, x)
    doc = info[0]
    pgraph = info[1]
    all_documents.append(doc)
    full_graph = pd.concat([pgraph, full_graph], axis=1)
    if count == 1000:
        break
    count = count + 1

print(full_graph)
full_graph.to_csv("ex.csv")
