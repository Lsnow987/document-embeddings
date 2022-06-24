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
    text_array = full_text.split("@")
    # print(text_array)
    paragraph_count = 1
    title = ""
    paragraph_list = list()
    for paragraph in text_array:
        if paragraph_count == 1:
            title = paragraph.split("\r\n", 1)[:1]
            # print(title)
        p = read_paragraph(paragraph, paragraph_count, doc_name)
        paragraph_list.append(p)
        paragraph_count += 1
    all_docs = Document(title, doc_name, paragraph_list)
    pgraphs.append([s.to_dict() for s in paragraph_list])
    return all_docs


def read_paragraph(paragraph, paragraph_count, doc_name):
    # print("\n new line - paragraph \n")
    # print(paragraph)
    length = len(paragraph)
    p1 = Paragraph(paragraph, paragraph_count, doc_name, length)
    return p1


# which number paragraph of which doc
# text and make some embedding

pgraphs = pd.DataFrame(columns=['paragraph_num', 'doc_num', 'paragraph', 'length'])

arr = os.listdir("C:/Users/ysnow/OneDrive/Desktop/responsa_for_research/")
all_documents = list()
for x in arr:
    f = open("C:/Users/ysnow/OneDrive/Desktop/responsa_for_research/" + x, "rb")
    text = f.read()
    text = text.decode("cp1255")
    # print(text)
    # print("\n new line - main\n")
    doc = read_document(text, x)
    all_documents.append(doc)
    if x == "0001000":
        break

print(pgraphs.size)