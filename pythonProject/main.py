import os
import pandas as pd
from os.path import exists
from transformers import BertModel, BertTokenizerFast
import torch


# https://www.geeksforgeeks.org/priority-queue-in-python/
class PriorityQueue(object):
    size = 0

    def __init__(self):
        self.size = 0
        self.queue = []

    def __str__(self):
        return ' '.join([str(i) for i in self.queue])

    # for checking if the queue is empty
    def isEmpty(self):
        return len(self.queue) == 0

    # for inserting an element in the queue
    def insert(self, data):
        self.size += 1
        self.queue.append(data)

    def get_size(self):
        return self.size

    # for popping an element based on Priority
    def delete(self):
        try:
            max_val = 0
            for i in range(len(self.queue)):
                if self.queue[i] > self.queue[max_val]:
                    max_val = i
            item = self.queue[max_val]
            del self.queue[max_val]
            self.size -= 1
            return item
        except IndexError:
            print()
            exit()


class Document:
    def __init__(self, title, doc_num, paragraphs, length):
        self.title = title
        self.doc_num = doc_num
        self.paragraphs = paragraphs
        self.length = length

    def to_dict(self):
        return {
            'title': self.title,
            'doc_num': self.doc_num,
            'paragraphs': str(self.paragraphs),
            'length': self.length,
        }


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

    def __str__(self):
        return str({
            'paragraph_num': self.paragraph_num,
            'doc_num': self.doc_num,
            'paragraph': self.paragraph,
            'length': self.length,
        })


def read_document(full_text, doc_name):
    # pgraphs = pd.DataFrame(columns=['paragraph_num', 'doc_num', 'paragraph', 'length'])
    length_of_doc = len(full_text)
    text_array = full_text.split("@")
    # print(text_array)
    paragraph_count = 1
    title = ""
    paragraph_list = list()
    for paragraph in text_array:
        if paragraph_count == 1:
            title = paragraph.split("\r\n", 1)[:1]
            paragraph = paragraph.split("\r\n", 1)[1:]
        else:
            paragraph = paragraph.split("\r\n", 1)[:]
            # leng = len(paragraph[0])
        if len(paragraph[0]) > 5:  # what number should this be to take out small paragraphs that don't mean anything
            p = read_paragraph(paragraph, paragraph_count, doc_name)
            paragraph_list.append(p)
        paragraph_count += 1
    docu = Document(title, doc_name, paragraph_list, length_of_doc)

    temp = paragraph_list  # pd.concat([pd.Series(s.to_dict()) for s in paragraph_list], axis=1)
    informat = [docu, temp]
    return informat


def read_paragraph(paragraph, paragraph_count, doc_name):
    length = len(paragraph[0])
    p1 = Paragraph(paragraph, paragraph_count, doc_name, length)
    return p1


class Manager:
    df = None

    def __init__(self):
        file = 'C:/Users/ysnow/OneDrive/Desktop/summer/pythonProject/paragraphs.csv'
        self.df = pd.read_csv(file)
        # file_exists = exists("C:/Users/ysnow/OneDrive/Desktop/summer/pythonProject/docs.csv")
        # df = None
        # if file_exists:
        #     df = pd.read_csv('C:/Users/ysnow/OneDrive/Desktop/summer/pythonProject/docs.csv')
        # else:
        #     df = "a"

        # id = which number model this is
        # name = the name of the model

    def make_paragraph_and_doc_csv(self):
        arr = os.listdir("/home/jacob/code/responaProjectReccomender/Data/")
        all_documents = list()

        count = 1

        full_graph = pd.DataFrame(columns=['paragraph_num', 'doc_num', 'paragraph', 'length'])
        doc_full_graph = pd.DataFrame(columns=['title', 'doc_num', 'paragraphs', 'length'])
        all_the_docs = []
        all_the_paragraphs = []
        for x in arr:
            f = open("/home/jacob/code/responaProjectReccomender/Data/" + x, "rb")
            text = f.read()
            text = text.decode("cp1255", errors="ignore")
            info = read_document(text, x)
            doc = info[0]
            all_the_docs.append(doc)
            all_the_paragraphs.append(info[1])

            # doc_pgraph = None
            all_documents.append(doc)
            if count == 1000:
                break
            count = count + 1

        # full_graph = pd.concat([pgraph, full_graph], axis=1)

        doc_pgraph = pd.concat([pd.Series(s.to_dict()) for s in all_the_docs], axis=1)
        pgraph = pd.concat([pd.Series(a.to_dict()) for s in all_the_paragraphs for a in s], axis=1)
        # full_graph = pd.concat([pgraph, full_graph], axis=1)
        # doc_full_graph = pd.concat([doc_pgraph, doc_full_graph], axis=1)
        # print(doc_full_graph)
        # doc_full_graph.to_csv("docs.csv")
        # full_graph.to_csv("paragraphs.csv")

        doc_pgraph.to_csv("docs.csv")

        pgraph.to_csv("paragraphs.csv")

    def addModel(self, name):
        return 0

        # id = which number paragraph this is in relation to all of the paragraphs
        # document is which document is this a part of
        # text is the text of the document

    # def addParagraphs(self, id, document, text):
    #     return 0

    def addEmbedings(self, modelID, pagraphID, embeding):
        alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
        alephbert = BertModel.from_pretrained('onlplab/alephbert-base')

        # if not finetuning - disable dropout
        alephbert.eval()

        same_topic_list = []
        inputs = alephbert_tokenizer("אבא לבית", return_tensors="pt")
        inputs1 = alephbert_tokenizer("אמא לבית", return_tensors="pt")
        inputs2 = alephbert_tokenizer("לבית אדם", return_tensors="pt")

        outputs = alephbert(**inputs)
        outputs1 = alephbert(**inputs1)
        outputs2 = alephbert(**inputs2)

        logits = outputs.last_hidden_state
        logits1 = outputs1.last_hidden_state
        logits2 = outputs2.last_hidden_state

        print(logits.size())
        print(logits1.size())
        print(logits2.size())

        return 0

    def search(self, paragraphID, modelID, count):
        # list of closet paragraph ID using a particular model
        length = len(self.df)
        num_rows = int(self.df.size / length)
        # all_embeddings = []
        modelID_row = self.get_modelID_row(modelID)
        paragraph_embedding = self.df.values[modelID_row][
            paragraphID]  # self.get_paragraph_embedding(paragraphID, modelID)

        return self.closest_embeddings(count, paragraph_embedding, modelID_row, num_rows, paragraphID)

    def get_modelID_row(self, modelID):  # access row by name
        modelID_row = None
        for i in range(len(self.df)):
            current_id = self.df.values[i][0]
            if modelID == current_id:
                modelID_row = i
                break
        return modelID_row

    # count = how many he wants that are close topically
    def closest_embeddings(self, count, paragraph_embedding, modelID_row, num_rows, paragraphID):
        same_topic_list = PriorityQueue()
        # for logit in all_embeddings:
        for i in range(num_rows):
            if i != paragraphID:
                logit = self.df.values[modelID_row][i]
                difference = logit - paragraph_embedding
                distance = torch.linalg.norm(torch.linalg.norm(difference, dim=2, ord=2))
                if same_topic_list.get_size() <= count:
                    same_topic_list.insert(logit)
                else:
                    original_embedding = same_topic_list.delete()
                    if original_embedding < logit:
                        same_topic_list.insert(original_embedding)
                    else:
                        same_topic_list.insert(logit)
        return same_topic_list

    # return first 2500 characters not in middle of word of of text


n = Manager()
n.search(None, None, None)
