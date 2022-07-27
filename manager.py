from typing import Collection
import pandas as pd
import os
import numpy as np
from transformers import BertModel, BertTokenizerFast, AutoModelForMaskedLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModel
import torch

pd.options.mode.chained_assignment = None  # default='warn'


class Manager:
    # Main Dataframe
    paragraphs_df = None

    # Constructor - no filepath
    def __init__(self):
        self.paragraphs_df = pd.DataFrame(columns=['paragraph_id', 'paragraph_text', 'document_id', 'length']).astype(
            'object')

    def loadCSV(self, filePath):
        self.paragraphs_df = pd.read_csv(filePath, index_col=0)
        # self.paragraphs_df.set_format("torch", columns='1AlphaBert')
        # self.paragraphs_df = pd.DataFrame()
        # self.paragraphs_df.from_csv(filePath)

    def loadDataFrame(self, filePath):
        self.paragraphs_df = pd.read_pickle(filePath)

    def doesModelExist(self, name):
        return name in self.paragraphs_df.columns

    def addModel(self, name, duplicates="yell"):
        if self.paragraphs_df.shape[0] == 0:
            raise Exception("No paragraphs to add model to, please add paragraphs first")

        if duplicates != 'ignore' and self.doesModelExist(name):
            if duplicates == "discard":
                return -1
            raise Exception("Model Already Exists")

        ser = pd.Series(np.zeros((self.paragraphs_df.shape[0], 3)).tolist(), dtype=object)
        self.paragraphs_df = pd.concat([self.paragraphs_df, ser.rename(name)], axis=1)
        return name

    def addParagraphEmbedings(self, model_name, paragraph_id_embeding_dictionary):
        ser = self.paragraphs_df[model_name]
        for paragraph_id, embedding in paragraph_id_embeding_dictionary.items():
            ser[paragraph_id] = embedding
        self.paragraphs_df[model_name] = ser
        return model_name

    def addParagraph(self, paragraph_text, document_id, duplicates="yell"):
        if duplicates != "ignore" and self.doesParagraphExist(paragraph_text, document_id):
            if duplicates == 'discard':
                return -1
            raise Exception("Duplicate Paragraph with Document ID: " + str(document_id))
        paragraph_id = self.paragraphs_df.shape[0]
        self.paragraphs_df.loc[paragraph_id] = [paragraph_id, paragraph_text, document_id, len(paragraph_text)] + [
            0] * (len(self.paragraphs_df.columns) - 4)
        return paragraph_id

    def doesParagraphExist(self, paragraph_text, document_id):
        return paragraph_text in self.paragraphs_df['paragraph_text']

    def getParagraph(self, paragraph_id):
        return self.paragraphs_df.loc[paragraph_id]

    def getParagraphSmall(self, paragraph_id):
        return self.paragraphs_df.loc[paragraph_id]

    def getParagraphIDs(self):
        return self.paragraphs_df['paragraph_id'].tolist()

    def getParagraphsByDocument(self, document_id):
        return self.paragraphs_df[self.paragraphs_df['document_id'] == document_id]['paragraph_id'].tolist()

    def exportCSV(self, filePath):
        self.paragraphs_df.to_csv(filePath)

    def exportDataFrame(self, filePath):
        self.paragraphs_df.to_pickle(filePath)

    def getEmbeding(self, paragraphID, model):
        return self.paragraphs_df.at[paragraphID, model]

    def search(self, model, paragraphID, count):
        return self.findClosest(self.getEmbeding(paragraphID, model), paragraphID, model, count)

    def findClosest(self, embedding, paragraphID, model, count):

        # Make sure that we don't ask for more data than in the dataset
        if count > self.paragraphs_df.shape[0]:
            count = self.paragraphs_df.shape[0]

        # Limiting our search area
        all_info = theRebbe.getParagraph(paragraphID)
        doc_id = all_info[2]
        searchMe = self.paragraphs_df.where(
            self.paragraphs_df['document_id'] != doc_id
        )

        # Calculating the distance between the embedding and the other embeddings
        for i in range(searchMe.shape[0]):
            if i == paragraphID:
                searchMe.drop(labels=paragraphID, axis=0)
                continue
            c = torch.dist(torch.tensor(embedding), torch.tensor(searchMe.at[i, model]))
            searchMe.at[i, 'distance'] = c.item()

        # Sorting the results by distance

        searchMe = searchMe.sort_values(by='distance')

        # Returning the top count results
        return searchMe.iloc[:count, :]


model_name = 'avichr/Legal-heBERT' # for legal HeBERT model trained from scratch

alephbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
alephbert = AutoModel.from_pretrained(model_name)
# alephbert_tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
# alephbert = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
# alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
# alephbert = BertModel.from_pretrained('onlplab/alephbert-base')
# alephbert.eval()

# checkpoint = 'ysnow9876/alephbert-base-finetuned-for-shut'
# tokenizer = AutoTokenizer.from_pretrained(checkpoint)
# model = BertModel.from_pretrained(checkpoint)
# model.eval()

# Create a Test Manager Class
theRebbe = Manager()

# print("The Rebbe: ")
# # print(theRebbe.paragraphs_df)
# # Now Let's Read the Documents
# fileNames = os.listdir("/home/dsi/lsnow/python/responsa_for_research/")
# all_documents = list()

# count = 1
# fileNames.sort()
# # fileNames = fileNames[:1000]
# counter = 0
# print(len(fileNames))
# for fileName in fileNames:
#     counter += 1
#     if counter % 100 == 0:
#         print(counter)
#     f = open("/home/dsi/lsnow/python/responsa_for_research/" + fileName, "rb")
#     text = f.read()
#     full_text = text.decode("cp1255", errors="ignore")

#     length_of_doc = len(full_text)
#     text_array = full_text.split("@")
#     paragraph_count = 1
#     title = ""
#     paragraph_list = list()
#     for paragraph in text_array:
#         if paragraph_count == 1:
#             title = paragraph.split("\r\n", 1)[:1]
#             paragraph = paragraph.split("\r\n", 1)[1:]
#         else:
#             paragraph = paragraph.split("\r\n", 1)[:]
#         try:
#             if len(paragraph[
#                        0]) > 400:  # what number should this be to take out small paragraphs that don't mean anything
#                 theRebbe.addParagraph(paragraph[0], fileName, "discard")
#             # theRebbe.addParagraph(paragraph, fileName)
#             paragraph_count += 1
#         except:
#             print("An exception occurred")

# theRebbe.addModel("1AlphaBert", duplicates='discard')
# theRebbe.addModel("fine_tuned_version", duplicates='discard')
# print("Data Added")
# # theRebbe.addParagraphEmbedings('AlphaBert',{0:[3,3,3],
# #                                                1:[[4,5,6],[7,8,9]]})

# theRebbe.exportCSV("paragraphs3.csv")
# theRebbe.exportDataFrame("embeddings2.pkt")
# print("exported")
# print(theRebbe.paragraphs_df)
theRebbe.loadDataFrame("embeddings2.pkt")
# theRebbe.exportCSV("paragraphs3_3.csv")
# print("exported")
counter = 0
print(theRebbe.paragraphs_df)
for id in theRebbe.getParagraphIDs():
    text = theRebbe.getParagraph(id)['paragraph_text']
    input = alephbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    output = alephbert(**input)
    enncoding = output.last_hidden_state

    # TODO I'm not sure what to do because the embedding are not the same size
    theRebbe.addParagraphEmbedings('1AlphaBert', {id: enncoding.detach().numpy()})

    if counter % 1000 == 0:
        print(counter)
        theRebbe.exportCSV("paragraphs3.csv")
        theRebbe.exportDataFrame("embeddings2.pkt")
    counter += 1
    if counter > 10000:
        break

# for id in theRebbe.getParagraphIDs():
#     text = theRebbe.getParagraph(id)['paragraph_text']
#     input = tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
#     output = model(**input)
#     enncoding = output.last_hidden_state

#     # TODO I'm not sure what to do because the embedding are not the same size
#     theRebbe.addParagraphEmbedings('fine_tuned_version', {id: enncoding.detach().numpy()})

#     if counter % 100 == 0:
#         print(counter)
#         theRebbe.exportCSV("paragraphs2.csv")
#         theRebbe.exportDataFrame("embeddings2.pkt")
#     counter += 1

# theRebbe.exportCSV("paragraphs2.csv")
# theRebbe.exportDataFrame("embeddings2.pkt")
# print(theRebbe.paragraphs_df)

# theRebbe.loadCSV("paragraphs.csv")
# results = theRebbe.search("1AlphaBert", 14, 10)
# results2 = theRebbe.search("fine_tuned_version", 14, 10)
# results.to_csv("long_para_search_aleph_bert.csv")
# results.to_csv("long_para_search_fine_tuned_bert.csv")




model = '1AlphaBert'
results1 = theRebbe.search(model,10,5)
results1.to_csv("all_paragraphs_final"+model+"_10_find5.csv")

results = theRebbe.search(model,76,5)
results.to_csv("all_paragraphs_final"+model+"_76_find5.csv")

results = theRebbe.search(model,106,5)
results.to_csv("all_paragraphs_final"+model+"_106_find5.csv")

results = theRebbe.search(model,138,5)
results.to_csv("all_paragraphs_final"+model+"_138_find5.csv")

results = theRebbe.search(model,805,5)
results.to_csv("all_paragraphs_final"+model+"_805_find5.csv")
