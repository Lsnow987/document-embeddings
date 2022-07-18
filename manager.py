import pandas as pd
import os
import numpy as np
import math
from transformers import BertModel, BertTokenizerFast
import torch
from tqdm import tqdm

pd.options.mode.chained_assignment = None  # default='warn'

class manager:
    # Main Dataframe
    paragraphs_df = None

    # Constructor - no filepath
    def __init__(self):
        self.paragraphs_df = pd.DataFrame(columns=['paragraph_id', 'paragraph_text', 'document_id','length']).astype('object')

    def loadCSV(self, filePath):
        self.paragraphs_df = pd.read_csv(filePath, index_col=0, encoding_errors='ignore')

    def loadDataFrame(self, filePath):
        self.paragraphs_df = pd.read_pickle(filePath)

    def doesModelExist(self,name):
        return(name in self.paragraphs_df.columns)

    def addModel(self, name, duplicates="yell"):
        if(self.paragraphs_df.shape[0] == 0):
            raise Exception("No paragraphs to add model to, please add paragraphs first")

        if(duplicates!= 'ignore' and self.doesModelExist(name)):
            if(duplicates=="discard"):
                return -1
            raise Exception ("Model Already Exists")

        ser = pd.Series(np.zeros((self.paragraphs_df.shape[0], 3)).tolist(),dtype=object)
        self.paragraphs_df = pd.concat([self.paragraphs_df, ser.rename(name)], axis=1)
        return name

    def addParagraphEmbedings(self, model_name, paragraph_id_embeding_dictionary):
        ser = self.paragraphs_df[model_name]
        for paragraph_id, embedding in paragraph_id_embeding_dictionary.items():
            ser[paragraph_id] = embedding
        self.paragraphs_df[model_name] = ser
        return model_name

    def addParagraph(self, paragraph_text, document_id, duplicates="yell"):
        if(duplicates != "ignore" and self.doesParagraphExist(paragraph_text,document_id)):
            if(duplicates == 'discard'):
                return -1
            raise Exception("Duplicate Paragraph with Document ID: " + str(document_id))
        paragraph_id = self.paragraphs_df.shape[0]
        self.paragraphs_df.loc[paragraph_id] = [paragraph_id, paragraph_text, document_id, len(paragraph_text)] + [0] * (len(self.paragraphs_df.columns)-4)
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

    def getEmbeding(self,paragraphID,model):
        return self.paragraphs_df.at[paragraphID,model]

    def search(self, model, paragraphID, count):
        print("Searching for " + str(count) + " closest paragraphs")
        print(self.getParagraph(paragraphID))
        return self.findClosest(self.getEmbeding(paragraphID,model),paragraphID, model, count)


    def findClosest(self, embedding, paragraphID, model, count):

        # Make sure that we don't ask for more data than in the dataset
        if(count > self.paragraphs_df.shape[0]):
            count = self.paragraphs_df.shape[0]

        #Limiting our search area
        all_info = theRebbe.getParagraph(paragraphID)
        doc_id = all_info[2]
        searchMe = self.paragraphs_df
        # self.paragraphs_df.where(
        #         self.paragraphs_df['document_id'] != doc_id
        #     )

        print("Calculating Distances")
        #Calculating the distance between the embedding and the other embeddings
        
        for i in tqdm(range(searchMe.shape[0])):
            if (type(self.getEmbeding(i,model)) == type([0,0,0])):
                searchMe.drop(labels=i, axis=0)
                continue 


            c = torch.dist(torch.tensor(embedding).cuda(), torch.tensor(searchMe.at[i, model]).cuda())
            searchMe.at[i, 'distance'] = c.item()

        print("Sorting")
        #Sorting the results by distance

        searchMe = searchMe.sort_values(by='distance')
        
        #Returning the top count results
        return searchMe.iloc[:(count+1), :]


# Create a Test Manager Class
theRebbe = manager()
theRebbe.loadDataFrame("all_paragraphs_11000_2Alephbert-FineTuned-A.pkt")
#  Now Let's Read the Documents
# fileNames = os.listdir("/zooper2/jacob.khalili/docEmbeddings/Data")
# all_documents = list()

# count = 1
# fileNames.sort()
# for fileName in fileNames:
#     f = open("/zooper2/jacob.khalili/docEmbeddings/Data/" + fileName, "rb")
#     text = f.read()
#     full_text = text.decode("cp1255", errors="ignore")

#     if(count % 100 == 0):
#         print("Processing Document: " + str(count))
#     count += 1

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

#         if len(paragraph[0]) > 150:  # what number should this be to take out small paragraphs that don't mean anything
#             theRebbe.addParagraph(paragraph[0], fileName,"discard")
#             #theRebbe.addParagraph(paragraph, fileName)
#         paragraph_count += 1


model = "2Alephbert-FineTuned-A"
# theRebbe.addModel(model,duplicates='discard')

# # print("Data Added")
# # #theRebbe.addParagraphEmbedings('AlphaBert',{0:[3,3,3],
# #                                                 1:[[4,5,6],[7,8,9]]})

# alephbert_tokenizer = BertTokenizerFast.from_pretrained("ysnow9876/alephbert-base-finetuned-for-shut")
# alephbert = BertModel.from_pretrained("ysnow9876/alephbert-base-finetuned-for-shut")
# alephbert.eval()

# counter = 0
# print(theRebbe.paragraphs_df)
# for id in theRebbe.getParagraphIDs():
#     text = theRebbe.getParagraph(id)['paragraph_text']
#     input = alephbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
#     output = alephbert(**input)
#     enncoding = output.last_hidden_state


#     #TODO I'm not sure what to do because the embedding are not the same size
#     theRebbe.addParagraphEmbedings(model,{id:enncoding.detach().numpy()})

#     if(counter % 100 == 0):
#         print(counter)

#     if(counter % 1000 == 0):
#         theRebbe.exportDataFrame("all_paragraphs_" + str(counter) + "_" + str(model) +".pkt")
#     counter += 1
    

# theRebbe.exportCSV("all_paragraphs_final"+model+".csv")
# theRebbe.exportDataFrame("all_paragraphs_final"+model+".pkt")
# print(theRebbe.paragraphs_df)


results = theRebbe.search(model,6,5)
results.to_csv("all_paragraphs_final"+model+"_6_find5.csv")

results1 = theRebbe.search(model,10,5)
results1.to_csv("all_paragraphs_final"+model+"_10_find5.csv")

results = theRebbe.search(model,76,5)
results.to_csv("all_paragraphs_final"+model+"_76_find10.csv")

print(theRebbe.getParagraph(6))
print(theRebbe.getParagraph(10))
print(theRebbe.getParagraph(76))
