import pandas as pd
import os
import numpy as np
import math
from transformers import BertModel, BertTokenizerFast
import torch

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
        return self.findClosest(self.getEmbeding(paragraphID,model),paragraphID, model, count)

    def findClosest(self, embedding, paragraphID, model, count):

        # Make sure that we don't ask for more data than in the dataset
        if(count > self.paragraphs_df.shape[0]):
            count = self.paragraphs_df.shape[0]

        #Limiting our search area
        all_info = theRebbe.getParagraph(paragraphID)
        doc_id = all_info[2]
        searchMe = self.paragraphs_df.where(
                self.paragraphs_df['document_id'] != doc_id
            )

        #Calculating the distance between the embedding and the other embeddings
        for i in range(searchMe.shape[0]):
            if (i == paragraphID):
                searchMe.drop(labels=paragraphID, axis=0)
                continue 
            c = torch.dist(torch.tensor(embedding), torch.tensor(searchMe.at[i, model]))
            searchMe.at[i, 'distance'] = c.item()

        #Sorting the results by distance

        searchMe = searchMe.sort_values(by='distance')
        
        #Returning the top count results
        return searchMe.iloc[:count, :]


# Create a Test Manager Class
theRebbe = manager()

print("The Rebbe: ")
print(theRebbe.paragraphs_df)
# Now Let's Read the Documents
fileNames = os.listdir("/home/jacob/code/responaProjectReccomender/Data/")
all_documents = list()

count = 1
fileNames.sort()
fileNames = fileNames[:1000]
for fileName in fileNames:
    f = open("/home/jacob/code/responaProjectReccomender/Data/" + fileName, "rb")
    text = f.read()
    full_text = text.decode("cp1255", errors="ignore")

    length_of_doc = len(full_text)
    text_array = full_text.split("@")
    paragraph_count = 1
    title = ""
    paragraph_list = list()
    for paragraph in text_array:
        if paragraph_count == 1:
            title = paragraph.split("\r\n", 1)[:1]
            paragraph = paragraph.split("\r\n", 1)[1:]
        else:
            paragraph = paragraph.split("\r\n", 1)[:]

        if len(paragraph[0]) > 1000:  # what number should this be to take out small paragraphs that don't mean anything
            theRebbe.addParagraph(paragraph[0], fileName,"discard")
            #theRebbe.addParagraph(paragraph, fileName)
        paragraph_count += 1


theRebbe.addModel("1AlphaBert",duplicates='discard')
print("Data Added")
#theRebbe.addParagraphEmbedings('AlphaBert',{0:[3,3,3],
#                                                1:[[4,5,6],[7,8,9]]})

alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
alephbert = BertModel.from_pretrained('onlplab/alephbert-base')
alephbert.eval()



counter = 0
print(theRebbe.paragraphs_df)
for id in theRebbe.getParagraphIDs():
    text = theRebbe.getParagraph(id)['paragraph_text']
    input = alephbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
    output = alephbert(**input)
    enncoding = output.last_hidden_state


    #TODO I'm not sure what to do because the embedding are not the same size
    theRebbe.addParagraphEmbedings('1AlphaBert',{id:enncoding.detach().numpy()})

    if(counter % 100 == 0):
        print(counter)
    counter += 1
    

theRebbe.exportCSV("long_para1.csv")
theRebbe.exportDataFrame("long_para1.pkt")
print(theRebbe.paragraphs_df)

results = theRebbe.search("1AlphaBert",14,10)
results.to_csv("long_para_search2.csv")
