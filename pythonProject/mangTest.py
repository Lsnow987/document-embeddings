from fileinput import filename
import pandas as pd
from os.path import exists
from transformers import BertModel, BertTokenizerFast
import torch

class Manager:
    df = None
    fileName = ""

    def __init__(self, fileNameIn):
        fileName = fileNameIn
        self.df = pd.read_csv(fileName, index_col=0, encoding_errors='ignore')

    ##Creates
    #id = which number model this is
    #name = the name of the model
    def addModel(self, id, name):
        return 0

        #id = which number paragraph this is in relation to all of the paragraphs
        #document is which document is this a part of
        #text is the text of the document
    def addParagraphs(self, id, document, text):
        return 0

    def getParagraphText(id):
        return 0

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
        #list of closet paragraph ID using a particular model
        length = len(self.df)
        num_rows = int(self.df.size/length)
        all_embeddings = []
        for i in range(num_rows):
            all_embeddings.append(self.df.values[3][i]) #change 3 based on model id

        return self.closest_embeddings(count, all_embeddings, text)



    def closest_embeddings(self, count, all_embeddings, paragraph_text):

        close = logits - logits1
        far = logits - logits2
        return same_topic_list

    #return first 2500 characters not in middle of word of of text




n = Manager("paragraphs.csv")
print(n.df)