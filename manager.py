from sqlite3 import Row
from time import time
from turtle import distance
from xmlrpc.client import DateTime
import pandas as pd
import os
import numpy as np
import math
from transformers import BertModel, BertTokenizerFast, BertTokenizer, BertForMaskedLM
import torch
from tqdm import tqdm
import time
import fpdf
from unidecode import unidecode
from rabtokenizer import RabbinicTokenizer
pd.options.mode.chained_assignment = None  # default='warn'

class manager:
    # Main Dataframe
    paragraphs_df = None

    # Constructor - no filepath
    def __init__(self):
        self.paragraphs_df = pd.DataFrame(columns=['paragraph_id', 'paragraph_text', 'document_id','length']).astype('object')

    def loadCSV(self, filePath):
        self.paragraphs_df = pd.read_csv(filePath, index_col=0) #encoding_errors='ignore'

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
        global first_id 
        global first_doc_id
        first_id = paragraphID
        first_doc_id = self.getParagraph(first_id)[2]

        print("Searching for " + str(count) + " closest paragraphs")
        print(self.getParagraph(paragraphID))
        return self.findClosest(self.getEmbeding(paragraphID,model),paragraphID, model, count)

    def createPDF(self, distances, filename):

        pdf = fpdf.FPDF()
        # pdf.set_font("Arial", size=12)
        pdf.add_font('DejaVu', '', 'DejaVuSansCondensed.ttf', uni=True)
        pdf.set_font('DejaVu', '', 14)
        # print(distances)
        for row in distances.index:
            # print(type(row))
            pdf.add_page()
            currId = row
            
            
            if(row > first_id):
                currId = row + dropped

            getParagraph = self.getParagraph(currId)
            doc_Id = getParagraph['document_id']

            # counter = 0
            # for index, rows in self.paragraphs_df.iterrows():
            #     if(rows['document_id'] == doc_Id):
            #         # if row['paragraph_id'] == first_id:
            #         break
            #     else:
            #         counter+=1    

            if(doc_Id == first_doc_id):
                local_id = getParagraph["localParagraphId"]
                # if(local_id == counter):
                #     print("EQUALS")
                # print("local id: "+str(local_id))
                # print("counter: "+ str(counter))
                currId = (currId + first_id - local_id) #currId + local_id #(first_id - counter)

            
            pdf.cell(200, 10, txt=f"Paragraph ID: {currId}", ln=1, align="C")
            pdf.ln()
            pdf.cell(200, 10, txt=f"Paragraph Distance: {distances[row]}", ln=1, align="C")
            pdf.ln()            
            
            pdf.cell(200, 10, txt=f"Doc ID: {doc_Id}", ln=1, align="C")
            pdf.ln()            
            text = self.getParagraph(currId)[1].encode('cp1255',errors='replace').decode('cp1255',errors='replace')
            text = text[::-1]
            text = text.split()
            text = np.flip(np.array(text))
            line_index = 0
            new_line = ""
            for word in text:
                if(line_index < 70):
                    new_line = new_line + word + " "
                    line_index = line_index + len(word) + 1
                else:
                    new_line = new_line.split()
                    new_line = np.flip(np.array(new_line))
                    new_line = " ".join(new_line)
                    pdf.cell(0, 10, txt=new_line, align="R")
                    pdf.ln()
                    new_line = word + " "
                    line_index = 0

            new_line = new_line.split()
            new_line = np.flip(np.array(new_line))
            new_line = " ".join(new_line)
            pdf.cell(0, 10, txt=new_line, align="R")
            pdf.ln()
        pdf.output(filename, "F")
        print("PDF Created")
        return "distances.pdf"

    def generate_datafrane_standard(self, directory):
        #  Now Let's Read the Documents
        # "/zooper2/jacob.khalili/docEmbeddings/Data/"
        fileNames = os.listdir(directory)

        count = 1
        fileNames.sort()
        for fileName in fileNames:
            f = open(directory + fileName, "rb")
            text = f.read()
            full_text = text.decode("cp1255", errors="ignore")

            if(count % 100 == 0):
                print("Processing Document: " + str(count))
            count += 1

            # length_of_doc = len(full_text)
            text_array = full_text.split("@")
            paragraph_count = 1
            # title = ""
            # paragraph_list = list()
            for paragraph in text_array:
                if paragraph_count == 1:
                    # title = paragraph.split("\r\n", 1)[:1]
                    paragraph = paragraph.split("\r\n", 1)[1:]
                else:
                    paragraph = paragraph.split("\r\n", 1)[:]

                if len(paragraph[0]) > 150:  # what number should this be to take out small paragraphs that don't mean anything
                    theRebbe.addParagraph(paragraph[0], fileName,"discard")
                paragraph_count += 1

    def generate_datafrane_per_document(self, directory):
        #  Now Let's Read the Documents
        # "/zooper2/jacob.khalili/docEmbeddings/Data/"
        fileNames = os.listdir(directory)

        count = 1
        fileNames.sort()
        for fileName in fileNames:
            f = open(directory + fileName, "rb")
            text = f.read()
            full_text = text.decode("cp1255", errors="ignore")

            if(count % 100 == 0):
                print("Processing Document: " + str(count))
            count += 1

            # length_of_doc = len(full_text)
            full_text = full_text.replace("@", "").replace("/", "")
            text_array = full_text.split(" ")
            final_text = " ".join(text_array[0:250]) + "..." + " ".join(text_array[-250:])

            theRebbe.addParagraph(final_text[0], fileName,"discard")

    def generate_datafrane_per_100(self, directory):
        #  Now Let's Read the Documents
        # "/zooper2/jacob.khalili/docEmbeddings/Data/"
        fileNames = os.listdir(directory)

        count = 1
        fileNames.sort()
        for fileName in fileNames:
            f = open(directory + fileName, "rb")
            text = f.read()
            full_text = text.decode("cp1255", errors="ignore")

            if(count % 100 == 0):
                print("Processing Document: " + str(count))
            count += 1

            # length_of_doc = len(full_text)
            full_text = full_text.replace("@", "").replace("/", "")
            text_array = full_text.split(" ")
            n = 100
            chunks = [text_array[i:i + n] for i in range(0, len(text_array), n)]
            for chunk in chunks:
                final_text = " ".join(chunk)
                theRebbe.addParagraph(final_text[0], fileName,"discard")

    #Given a paragraph id, find the closest paragraphs to it
    def generate_embeddings(self, model, startValue, end):
        alephbert_tokenizer = None
        alephbert = None
        if model == "1AlphaBert":
            alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
            alephbert = BertModel.from_pretrained('onlplab/alephbert-base')
        else:
            alephbert_tokenizer = BertTokenizerFast.from_pretrained("ysnow9876/alephbert-base-finetuned-for-shut")
            alephbert = BertModel.from_pretrained("ysnow9876/alephbert-base-finetuned-for-shut")
        # alephbert.eval()
        counter = 0
        for id in theRebbe.getParagraphIDs():
            if id >= startValue:
                text = theRebbe.getParagraph(id)['paragraph_text']
                input = alephbert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
                output = alephbert(**input)
                enncoding = output.last_hidden_state[0,0,:]


                #TODO I'm not sure what to do because the embedding are not the same size
                theRebbe.addParagraphEmbedings(model,{id:enncoding.detach().numpy()})

                if(counter % 1000 == 0):
                    print(counter)

                #if(counter % 5000 == 0 and counter != startValue and counter != 0):
                    # theRebbe.exportCSV("all_paragraphs_final"+model+".csv")
                    # theRebbe.exportDataFrame("all_paragraphs_final"+model+".pkt")
                counter += 1
                
                if counter == end:
                    # theRebbe.exportCSV("all_paragraphs_final"+model+".csv")
                    # theRebbe.exportDataFrame("all_paragraphs_final"+model+".pkt")
                    break


    def findClosest(self, embedding, paragraphID, model, count):

        # Make sure that we don't ask for more data than in the dataset
        if(count > self.paragraphs_df.shape[0]):
            count = self.paragraphs_df.shape[0]

        #Limiting our search area
        all_info = theRebbe.getParagraph(paragraphID)
        doc_id = all_info[2]
        searchMe = self.paragraphs_df.copy()
        # print(searchMe)
        # time.sleep(1)
        global dropped
        dropped = 0
        for index, row in searchMe.iterrows():
            if(row['document_id'] == doc_id and row['paragraph_id'] != paragraphID):
                searchMe.drop(index, inplace=True, axis='index')
                dropped += 1

        # searchMe.drop(searchMe[searchMe['document_id'] == doc_id].index, inplace=True)

        print("Calculating Distances")
        #Calculating the distance between the embedding and the other embeddings
        
        batch_size = 1024
        num_rows = searchMe.shape[0]
        num_batches = 66 #math.ceil(num_rows / batch_size)

        #Create a Series
        df = pd.DataFrame(columns=[])
        distances = pd.Series()
        for i in tqdm(range(num_batches)):
            
            start = i * batch_size
            end = min((i + 1) * batch_size, num_rows)
            batch = searchMe.iloc[start:end]
            batch_embeddings = batch[model].values

            batch_embeddings=np.vstack(batch_embeddings).astype(np.float64)
            batch_embeddings = torch.from_numpy(batch_embeddings)
            
            batch_embeddings = batch_embeddings - torch.tensor(embedding)
            batch_embeddings = torch.norm(batch_embeddings, p=2, dim=1)
    
            new = pd.Series(batch_embeddings.numpy())

            distances = pd.concat([distances, new], axis=0, ignore_index=True)

        distances = distances.sort_values()
        
        #Returning the top count results
        return distances.iloc[:(count+1)]

    def add_local_paragraph_ids(self):
        # theRebbe.addModel("localParagraphId")
        counter = 0
        inner_counter = 0
        initial_doc_id = None
        for index, rows in theRebbe.paragraphs_df.iterrows():
            # getParagraph = theRebbe.getParagraph(currId)
            doc_Id = rows['document_id']
            if initial_doc_id == None:
                initial_doc_id = doc_Id
            elif initial_doc_id == doc_Id:
                inner_counter += 1
            else:
                initial_doc_id = doc_Id
                counter+=1
                counter += inner_counter
                inner_counter = 0
            theRebbe.paragraphs_df.loc[index, "localParagraphId"] = counter

        print(theRebbe.paragraphs_df)
        theRebbe.exportDataFrame("all_paragraphs_final"+model+".pkt")
           
        # theRebbe.paragraphs_df.to_csv("local_ids.csv")


# Create a Test Manager Class
theRebbe = manager()
model = "1AlphaBert"
<<<<<<< HEAD
directory = "/home/jacob/code/responaProjectReccomender/Data/"
=======
dataframe = "all_paragraphs_final1AlphaBert.pkt"
directory = "home/..."
theRebbe.loadDataFrame(dataframe)

bert_path = "C:/Users/ysnow/OneDrive/Desktop/python/Berel/"
tokenizer = RabbinicTokenizer(BertTokenizer.from_pretrained(os.path.join(bert_path, 'vocab.txt')))
model = BertForMaskedLM.from_pretrained(bert_path)

# theRebbe.add_local_paragraph_ids()

#theRebbe.combine_files(directory)
#theRebbe.addModel(model,duplicates='discard')
>>>>>>> 5feea106aa3e35d77b68e94e11cdfc51dcbc528b

# # theRebbe.generate_datafrane_per_document(directory)
# theRebbe.loadDataFrame("documents_250+250.pkt")
# print(theRebbe.paragraphs_df.shape)
# #theRebbe.combine_files(directory)
# theRebbe.addModel(model,duplicates='discard')
# start = 0
<<<<<<< HEAD
# end = 26801
# theRebbe.generate_embeddings(model, 0, 5360)
# theRebbe.exportDataFrame("documents_250+250_"+model+".pkt")
# theRebbe.generate_embeddings(model, 5361, 10720)
# theRebbe.exportDataFrame("documents_250+250_"+model+".pkt")
# theRebbe.generate_embeddings(model, 10721, 16080)
# theRebbe.exportDataFrame("documents_250+250_"+model+".pkt")
# theRebbe.generate_embeddings(model, 16081, 21440)
# theRebbe.exportDataFrame("documents_250+250_"+model+".pkt")
# theRebbe.generate_embeddings(model, 21441, 26801)
# theRebbe.exportDataFrame("documents_250+250_"+model+".pkt")


# x = 0
# before = time.time()
# num_of_searches = 6
# while x < num_of_searches:
    
#     results1 = theRebbe.search(model,x,10)
#     theRebbe.createPDF(results1, f"documents_250+250_{model}_find10_"+str(x)+".pdf")
#     x+=1
=======
# end = 100_000
# theRebbe.generate_embeddings(model, start, end)

# x = 0
# before = time.time()
# num_of_searches = 30
# while x < num_of_searches:
#     # theRebbe.loadDataFrame(dataframe)
#     results1 = theRebbe.search(model,x,10)
#     theRebbe.createPDF(results1, f"all_paragraphs_final{model}_find10_"+str(x)+".pdf")
#     # results1.to_csv(str(x)+"_take_2_all_paragraphs_final"+model+"_find10.csv")
#     x+=1

>>>>>>> 5feea106aa3e35d77b68e94e11cdfc51dcbc528b
# after = time.time()
# print((after - before)/num_of_searches)
