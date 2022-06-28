import pandas as pd
import torch
import os

class manager:
    # Main Dataframe
    paragraphs_df = None

    # Constructor - no filepath
    def __init__(self):
        self.paragraphs_df = pd.DataFrame(columns=['paragraph_id', 'paragraph_text', 'document_id','length'])

    def loadCSV(self, filePath):
        self.paragraphs_df = pd.read_csv(filePath, index_col=0, encoding_errors='ignore')

    def addModel(self, id, name):
        idNameCombo = str(id) + "-" + name
        self.paragraphs_df[idNameCombo] =  torch.tensor([0] * self.paragraphs_df.shape[0])
        return idNameCombo

    def addParagraph(self, paragraph_text, document_id):
        paragraph_id = self.paragraphs_df.shape[0]
        self.paragraphs_df.loc[paragraph_id] = [paragraph_id, paragraph_text, document_id, len(paragraph_text)] + [0] * (len(self.paragraphs_df.columns)-4)
        #print(self.paragraphs_df)
        return paragraph_id

    def getParagraph(self, paragraph_id):
        return self.paragraphs_df.loc[paragraph_id]

    def getParagraphs(self):
        return self.paragraphs_df['paragraph_id'].tolist()

    def getParagraphsByDocument(self, document_id):
        return self.paragraphs_df[self.paragraphs_df['document_id'] == document_id]['paragraph_id'].tolist()

    def exportDataFrame(self, filePath):
        self.paragraphs_df.to_csv(filePath)


# Create a Test Manager Class
theRebbe = manager()

theRebbe.addModel(1, "AlphaBert")

# Now Let's Read the Documents
fileNames = os.listdir("/home/jacob/code/responaProjectReccomender/Data/")
all_documents = list()

count = 1
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

        if len(paragraph[0]) > 5:  # what number should this be to take out small paragraphs that don't mean anything
            theRebbe.addParagraph(paragraph[0], fileName)
            #theRebbe.addParagraph(paragraph, fileName)
        paragraph_count += 1

    # doc_pgraph = None
    if count == 10:
        break
    count = count + 1


theRebbe.exportDataFrame("test.csv")

