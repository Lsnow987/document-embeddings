<!-----

Yay, no errors, warnings, or alerts!

Conversion time: 1.182 seconds.


Using this Markdown file:

1. Paste this output into your source file.
2. See the notes and action items below regarding this conversion run.
3. Check the rendered output (headings, lists, code blocks, tables) for proper
   formatting and use a linkchecker before you publish this page.

Conversion notes:

* Docs to Markdown version 1.0β33
* Tue Aug 02 2022 02:46:45 GMT-0700 (PDT)
* Source doc: readme
----->


# Responsa Recommender

A tool for finding similar and related rabbinic responsa. 

# What are Responsa?

Responsa are one of the cornerstones of modern halachic (Jewish Law) literature. Careful learning, analysis and understandings of these scholarly letters inform not only the recipient on how they are to act, but establish important halachic precedence and principles used to decide complex questions for future generations.

# Why is a tool like this needed? 

Whether it’s used to find precedence, used to understand the principles on which these decisions are made, or simply to find opinions of other Jewish scholars, a tool like this would be indispensable to the torah learning community.  

# How does it work? 

We use a BERT econder, to generate encoding for each paragraph, and find the paragraphs that are closests to the one searched for.  The BERT Model used is a fine-tuned version of alephbert-base. The model was trained on a subset of the Bar Ilan Responsa  Project’s collection of rabanic responsa.

The current version of the model can be found at [https://huggingface.co/ysnow9876/alephbert-base-finetuned-for-shut](https://huggingface.co/ysnow9876/alephbert-base-finetuned-for-shut)

See information for how to use the model in the Model card there.

# Basic Code Structure

There are two main files, which are used for this project. 



* Fine_tuning.py, which is used to train BERT Models, using the hugging face framework. 
* Manager.py, which contains a manger class, which is used to handle the data, and perform searches

# How to get things running:

## Requirements

To install all dependencies required run pip3 install -r requirements.txt

## Generating Dataframe



    * Make sure that all the text is stored using ‘cp1255’ encoding, and all document names are valid integers, and stored in the same folder.
    *  The manager.py contains 3 different methods of generating dataframes:
        * Generate_datafrane_standard: each paragraph is its own row
        * calling generate_dataframe_per_document: each document will be its own row:  The text that will be stored for each document are the first 250 words and last 250 words in the document
        * Generate_dataframe_per_100:  where each row will consist of 100 words, this is useful when the BERT model being used as a max size of 128 tokens. 

After generating the dataframe, it can be exported to a .pkt file by calling theRebbe.exportDataFrame(filePath), where theRebbe is an instance of the manager class and filePath is the path to where the file should be stored along with the name ending in .pkt. It can also be exported to a .csv file, which will be helpful for running fineTuning.py, by calling theRebbe.exportCSV(filePath) in the same way. 

Similarly, dataframes and CSV files can be imported using .loadCSV(), and .loadDataFrame() 

# Fine Tuning The Model:

In order to run fineTuning.py all of the data that the model will be fine-tuned on must be stored in one csv file, which was obtained above.

Notes:

On line 25 model_checkpoint is set to the name of the pretrained model that is being fine-tuned.

Line 23 (notebook_login()) will make a prompt on the screen asking for an access token to hugging face hub. The instruction for how to generate an access token can be found here: 

[User access tokens (huggingface.co)](https://huggingface.co/docs/hub/security-tokens). An account is required to make an access token.

Once all of the data is stored in a csv file, fileName on line 61 must be set to the name of the csv file.

The chunk size (how big paragraphs would be passed in when fine-tuning) to 128 tokens on line 72 because of ram limits.

On lines 82-84 the train size was set to 85 percent of the dataset and the test size to 15 percent of the dataset. Replace size_of_dataset with however many rows are in the dataset being used.

On line 109 the batch size is set equal to 16 due to ram limits. The number can be made bigger if more ram is available.

On line 128 num_train_epochs is set to 3. The number of epochs can be changed to have more or less epochs.

On line 140 mode_name should be set to whatever the name of the fine_tuned model should be.

## Generating Embeddings

If the data frame generated was not the dataframe where there is one general embedding for each document, theRebbe.add_local_paragraph_ids()  must be called. This provides useful information that will later be used to prevent a paragraph from the same document as the paragraph being searched for from showing up in the search results.

If the data frame generated was the one that provided general embeddings for each document lines 135-137 which prevent paragraphs from the same document as the paragraph being searched for from appearing in the search results, must be commented out.

After generating the data frames containing all the texts, a column can be added containing the CLS tokens (embeddings that contain information about the paragraph) generated by any BERT model by first calling theRebbe.addModel(“modelName”) where modelName should be whatever the name of the model is. Then theRebbe.generateEmbeddings(prefix, model, startValue, end) should be called. Prefix is whatever the first part of the fileName should be. Model is the name of the model as it was put into theRebbe.addModel. Start is from which row should embeddings start being generated from (generally 0), and end is which row embeddings should stop being generated at (generally the last row).

## Searching

Once the embeddings were generated a search can be done by calling theRebbe.search(model, paragraphId, num_of_search_results). This function will return an array of distance, and paragraph ID. Model should be set to whichever model’s embeddings are to be used for the search. ParagraphId is the number id of the paragraph for which similar paragraphs are being searched for. This can be found in the .csv or.pkt file. num_of_search_results is the number of paragraphs that are topically similar that should be returned. 

In order to convert the search results into a pdf containing hebrew text we used the DejaVuSansCondensed font. This font must be installed in order for everything to run smoothly. It can be installed by either downloading the DejaVuSansCondensed.ttf from the github repository or from somewhere else online. Then replace the word DejaVuSansCondensed.ttf on line 120 with the path to the ttf file (or leave it the same if the ttf file is stored in the same directory as the manager class)

This information can be turned into a pdf by calling theRebbe.createPDF(distances, filename). Distances are the results received from running the search. Filename is the directory where the file should be stored and what the filename should be when the pdf is saved.

# Next Steps:



1. We believe the model could benefit from more training, and a more diverse dataset. For example, training the model first on an array of more formal hebrew literature, and then more fine-tuning specific to rabbanic responsa
2. Furthermore, developing metrics of assessment is crucial in determining how successful an approach is on a larger scale. Integrating a tool like this into the public sphere to get some ground truth data is critically important to the project's ability to improve in the long term. 
3. A more careful integration of the different approaches, we did a simple sum of the distances, this would likely benefit from a weighted sum, and/or a discriminator model. 
4. Building a discriminator model, which would be useful in removing irrelevant results. 
5. Test using the rabbanic BERT being developed.
