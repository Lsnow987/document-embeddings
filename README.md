# document-embeddings
We have Used onlplab/alephbert-base to find similar rabbinic texts. We are in the midst of fine-tuning this model based on rabbinic texts to be more accurate in finding similarities.

We were given thousands of different rabbinic texts in which people asked rabbis different questions. We combined all the data into a csv file, added document embeddings using onlplab/alephbert-base, and then compared the embeddings to find the most similar document to the current document.

We are currently fine-tuning the onlplab/alephbert-base through masked language modeling based on the texts we were given to achieve more accurate similarity results

The current version of the model can be found at https://huggingface.co/ysnow9876/alephbert-base-finetuned-for-shut
