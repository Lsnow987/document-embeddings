# document-embeddings
We Used alephbert-base to find similar rabbinic texts. We fine-tuned this model based on rabbinic texts to be more accurate in finding similarities.

We were given thousands of different rabbinic texts in which people asked rabbis different questions. We combined all the data into a .pkt file, added document embeddings using onlplab/alephbert-base, and then compared the embeddings to find the most similar document to the current document.

We fine-tunied the alephbert-base model through masked language modeling based on the texts we were given to achieve more accurate similarity results.

The current version of the model can be found at https://huggingface.co/ysnow9876/alephbert-base-finetuned-for-shut
