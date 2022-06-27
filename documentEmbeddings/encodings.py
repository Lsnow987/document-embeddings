# https://huggingface.co/biu-nlp/alephbert-base
# from transformers import BertModel, BertTokenizerFast
#
# alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
# alephbert = BertModel.from_pretrained('onlplab/alephbert-base')
#
# # if not finetuning - disable dropout
# alephbert.eval()
#
# alephbert.encode("גם כשהשפה העברית לא שימשה שפת דיבור")

# from transformers import BertModel, BertTokenizerFast
# import torch
#
# alephbert_tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
# alephbert = BertModel.from_pretrained('onlplab/alephbert-base')
#
# # if not finetuning - disable dropout
# alephbert.eval()
#
# inputs = alephbert_tokenizer("אבא לבית", return_tensors="pt")
# inputs1 = alephbert_tokenizer("אמא לבית", return_tensors="pt")
# inputs2 = alephbert_tokenizer("לבית אדם", return_tensors="pt")
#
# outputs = alephbert(**inputs)
# outputs1 = alephbert(**inputs1)
# outputs2 = alephbert(**inputs2)
#
# logits = outputs.last_hidden_state
# logits1 = outputs1.last_hidden_state
# logits2 = outputs2.last_hidden_state
#
#
# logits.size()
# logits1.size()
# logits2.size()
#
# close = logits - logits1
# far = logits - logits2
#
# print(torch.linalg.norm(close, dim=2))
# print(torch.linalg.norm(far, dim=2))
