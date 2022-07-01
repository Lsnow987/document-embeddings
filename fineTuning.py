import datasets
import pandas as pd
import os
import numpy as np
import math
from transformers import BertModel, BertTokenizerFast
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling
import torch
from datasets import load_dataset
df = pd.read_csv('test.csv', index_col=0)
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.70

train = df[msk]
train.to_csv("train.csv")

testAndVal = df[~msk]

msk = np.random.rand(len(testAndVal)) <= 0.50
test = testAndVal[~msk]

val = testAndVal[msk]

test.to_csv("testing.csv")
val.to_csv("validation.csv")
import html
from datasets import load_dataset

data_files = {"train": "train.csv", "validation": "validation.csv", "test": "testing.csv"}
test_set = load_dataset("csv", data_files=data_files)
# print(test_set)

# drug_sample = test_set["train"].shuffle(seed=42).select(range(20))
# Peek at the first few examples
# print(drug_sample[:3])
test_set = test_set.map(lambda x: {"paragraph_text": html.unescape(x["paragraph_text"])})


# print(test_set)


def tokenize_function(examples):
    result = tokenizer(examples["paragraph_text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result


# Use batched=True to activate fast multithreading!
tokenized_datasets = test_set.map(
    tokenize_function, batched=True, remove_columns=["Unnamed: 0", "document_id", "length", "split",
                                                     "1AlphaBert", "paragraph_text", "paragraph_id"]
)

print(tokenized_datasets)

chunk_size = 128


# tokenized_samples = tokenized_datasets["train"][:3]

# for idx, sample in enumerate(tokenized_samples["input_ids"]):
# print(f"'>>> Review {idx} length: {len(sample)}'")

# concatenated_examples = {
#     k: sum(tokenized_samples[k], []) for k in tokenized_samples.keys()
# }
# total_length = len(concatenated_examples["input_ids"])
# print(f"'>>> Concatenated reviews length: {total_length}'")
#
# chunks = {
#     k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
#     for k, t in concatenated_examples.items()
# }

# for chunk in chunks["input_ids"]:
# print(f"'>>> Chunk length: {len(chunk)}'")


def group_texts(examples):
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i: i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


lm_datasets = tokenized_datasets.map(group_texts, batched=True)
print(lm_datasets)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

# samples = [lm_datasets["train"][i] for i in range(2)]
# for sample in samples:
#     _ = sample.pop("word_ids")
#
# for chunk in data_collator(samples)["input_ids"]:
#     print(f"\n'>>> {tokenizer.decode(chunk)}'")

# from huggingface_hub import notebook_login
#
# notebook_login()
train_size = 7000
test_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)
print(downsampled_dataset)


def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}


downsampled_dataset = downsampled_dataset.remove_columns(["word_ids"])
eval_dataset = downsampled_dataset["test"].map(
    insert_random_mask,
    batched=True,
    remove_columns=downsampled_dataset["test"].column_names,
)
eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels",
    }
)

from torch.utils.data import DataLoader
from transformers import default_data_collator

batch_size = 64
train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)

from torch.optim import AdamW

optimizer = AdamW(model.parameters(), lr=5e-5)

from accelerate import Accelerator

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

from transformers import get_scheduler

num_train_epochs = 3
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


from huggingface_hub import get_full_repo_name

model_name = "ysnow9876/alephbert-base-finetuned-for-shut"
# repo_name = get_full_repo_name(model_name)
# print(repo_name)

from huggingface_hub import Repository

output_dir = model_name
repo = Repository(output_dir, clone_from=model_name)

from tqdm.auto import tqdm
import torch
import math

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in train_dataloader:
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    print(f">>> Epoch {epoch}: Perplexity: {perplexity}")

    # Save and upload
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_dir)
        repo.push_to_hub(
            commit_message=f"Training in progress epoch {epoch}", blocking=False
        )








from transformers import TrainingArguments

batch_size = 8
# Show the training loss with every epoch
logging_steps = len(test_set["train"]) // batch_size
model_name = model_checkpoint.split("/")[-1]

training_args = TrainingArguments(
    output_dir=f"{model_name}-finetuned-for-shut",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    push_to_hub=True,
    # fp16=True,
    logging_steps=logging_steps,
)

from transformers import Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=test_set["train"],
    eval_dataset=test_set["test"],
    data_collator=data_collator,
)

import math

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.train()

eval_results = trainer.evaluate()
print(f">>> Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

trainer.push_to_hub()
