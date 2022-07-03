# a lot of the code here is either taken from or base off of the code here: https://huggingface.co/course/chapter7/3?fw=pt
import html
from datasets import load_dataset
from huggingface_hub import notebook_login
from torch.utils.data import DataLoader
from transformers import default_data_collator
from torch.optim import AdamW
from accelerate import Accelerator
from transformers import get_scheduler
from huggingface_hub import Repository
from tqdm.auto import tqdm
import torch
import math

def tokenize_function(paragraphs):
    result = tokenizer(paragraphs["paragraph_text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(paragraphs):
    # Concatenate all texts
    concatenated_examples = {k: sum(paragraphs[k], []) for k in paragraphs.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(paragraphs.keys())[0]])
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

def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

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


data_files = {"train": "train.csv", "validation": "validation.csv", "test": "testing.csv"}
test_set = load_dataset("csv", data_files=data_files)

test_set = test_set.map(lambda x: {"paragraph_text": html.unescape(x["paragraph_text"])})

# Use batched=True to activate fast multithreading!
tokenized_datasets = test_set.map(
    tokenize_function, batched=True, remove_columns=["Unnamed: 0", "document_id", "length", "split",
                                                     "1AlphaBert", "paragraph_text", "paragraph_id"]
)

chunk_size = 128 #what do we want to make the chunk size - will use more ram if use a bigger chunk size

lm_datasets = tokenized_datasets.map(group_texts, batched=True)
print(lm_datasets)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

notebook_login()

#we prob won't actually be using this because we already did a train,validation,test set split
train_size = 7000 # change for actual amount
test_size = int(0.1 * train_size)

downsampled_dataset = lm_datasets["train"].train_test_split(
    train_size=train_size, test_size=test_size, seed=42
)

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

batch_size = 64  # what do we want this to be?
train_dataloader = DataLoader(
    downsampled_dataset["train"],
    shuffle=True,
    batch_size=batch_size,
    collate_fn=data_collator,
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)

optimizer = AdamW(model.parameters(), lr=5e-5)

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

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
# ysnow9876/
model_name = "alephbert-base-finetuned-for-shut"
repo_name = get_full_repo_name(model_name)
print(repo_name)
print("\n reached here \n")
output_dir = model_name
repo = Repository(output_dir, clone_from=repo_name)

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
