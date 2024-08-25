import torch
from datasets import Dataset
from translation.translation_dataloader import TranslationDataset
import pandas as pd
from transformers import AutoTokenizer
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, \
    AutoModelForSeq2SeqLM
from transformers import DataCollatorForSeq2Seq

MODEL = "sdadas/polish-bart-base"
TOKENIZER = "facebook/bart-large"

checkpoint = MODEL
source_lang = "jpm"
target_lang = "pl"

train_data = []
val_data = []
test_data = []

def preprocess_function(example):
    inputs = example["translation"][source_lang]
    targets = example["translation"][target_lang]
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    labels = tokenizer(targets, max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    model_inputs["translation"] = example["translation"]
    return model_inputs

data = TranslationDataset(
    files=("../../../../data/translation/dataset_part1.txt", "../../../../data/translation/dataset_part2.txt"))
train_iter, val_iter, test_iter = torch.utils.data.random_split(data, [0.9, 0.05, 0.05])

for sample, target in train_iter:
    train_data.append({"translation": {"jpm": sample, "pl": target}})

for sample, target in val_iter:
    val_data.append({"translation": {"jpm": sample, "pl": target}})

df = pd.DataFrame(train_data)
train_dataset = Dataset.from_pandas(df)
df = pd.DataFrame(val_data)
val_dataset = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER, return_tensors="pt")

train_data = train_dataset.map(preprocess_function)
val_data = val_dataset.map(preprocess_function)

model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, device_map="auto")

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

training_args = Seq2SeqTrainingArguments(
    output_dir="./results_bart",
    eval_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=6,
    fp16=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    tokenizer=tokenizer,
    data_collator=data_collator
)

trainer.train()
trainer.save_model("./model_bart")
