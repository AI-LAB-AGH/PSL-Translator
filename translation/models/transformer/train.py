import os
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torch.nn as nn
from transformers import AutoTokenizer
from translation.models.transformer.model import Seq2SeqTransformer
from processing_helpers import *
from train_helpers import *
from translation.torch_datasets.translation_dataset import TranslationDataset
from translation.torch_datasets.gec_dataset import GecDataset
from translation.torch_datasets.paraphrases_dataset import ParaphraseDataset
import definitions


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 32
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NUM_EPOCHS = 10

# Use Hugging Face tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("dkleczek/bert-base-polish-uncased-v1", model_max_length=64)  # Example Polish tokenizer

PAD_IDX = tokenizer.pad_token_id
BOS_IDX = tokenizer.bos_token_id or tokenizer.cls_token_id
EOS_IDX = tokenizer.eos_token_id or tokenizer.sep_token_id

def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_encoded = tokenizer(src_sample.rstrip("\n"), padding='max_length', truncation=True, return_tensors="pt")
        tgt_encoded = tokenizer(tgt_sample.rstrip("\n"), padding='max_length', truncation=True, return_tensors="pt")
        src_batch.append(src_encoded['input_ids'].squeeze(0))
        tgt_batch.append(tgt_encoded['input_ids'].squeeze(0))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


train_translation = TranslationDataset(os.path.join(definitions.ROOT_DIR, "translation/nlp_data/data.txt"))
val_translation = TranslationDataset(os.path.join(definitions.ROOT_DIR, "translation/nlp_data/val_data.txt"))
train_translation_dataloader = DataLoader(train_translation, batch_size=BATCH_SIZE, collate_fn=collate_fn)
val_translation_dataloader = DataLoader(val_translation, batch_size=BATCH_SIZE, collate_fn=collate_fn)

train_gec = GecDataset(os.path.join(definitions.ROOT_DIR, "translation/nlp_data/gec_dataset.jsonl"))
val_gec = GecDataset(os.path.join(definitions.ROOT_DIR, "translation/nlp_data/gec_dataset.jsonl"), val=True)
train_gec_dataloader = DataLoader(train_gec, batch_size=BATCH_SIZE, collate_fn=collate_fn)
val_gec_dataloader = DataLoader(val_gec, batch_size=BATCH_SIZE, collate_fn=collate_fn)

train_paraphrase = ParaphraseDataset()
val_paraphrase = ParaphraseDataset(val=True)
train_paraphrase_dataloader = DataLoader(train_paraphrase, batch_size=BATCH_SIZE, collate_fn=collate_fn)
val_paraphrase_dataloader = DataLoader(val_paraphrase, batch_size=BATCH_SIZE, collate_fn=collate_fn)

# Transformer model using the pretrained embeddings
transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, tokenizer.vocab_size, tokenizer.vocab_size, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

# Train the model
train(NUM_EPOCHS, train_paraphrase_dataloader, val_paraphrase_dataloader, transformer, optimizer, loss_fn)
train(NUM_EPOCHS, train_gec_dataloader, val_gec_dataloader, transformer, optimizer, loss_fn)
train(NUM_EPOCHS, train_translation_dataloader, val_translation_dataloader, transformer, optimizer, loss_fn)

# Save the model
torch.save(transformer.state_dict(), "trained_models/transformer_translation_only.pth")
