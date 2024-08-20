from timeit import default_timer as timer
from torch.utils.data import DataLoader
import torch.nn as nn
from utils import *
from model import Seq2SeqTransformer
from evaluation import val_loss as val_eval, bleu_evaluation

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
token_transform = {}
vocab_transform = build_vocab_transform()
SRC_LANGUAGE = 'jpm'
TGT_LANGUAGE = 'pl'
token_transform[SRC_LANGUAGE] = get_tokenizer(None, 'pl')
token_transform[TGT_LANGUAGE] = get_tokenizer(None, language='pl')
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']

def train_epoch(model, optimizer, loss_fn, train_iter):
    model.train()
    losses = 0
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(train_dataloader))

data = TranslationDataset(files=("../../../data/translation/dataset_part1.txt", "../../../data/translation/dataset_part2.txt"))
train_iter, val_iter, test_iter = torch.utils.data.random_split(data, [0.9, 0.05, 0.05])

torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 4
FFN_HID_DIM = 128
BATCH_SIZE = 64
NUM_ENCODER_LAYERS = 2
NUM_DECODER_LAYERS = 2

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)


NUM_EPOCHS = 15

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer, loss_fn, train_iter)
    end_time = timer()
    val_loss = val_eval(transformer, val_iter, loss_fn, BATCH_SIZE)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

print("Your BLEU score is: " + str(bleu_evaluation(transformer, test_iter)))

torch.save(transformer, 'transformer.pth')



