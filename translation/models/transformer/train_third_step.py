from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from translation.models.transformer.model import Seq2SeqTransformer
from processing_helpers import *
from train_helpers import *
from translation.torch_datasets.translation_dataset import TranslationDataset
import definitions
import os

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3
NUM_EPOCHS = 10


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(build_text_transform(token_transform, vocab_transform)(src_sample.rstrip("\n")))
        tgt_batch.append(build_text_transform(token_transform, vocab_transform)(tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch


train_iter = TranslationDataset(os.path.join(definitions.ROOT_DIR, "translation/nlp_data/data.txt"))
val_iter = TranslationDataset(os.path.join(definitions.ROOT_DIR, "translation/nlp_data/val_data.txt"))

train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

token_transform = get_tokenizer('spacy', language='pl_core_news_sm')
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
vocab_transform = build_vocab_from_iterator(yield_tokens(train_iter, token_transform),
                                            min_freq=1,
                                            specials=special_symbols,
                                            special_first=True)
vocab_transform.set_default_index(UNK_IDX)
VOCAB_SIZE = len(vocab_transform)

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                 NHEAD, VOCAB_SIZE, VOCAB_SIZE, FFN_HID_DIM)

transformer.load_state_dict(torch.load('trained_models/transformer.pth', weights_only=True))
transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

train(NUM_EPOCHS, train_dataloader, val_dataloader, transformer, optimizer, loss_fn)

torch.save(transformer.state_dict(), "trained_models/transformer.pth")

