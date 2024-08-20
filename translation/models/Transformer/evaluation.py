import nltk
from utils import *
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
SRC_LANGUAGE = 'jpm'
TGT_LANGUAGE = 'pl'






def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = build_text_transform()[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(build_vocab_transform()[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


def bleu_evaluation(transformer, test_iter):
    score = 0
    for sample, target in test_iter:
        prediction = translate(transformer, sample).split()
        target = target.split()
        if len(sample) < 3 or len(target) < 3:
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([target], prediction, weights=[0.5, 0.5])
        else:
            BLEUscore = nltk.translate.bleu_score.sentence_bleu([target], prediction)
        score += BLEUscore
    return score / len(test_iter)

def val_loss(model, val_iter, loss_fn, batch_size):
    model.eval()
    losses = 0
    val_dataloader = DataLoader(val_iter, batch_size=batch_size, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(val_dataloader))

