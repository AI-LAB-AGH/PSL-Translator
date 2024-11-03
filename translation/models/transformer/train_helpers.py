import torch
from processing_helpers import create_mask
from timeit import default_timer as timer
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_epoch(model, optimizer, dataloader, loss_fn):
    model.train()
    losses = 0

    for src, tgt in dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(list(dataloader))


NUM_EPOCHS_PARAPHRASE = 10
NUM_EPOCHS_GEC = 10
NUM_EPOCHS_TRANSLATION = 10


def train(num_epochs, train_dataloader, val_dataloader, transformer, optimizer, loss_fn):
    for epoch in range(1, num_epochs + 1):
        start_time = timer()
        train_loss = train_epoch(transformer, optimizer, train_dataloader, loss_fn)
        end_time = timer()
        val_loss = evaluate(transformer, val_dataloader, loss_fn)
        print((
            f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

def evaluate(model, dataloader, loss_fn):
    model.eval()
    losses = 0

    for src, tgt in dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(list(dataloader))