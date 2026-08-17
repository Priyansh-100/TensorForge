"""
Seq2seq training library: datasets, training loop, greedy decoding.

Tasks:
  reverse  — reverse a sequence of integers ([2,5,1,3] -> [3,1,5,2])
  copy     — copy a sequence of integers ([2,5,1,3] -> [2,5,1,3])

CLI entry points: scripts/train_seq2seq.py (single process),
scripts/train_dist.py (DistributedDataParallel).
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformer.model import Transformer, create_masks, create_look_ahead_mask, NoamSchedule

SOS_TOKEN = 1  # start-of-sequence
PAD_TOKEN = 0


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class Seq2SeqDataset(Dataset):
    """Base class: random sequences with a target transformation."""
    SOS_TOKEN = SOS_TOKEN
    PAD_TOKEN = PAD_TOKEN

    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        # Tokens in [2, vocab_size) to avoid SOS and PAD
        self.data = torch.randint(2, vocab_size, (num_samples, seq_len))
        self.labels = self.transform(self.data)

    @staticmethod
    def transform(data: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        src = self.data[idx]
        tgt = self.labels[idx]
        sos = torch.full((1,), self.SOS_TOKEN, dtype=torch.long)
        tgt_input = torch.cat([sos, tgt[:-1]])
        tgt_output = tgt
        return src, tgt_input, tgt_output


class ReverseDataset(Seq2SeqDataset):
    @staticmethod
    def transform(data: torch.Tensor) -> torch.Tensor:
        return data.flip(dims=[1])


class CopyDataset(Seq2SeqDataset):
    @staticmethod
    def transform(data: torch.Tensor) -> torch.Tensor:
        return data.clone()


DATASETS = {"reverse": ReverseDataset, "copy": CopyDataset}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(task: str, epochs: int, save_path: str = "model.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Hyperparams (small for fast CPU/MPS training)
    VOCAB_SIZE = 20
    SEQ_LEN = 5
    D_MODEL = 64
    NUM_HEADS = 4
    D_FF = 128
    NUM_LAYERS = 2
    BATCH_SIZE = 32

    model = Transformer(
        src_vocab_size=VOCAB_SIZE,
        tgt_vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        num_layers=NUM_LAYERS,
        max_len=SEQ_LEN + 1,
    ).to(device)

    print(f"Model parameters: {model.count_params():,}")

    dataset_cls = DATASETS[task]
    train_data = dataset_cls(num_samples=2000, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)

    pad_idx = PAD_TOKEN

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    # The NoamSchedule lambda returns the full paper formula, so base LR = 1.0
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)
    scheduler = NoamSchedule(optimizer, d_model=D_MODEL, warmup_steps=400)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for src, tgt_input, tgt_output in train_loader:
            src, tgt_input, tgt_output = src.to(device), tgt_input.to(device), tgt_output.to(device)

            src_mask, tgt_mask = create_masks(src, tgt_input, pad_idx)

            optimizer.zero_grad()
            logits = model(src, tgt_input, src_mask, tgt_mask)
            loss = criterion(logits.view(-1, logits.size(-1)), tgt_output.view(-1))
            loss.backward()
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        if epoch % 10 == 0:
            avg_loss = total_loss / len(train_loader)
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch:2d} | Loss: {avg_loss:.4f} | LR: {lr_now:.2e}")

    torch.save(model.state_dict(), save_path)
    print(f"Training done! Model saved to {save_path}")
    return model, device


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def greedy_decode(model, src, device, max_len=10, sos_idx=SOS_TOKEN):
    """Autoregressive decoding."""
    model.eval()
    src = src.to(device)
    src_mask = (src != PAD_TOKEN).unsqueeze(1).unsqueeze(2)

    with torch.no_grad():
        enc_output = model.encoder(src, src_mask)

    tgt = torch.full((src.size(0), 1), sos_idx, dtype=torch.long, device=device)

    for _ in range(max_len):
        look_ahead = create_look_ahead_mask(tgt.size(1)).to(device)
        tgt_pad_mask = (tgt != PAD_TOKEN).unsqueeze(1).unsqueeze(2)
        tgt_mask = tgt_pad_mask & look_ahead

        with torch.no_grad():
            dec_output = model.decoder(tgt, enc_output, tgt_mask, src_mask)
            logits = model.output_proj(dec_output)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

        tgt = torch.cat([tgt, next_token], dim=1)

    return tgt[:, 1:]  # strip SOS


def evaluate(model, device, task: str):
    model.eval()
    vocab_size = 20
    seq_len = 5

    test_src = torch.randint(2, vocab_size, (5, seq_len))
    expected_transform = "flip" if task == "reverse" else "copy"

    print("\n--- Inference examples ---")
    for i in range(test_src.size(0)):
        src_seq = test_src[i].tolist()
        predicted = greedy_decode(model, test_src[i].unsqueeze(0), device, max_len=seq_len)
        pred_seq = predicted[0].tolist()
        expected = src_seq[::-1] if expected_transform == "flip" else src_seq[:]
        print(f"Input:    {src_seq}")
        print(f"Expected: {expected}")
        print(f"Predicted: {pred_seq}")
        print(f"{'✓' if pred_seq == expected else '✗'}")
        print()