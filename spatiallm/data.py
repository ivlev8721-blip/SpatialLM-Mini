import hashlib
from pathlib import Path

import torch


class ByteTokenizer:
    bos_id = 256
    vocab_size = 257

    def encode(self, text):
        return list(text.encode("utf-8"))

    def decode(self, ids):
        return bytes(i for i in ids if 0 <= i < 256).decode("utf-8", errors="replace")


def load_corpus(path, seq_len, val_fraction=0.1):
    if not 0 < val_fraction < 1:
        raise ValueError("val_fraction must be in (0, 1)")
    raw = Path(path).read_bytes()
    raw.decode("utf-8")  # Fail clearly for non-UTF-8 input.
    boundary = int(len(raw) * (1 - val_fraction))
    if min(boundary, len(raw) - boundary) < seq_len + 1:
        raise ValueError("corpus too short: each split needs at least seq_len+1 bytes")
    tokens = torch.tensor(list(raw), dtype=torch.long)
    return tokens[:boundary], tokens[boundary:], hashlib.sha256(raw).hexdigest()


def sample_batch(tokens, batch_size, seq_len, generator, device):
    starts = torch.randint(len(tokens) - seq_len, (batch_size,), generator=generator)
    indices = starts[:, None] + torch.arange(seq_len + 1)[None, :]
    batch = tokens[indices].to(device)
    return batch[:, :-1], batch[:, 1:]
