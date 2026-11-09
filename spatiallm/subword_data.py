"""A fixed GPT-2 tokenizer, local cache and independently encoded text splits."""
import hashlib
import importlib.metadata
import os
from pathlib import Path
import struct

import torch


class GPT2Tokenizer:
    def __init__(self, cache=None):
        import tiktoken
        if cache is None:
            cache = Path(__file__).resolve().parents[1] / "data/gpt2_cache"
        os.environ["TIKTOKEN_CACHE_DIR"] = str(Path(cache).resolve())
        self.encoding = tiktoken.get_encoding("gpt2")
        self.vocab_size = self.encoding.n_vocab
        self.eot_id = self.encoding.eot_token

    def encode(self, text):
        return self.encoding.encode(text, disallowed_special=())

    def decode(self, tokens):
        return self.encoding.decode(tokens)


def prepare_subwords(text_dir, out, cache=None):
    from .runtime import save_json
    out = Path(out)
    if out.exists():
        raise FileExistsError("Prepared token data already exists")
    tokenizer = GPT2Tokenizer(cache)
    splits, info = {}, {}
    for part in ("train", "validation", "test"):
        raw = (Path(text_dir) / f"{part}.txt").read_bytes()
        ids = tokenizer.encode(raw.decode("utf-8"))
        splits[part] = torch.tensor(ids, dtype=torch.long)
        info[part] = dict(bytes=len(raw), tokens=len(ids), bytes_per_token=len(raw) / len(ids),
                          text_sha256=hashlib.sha256(raw).hexdigest(),
                          token_uint32_le_sha256=hashlib.sha256(struct.pack(f"<{len(ids)}I", *ids)).hexdigest())
    manifest = dict(tokenizer="gpt2", tiktoken=importlib.metadata.version("tiktoken"),
                    vocab_size=tokenizer.vocab_size, splits=info,
                    note="Fixed pretrained tokenizer only; no pretrained model weights. Each existing text split encoded independently.")
    out.mkdir(parents=True)
    torch.save(dict(format_version=1, manifest=manifest, splits=splits), out / "tokens.pt")
    save_json(out / "manifest.json", manifest)
    return manifest


def load_subwords(directory):
    state = torch.load(Path(directory) / "tokens.pt", weights_only=True, map_location="cpu")
    if state.get("format_version") != 1 or state["manifest"]["tokenizer"] != "gpt2":
        raise ValueError("Unsupported token data")
    for part, values in state["splits"].items():
        if values.ndim != 1 or values.dtype != torch.long or len(values) < 2:
            raise ValueError(f"Invalid {part} tokens")
        digest = hashlib.sha256(struct.pack(f"<{len(values)}I", *values.tolist())).hexdigest()
        if digest != state["manifest"]["splits"][part]["token_uint32_le_sha256"]:
            raise ValueError(f"{part} token checksum mismatch")
    return state["splits"], state["manifest"]
