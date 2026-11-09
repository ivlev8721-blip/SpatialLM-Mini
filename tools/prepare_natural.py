"""Prepare a pinned local Gutenberg text; retain the raw file and its licence."""
import argparse
import hashlib
import json
from pathlib import Path


def prepare(raw_path, out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    raw = Path(raw_path).read_bytes()
    text = raw.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
    # Remove publisher material and the preface, not just the Gutenberg header.
    start = text.index("It is a truth universally acknowledged")
    end = text.index("*** END OF THE PROJECT GUTENBERG EBOOK", start)
    body = text[start:end].strip() + "\n"
    out.mkdir(parents=True)
    # Contiguous character splits avoid cutting UTF-8 codepoints. Each partition
    # gets its own byte sequence; no training or evaluation window crosses it.
    a, b = int(len(body) * 0.8), int(len(body) * 0.9)
    parts = {"train": body[:a], "validation": body[a:b], "test": body[b:]}
    meta = {"source": "https://www.gutenberg.org/cache/epub/1342/pg1342.txt",
            "book": "Pride and Prejudice — Jane Austen",
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "clean_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
            "split_method": "contiguous characters, 80/10/10, then UTF-8 encoding",
            "splits": {}}
    for name, content in parts.items():
        blob = content.encode("utf-8")
        (out / f"{name}.txt").write_bytes(blob)
        meta["splits"][name] = {"bytes": len(blob), "sha256": hashlib.sha256(blob).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/pride_and_prejudice_raw.txt")
    parser.add_argument("--out", default="data/natural")
    args = parser.parse_args()
    print(json.dumps(prepare(args.raw, args.out), ensure_ascii=False, indent=2))
