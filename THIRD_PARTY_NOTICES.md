# Third-party materials

The project's MIT license covers its original code and documentation. It does not replace the terms of the following materials or claim their authorship.

## Jane Austen / Project Gutenberg

The natural-language corpus is Jane Austen's *Pride and Prejudice*, Project Gutenberg ebook 1342. The pinned source file is included as [`data/pride_and_prejudice_raw.txt`](data/pride_and_prejudice_raw.txt), with its original Gutenberg notice and full license intact. The prepared text partitions and token IDs derive from that source.

- Book: https://www.gutenberg.org/ebooks/1342
- Source URL recorded in the data manifest: https://www.gutenberg.org/cache/epub/1342/pg1342.txt
- Project Gutenberg license and trademark terms: https://www.gutenberg.org/policy/license.html

Gutenberg identifies this edition as usable without copyright restrictions in the United States. Rules in other jurisdictions may differ. Refer to the retained source notice and Gutenberg's terms when redistributing the corpus. The project's MIT notice does not relicense the source text or remove the Gutenberg terms.

## Tokenizer and dependencies

GPT-2 tokenization is provided by `tiktoken`; no pretrained GPT-2 model weights are included. Tokenizer assets are retrieved on first use and cached locally, outside tracked Git files. Their original terms apply. The repository contains token IDs generated from the pinned corpus, not a copy of the tokenizer's downloaded asset cache.

- tiktoken source and license: https://github.com/openai/tiktoken
- GPT-2 source and tokenizer assets: https://github.com/openai/gpt-2
- PyTorch source and license: https://github.com/pytorch/pytorch
- NumPy source and license: https://github.com/numpy/numpy

Dependencies are installed separately and retain their own licenses. See the lock files for the versions used locally.

## Research references

The related-work links in the READMEs credit the authors of existing sparse, geometric/content-based, and hybrid attention research. This project does not claim to have invented sparse attention or alternating local/global attention. Its own architecture and experiments should be evaluated on their specific implementation and evidence.
