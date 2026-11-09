import argparse
from pathlib import Path

from .runtime import load_checkpoint, seed_everything, select_device
import torch
from .data import ByteTokenizer
from .model import LanguageModel, ModelConfig


@torch.no_grad()
def generate(model, prompt, max_new_tokens=128, temperature=0.8, top_k=40, seed=42):
    if model.config.vocab_size != 257:
        raise ValueError("byte generation requires vocab_size=257; use subword16k generation")
    if max_new_tokens < 0 or temperature < 0 or top_k < 0:
        raise ValueError("max_new_tokens, temperature and top_k must be nonnegative")
    tokenizer = ByteTokenizer()
    ids = tokenizer.encode(prompt) or [tokenizer.bos_id]
    device = next(model.parameters()).device
    tokens = torch.tensor([ids], dtype=torch.long, device=device)
    generator = torch.Generator(device=device).manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        for _ in range(max_new_tokens):
            logits, _ = model(tokens[:, -model.config.max_seq_len:])
            scores = logits[:, -1].float()
            scores[:, tokenizer.bos_id] = -torch.inf
            if temperature == 0:
                next_token = scores.argmax(-1, keepdim=True)
            else:
                scores = scores / temperature
                if top_k:
                    threshold = torch.topk(scores, min(top_k, 256), dim=-1).values[:, -1:]
                    scores = scores.masked_fill(scores < threshold, -torch.inf)
                next_token = torch.multinomial(scores.softmax(-1), 1, generator=generator)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokenizer.decode(tokens[0].tolist())
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Space")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    seed_everything(args.seed)
    state = load_checkpoint(args.checkpoint)
    model = LanguageModel(ModelConfig(**state["model_config"])).to(select_device(args.device))
    model.load_state_dict(state["model"])
    text = generate(model, args.prompt, args.max_new_tokens, args.temperature, args.top_k, args.seed)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
