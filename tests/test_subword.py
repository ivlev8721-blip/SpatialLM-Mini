import copy
import importlib.util
import unittest
import torch

from spatiallm.model import LanguageModel, ModelConfig
from spatiallm.runtime import seed_everything


class SubwordModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        seed_everything(42)

    def test_chunk_loss_and_all_gradients_equal_full_output(self):
        for arch in ("spatial", "transformer", "hybrid"):
            a = LanguageModel(ModelConfig(architecture=arch, vocab_size=521, dim=32, layers=2,
                                         heads=4, max_seq_len=16, dropout=0.1))
            b = copy.deepcopy(a)
            b.gradient_checkpointing = True
            x, y = torch.randint(521, (2, 13)), torch.randint(521, (2, 13))
            seed_everything(123)
            full = a(x, y)[1]
            seed_everything(123)
            chunked = b.forward_loss(x, y, chunk_size=7)
            torch.testing.assert_close(full, chunked)
            full.backward()
            chunked.backward()
            for (name, p), (_, q) in zip(a.named_parameters(), b.named_parameters()):
                torch.testing.assert_close(p.grad, q.grad, rtol=2e-4, atol=2e-6, msg=name)

    def test_chunked_eval_matches_full_for_incomplete_chunk(self):
        model = LanguageModel(ModelConfig(vocab_size=521, dim=16, heads=2, layers=1, max_seq_len=16)).eval()
        x, y = torch.randint(521, (2, 13)), torch.randint(521, (2, 13))
        with torch.no_grad():
            torch.testing.assert_close(model(x, y)[1], model.forward_loss(x, y, chunk_size=8))
        with self.assertRaises(ValueError):
            model.forward_loss(x, y, chunk_size=0)

    def test_old_byte_generator_rejects_subword_model(self):
        from spatiallm.generate import generate
        model = LanguageModel(ModelConfig(vocab_size=521, dim=16, heads=2, layers=1))
        with self.assertRaisesRegex(ValueError, "byte generation"):
            generate(model, "hello", 1)

    @unittest.skipUnless(importlib.util.find_spec("tiktoken"), "optional tiktoken dependency required")
    def test_gpt2_roundtrip_and_cached_token_checksums(self):
        from spatiallm.subword_data import GPT2Tokenizer, load_subwords
        from pathlib import Path
        root = Path(__file__).resolve().parents[1]
        if not (root / "data/subword16k/tokens.pt").exists():
            self.skipTest("prepared subword data required")
        tok = GPT2Tokenizer()
        text = "Hello, geometry! Привет 🌍 <|endoftext|>"
        self.assertEqual(tok.decode(tok.encode(text)), text)
        self.assertEqual(tok.vocab_size, 50257)
        splits, manifest = load_subwords(root / "data/subword16k")
        self.assertEqual(len(splits["test"]), manifest["splits"]["test"]["tokens"])
        self.assertGreater(len(splits["validation"]), 16384)


if __name__ == "__main__":
    unittest.main()
