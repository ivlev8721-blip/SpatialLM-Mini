import copy
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from spatiallm.runtime import load_checkpoint, save_checkpoint, seed_everything
import torch
from spatiallm.data import ByteTokenizer, load_corpus, sample_batch
from spatiallm.generate import generate
from spatiallm.model import LanguageModel, ModelConfig, SpatialAttention
from spatiallm.train import train


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        seed_everything(123)

    def model(self, architecture="spatial", **kw):
        return LanguageModel(ModelConfig(architecture=architecture, dim=32, layers=2,
                                        heads=4, max_seq_len=32, **kw))

    def test_tokenizer_unicode(self):
        tok = ByteTokenizer()
        text = "Привет, geometry! 🌍\n"
        self.assertEqual(tok.decode(tok.encode(text)), text)
        self.assertEqual(tok.decode([256, 65]), "A")

    def test_shapes_and_finite_backward(self):
        for arch in ("spatial", "transformer"):
            model = self.model(arch)
            for t in (1, 17, 32):
                x = torch.randint(256, (2, t))
                logits, loss = model(x, x)
                self.assertEqual(logits.shape, (2, t, 257))
                loss.backward()
                self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                    for p in model.parameters()))
                model.zero_grad(set_to_none=True)

    def test_no_future_leakage(self):
        for arch in ("spatial", "transformer"):
            model = self.model(arch).eval()
            x = torch.randint(256, (2, 24))
            y = x.clone()
            y[:, 11:] = (y[:, 11:] + 71) % 256
            with torch.no_grad():
                a, _ = model(x)
                b, _ = model(y)
                prefix, _ = model(x[:, :11])
            torch.testing.assert_close(a[:, :11], b[:, :11], rtol=0, atol=0)
            torch.testing.assert_close(a[:, :11], prefix, rtol=1e-5, atol=1e-6)

    def test_geometric_parameters_learn(self):
        model = self.model()
        x, y = torch.randint(256, (2, 24)), torch.randint(256, (2, 24))
        model(x, y)[1].backward()
        for block in model.blocks:
            self.assertGreater(block.mix.coordinates.weight.grad.abs().sum().item(), 0)
            self.assertGreater(block.mix.log_scale.grad.abs().sum().item(), 0)

    def test_sparse_causal_topology(self):
        mix = SpatialAttention(ModelConfig(max_seq_len=1024, local_window=8))
        self.assertLess(mix.indices.numel(), 1024 * 20)
        for t in (0, 1, 9, 1023):
            edges = mix.indices[t][mix.valid[t]]
            self.assertTrue((edges <= t).all())
            self.assertEqual(edges.numel(), edges.unique().numel())
            self.assertIn(t, edges.tolist())

    def test_bad_configuration_and_context(self):
        for kwargs in (dict(dim=33, heads=4), dict(layers=0), dict(architecture="bad"),
                       dict(local_window=0), dict(dropout=1)):
            with self.assertRaises(ValueError):
                ModelConfig(**kwargs)
        for size in (0, 33):
            with self.assertRaises(ValueError):
                self.model()(torch.zeros((1, size), dtype=torch.long))

    def test_generation_reproducible_and_crops(self):
        model = self.model()
        prompt = "Точка " * 10
        a = generate(model, prompt, 8, seed=7)
        self.assertEqual(a, generate(model, prompt, 8, seed=7))
        self.assertTrue(a.startswith(prompt))
        self.assertTrue(model.training)
        self.assertEqual(generate(model, "Привет", 0), "Привет")
        self.assertIsInstance(generate(model, "", 3), str)
        with self.assertRaises(ValueError):
            generate(model, "x", temperature=-1)

    def test_overfits_tiny_batch(self):
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        x = torch.tensor([[65, 66, 67, 65, 66, 67] for _ in range(2)])
        y = torch.tensor([[66, 67, 65, 66, 67, 65] for _ in range(2)])
        initial = model(x, y)[1].item()
        for _ in range(35):
            optimizer.zero_grad()
            loss = model(x, y)[1]
            loss.backward()
            optimizer.step()
        self.assertLess(model(x, y)[1].item(), initial * 0.1)


class PipelineTests(unittest.TestCase):
    def test_data_split_and_shift(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "text.txt"
            raw = bytes(range(32, 127)) * 20
            path.write_bytes(raw)
            a, b, digest = load_corpus(path, 16)
            self.assertEqual(a.tolist() + b.tolist(), list(raw))
            self.assertEqual(len(digest), 64)
            g = torch.Generator().manual_seed(1)
            x, y = sample_batch(a, 4, 16, g, "cpu")
            torch.testing.assert_close(x[:, 1:], y[:, :-1])
            path.write_text("short", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_corpus(path, 16)

    def test_checkpoint_resume_exact_and_reject_changed_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "data.txt"
            path.write_text("Geometry is language. Точка рядом с кубом.\n" * 100, encoding="utf-8")
            config = {"model": dict(dim=32, layers=1, heads=4, max_seq_len=32, dropout=0.1),
                      "training": dict(steps=6, seq_len=16, batch_size=2, grad_accum=2,
                                       eval_every=3, eval_batches=1, seed=12)}
            with redirect_stdout(io.StringIO()):
                whole, _ = train(config, path, root / "whole", "cpu", threads=2)
                part = copy.deepcopy(config)
                part["training"]["steps"] = 3
                train(part, path, root / "part", "cpu", threads=2)
                # v0.2 checkpoints lack the newly optional model fields.
                legacy = load_checkpoint(root / "part/last.pt")
                legacy["model_config"].pop("ff_dim")
                legacy["model_config"].pop("hybrid_every")
                save_checkpoint(root / "part/last.pt", legacy)
                resumed, _ = train(config, path, root / "part", "cpu", root / "part/last.pt", threads=2)
            for name, tensor in whole.state_dict().items():
                torch.testing.assert_close(tensor, resumed.state_dict()[name], rtol=0, atol=0)
            saved = load_checkpoint(root / "part/last.pt")
            self.assertEqual(saved["step"], 6)
            with self.assertRaises(FileExistsError):
                train(config, path, root / "part", "cpu", threads=2)
            path.write_text("Changed geometry.\n" * 100, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                train(config, path, root / "other", "cpu", root / "part/last.pt", threads=2)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA-enabled PyTorch and GPU required")
    def test_cuda_fp16_train_generate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "data.txt"
            path.write_text("Space has three dimensions.\n" * 100, encoding="utf-8")
            config = {"model": dict(dim=32, layers=1, heads=4, max_seq_len=32),
                      "training": dict(steps=3, seq_len=16, batch_size=2, amp=True, eval_batches=1)}
            with redirect_stdout(io.StringIO()):
                model, history = train(config, path, root / "run", "cuda")
            self.assertTrue(torch.isfinite(torch.tensor(history[-1]["val_loss"])))
            self.assertTrue(generate(model, "Space", 4).startswith("Space"))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA-enabled PyTorch and GPU required")
    def test_cuda_fp16_resume_exact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = root / "data.txt"
            path.write_text("A cube is near a sphere.\n" * 100, encoding="utf-8")
            config = {"model": dict(dim=32, layers=1, heads=4, max_seq_len=32, dropout=0.1),
                      "training": dict(steps=6, seq_len=16, batch_size=2, amp=True,
                                       grad_accum=2, eval_every=3, eval_batches=1)}
            with redirect_stdout(io.StringIO()):
                whole, _ = train(config, path, root / "whole", "cuda")
                part = copy.deepcopy(config)
                part["training"]["steps"] = 3
                train(part, path, root / "part", "cuda")
                resumed, _ = train(config, path, root / "part", "cuda", root / "part/last.pt")
            for name, tensor in whole.state_dict().items():
                torch.testing.assert_close(tensor, resumed.state_dict()[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
