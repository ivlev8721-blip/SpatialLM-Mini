from dataclasses import asdict
import unittest

from spatiallm.runtime import seed_everything
import torch
from spatiallm.experiment import evaluate_partition, matched_configs, summarize, restrict_evaluation
from spatiallm.model import LanguageModel, ModelConfig, SpatialAttention, TransformerAttention


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        seed_everything(42)

    def test_hybrid_schedule_causality_and_gradients(self):
        model = LanguageModel(ModelConfig(architecture="hybrid", dim=32, layers=4, heads=4, max_seq_len=32))
        self.assertEqual([type(b.mix) for b in model.blocks],
                         [SpatialAttention, TransformerAttention, SpatialAttention, TransformerAttention])
        x = torch.randint(256, (2, 24))
        changed = x.clone()
        changed[:, 12:] = (changed[:, 12:] + 1) % 256
        a = model(x)[0]
        b = model(changed)[0]
        torch.testing.assert_close(a[:, :12], b[:, :12], rtol=0, atol=0)
        model(x, changed)[1].backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertGreater(model.blocks[0].mix.coordinates.weight.grad.abs().sum().item(), 0)

    def test_parameter_matching(self):
        configs = matched_configs(128, 4, 4, 128, 8)
        counts = [LanguageModel(c).parameter_count() for c in configs.values()]
        self.assertLess((max(counts) - min(counts)) / min(counts), 0.002)
        self.assertEqual(len({c.dim for c in configs.values()}), 1)

    def test_partition_loss_weighted_over_partial_batch(self):
        model = LanguageModel(ModelConfig(dim=16, heads=2, layers=1, max_seq_len=8)).eval()
        tokens = torch.arange(39, dtype=torch.long)
        result = evaluate_partition(model, tokens, 8, 3, torch.device("cpu"))
        losses = [model(tokens[i:i+8][None], tokens[i+1:i+9][None])[1].item() for i in (0, 8, 16, 24)]
        self.assertAlmostEqual(result["loss"], sum(losses) / 4, places=5)
        self.assertEqual(result["targets"], 32)
        self.assertEqual(result["unused_tail_bytes"], 6)

    def test_legacy_config_shapes_and_hybrid_validation(self):
        legacy = dict(dim=32, layers=2, heads=4, max_seq_len=32)
        cfg = ModelConfig(**legacy)
        state = LanguageModel(cfg).state_dict()
        normalized = ModelConfig(**asdict(cfg))
        LanguageModel(normalized).load_state_dict(state)
        for kwargs in (dict(architecture="hybrid", layers=1),
                       dict(architecture="hybrid", layers=4, hybrid_every=2.5), dict(ff_dim=-1)):
            with self.assertRaises(ValueError):
                ModelConfig(**kwargs)

    def test_common_evaluation_prefix_preserves_training_data(self):
        parts = {name: torch.arange(n) for name, n in (("train", 100), ("validation", 90), ("test", 80))}
        chosen = restrict_evaluation(parts, 16, 64)
        self.assertIs(chosen["train"], parts["train"])
        self.assertEqual(len(chosen["validation"]), 65)
        self.assertEqual(len(chosen["test"]), 65)
        self.assertEqual(len(parts["validation"]), 90)
        for context in (8, 16, 32, 64):
            selected = restrict_evaluation(parts, context, 64)
            torch.testing.assert_close(selected["test"][1:], parts["test"][1:65])
        for target in (-1, 63, 96):
            with self.assertRaises(ValueError):
                restrict_evaluation(parts, 16, target)

    def test_long_context_causality_and_backward(self):
        for architecture in ("spatial", "transformer", "hybrid"):
            model = LanguageModel(ModelConfig(architecture=architecture, dim=32, layers=2, heads=4, max_seq_len=512))
            x = torch.randint(256, (1, 512))
            changed = x.clone()
            changed[:, 384:] = (changed[:, 384:] + 31) % 256
            with torch.no_grad():
                before, _ = model(x)
                after, _ = model(changed)
            torch.testing.assert_close(before[:, :384], after[:, :384], rtol=0, atol=0)
            model(x, changed)[1].backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))


if __name__ == "__main__":
    unittest.main()
