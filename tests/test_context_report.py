import json
from pathlib import Path
import tempfile
import unittest

from tools.compare_contexts import compare


def fixture(context):
    args = dict(batch_size=1024 // context, seq_len=context, steps=500, seconds=45,
                dim=128, layers=4, heads=4, local_window=8, lr=0.0006,
                weight_decay=0.01, seeds=[42, 43, 44])
    records = []
    for index, arch in enumerate(("spatial", "transformer", "hybrid")):
        for seed in (42, 43, 44):
            for comparison in ("equal_steps", "equal_time"):
                loss = {"spatial": 2.2, "transformer": 2.0, "hybrid": 1.8}[arch] + 0.01 * (seed - 42)
                metric = dict(loss=loss, targets=70144)
                records.append(dict(architecture=arch, seed=seed, comparison=comparison,
                                    model_config=dict(max_seq_len=context), parameters=842624,
                                    steps=500, training_tokens=512000, training_seconds=45.0,
                                    training_tokens_per_second=512000 / 45,
                                    peak_allocated_mib=120.0, test=metric, validation=dict(metric)))
    return dict(protocol=dict(arguments=args, data_sha256=dict(train="train", validation="val", test="test"),
                              evaluation_prefix_sha256=dict(validation="same-val", test="same-test"),
                              precision="FP32"), records=records)


class ContextReportTests(unittest.TestCase):
    def write(self, root, changes=None):
        for context in (128, 256, 512):
            data = fixture(context)
            if changes and context == 512:
                changes(data)
            name = "context128_common" if context == 128 else f"context{context}_v1"
            out = Path(root) / name
            out.mkdir()
            (out / "results.json").write_text(json.dumps(data), encoding="utf-8")

    def test_paired_comparison_and_all_groups(self):
        with tempfile.TemporaryDirectory() as root:
            self.write(root)
            result = compare(root)
            self.assertEqual(len(result["summary"]), 18)
            for pair in result["paired_deltas"]:
                expected = -0.2 if pair["architecture"] == "hybrid" else 0.2
                self.assertAlmostEqual(pair["mean_delta"], expected)
                self.assertEqual(pair["wins"], 3 if expected < 0 else 0)

    def test_rejects_different_evaluation_prefix(self):
        with tempfile.TemporaryDirectory() as root:
            self.write(root, lambda d: d["protocol"]["evaluation_prefix_sha256"].update(test="different"))
            with self.assertRaisesRegex(ValueError, "evaluation positions"):
                compare(root)

    def test_rejects_unfair_budget_or_targets(self):
        changes = [lambda d: d["protocol"]["arguments"].update(batch_size=4),
                   lambda d: d["records"][0]["test"].update(targets=70000),
                   lambda d: d["records"][0].update(training_tokens=511000)]
        for change in changes:
            with tempfile.TemporaryDirectory() as root:
                self.write(root, change)
                with self.assertRaises(ValueError):
                    compare(root)


if __name__ == "__main__":
    unittest.main()
