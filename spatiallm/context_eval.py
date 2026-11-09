"""Re-evaluate existing snapshots on a shared byte prefix without retraining."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from .runtime import environment, load_checkpoint, save_json, seed_everything, select_device
import torch
from .experiment import evaluate_partition, restrict_evaluation, summarize
from .model import LanguageModel, ModelConfig


def reevaluate(source, data, out, targets=70144, device_name="auto", batch_size=8):
    source, data, out = Path(source), Path(data), Path(out)
    if out.exists():
        raise FileExistsError("Choose a fresh output directory")
    original = json.loads((source / "results.json").read_text(encoding="utf-8"))
    partitions, prefix_hashes = {}, {}
    for part, expected in original["protocol"]["data_sha256"].items():
        raw = (data / f"{part}.txt").read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"{part} dataset differs from original experiment")
        if part != "train":
            partitions[part] = torch.tensor(list(raw), dtype=torch.long)
            prefix_hashes[part] = hashlib.sha256(raw[:targets + 1]).hexdigest()
    device = select_device(device_name)
    torch.set_num_threads(4)
    seed_everything(42)
    records = []
    for row in original["records"]:
        path = source / f"{row['architecture']}_seed{row['seed']}" / f"{row['comparison']}.pt"
        state = load_checkpoint(path)
        cfg = ModelConfig(**state["model_config"])
        selected = restrict_evaluation(partitions, cfg.max_seq_len, targets)
        model = LanguageModel(cfg).to(device)
        model.load_state_dict(state["model"])
        updated = copy.deepcopy(row)
        updated["original_evaluation"] = {key: row[key] for key in ("validation", "test")}
        for part in ("validation", "test"):
            updated[part] = evaluate_partition(model, selected[part], cfg.max_seq_len, batch_size, device)
        records.append(updated)
        print(json.dumps({"architecture": row["architecture"], "seed": row["seed"],
                          "comparison": row["comparison"], "test_loss": updated["test"]["loss"]}), flush=True)
        del model
    protocol = copy.deepcopy(original["protocol"])
    protocol["evaluation_prefix_sha256"] = prefix_hashes
    protocol["reevaluation"] = dict(source=str(source), targets=targets, environment=environment(device),
                                    note="Only evaluation changed; original weights, training times and training budgets preserved.")
    result = dict(protocol=protocol, records=records, **summarize(records))
    save_json(out / "results.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="runs/natural_v1")
    parser.add_argument("--data", default="data/natural")
    parser.add_argument("--out", default="runs/context128_common")
    parser.add_argument("--targets", type=int, default=70144)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    reevaluate(args.source, args.data, args.out, args.targets, args.device)
