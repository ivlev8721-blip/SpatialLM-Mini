"""Generate a small original bilingual corpus without network dependencies."""
import argparse
from pathlib import Path
import random


def make_demo(path, lines=3000, seed=123):
    rng = random.Random(seed)
    colors = ["red", "green", "blue", "white"]
    shapes = ["cube", "sphere", "pyramid", "plane"]
    relations = ["near", "above", "below", "behind"]
    records = []
    for _ in range(lines):
        color, shape, relation = rng.choice(colors), rng.choice(shapes), rng.choice(relations)
        other = rng.choice(shapes)
        x, y, z = [rng.randint(0, 9) for _ in range(3)]
        records.extend([
            f"The {color} {shape} is {relation} the {other}.\n",
            f"Point A has coordinates ({x}, {y}, {z}). Point B is near point A.\n",
            f"Точка А: ({x}, {y}, {z}). Куб рядом со сферой. Пространство имеет три измерения.\n",
        ])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    # Preserve the distributed Windows corpus byte-for-byte on every platform.
    path.write_text("".join(records), encoding="utf-8", newline="\r\n")
    print(f"Created {path}: {path.stat().st_size} UTF-8 bytes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/demo.txt")
    parser.add_argument("--lines", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    make_demo(args.out, args.lines, args.seed)
