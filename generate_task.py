from __future__ import annotations

import argparse
import json

from palletizing import generate_multitype_task


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a multitype palletizing task with repeated box types.")
    parser.add_argument("--count", type=int, default=70)
    parser.add_argument("--seed", type=int, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    task = generate_multitype_task(count=args.count, seed=args.seed)
    print(json.dumps(task.as_dict(), indent=2))


if __name__ == "__main__":
    main()
