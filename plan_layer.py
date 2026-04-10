from __future__ import annotations

import argparse
import json
from pathlib import Path

from palletizing import PalletSpec, load_task_boxes, plan_best_layer_pattern


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a single pallet layer pattern from task boxes.")
    parser.add_argument("--task-file", type=Path, required=True, help="Path to a generated task JSON file.")
    parser.add_argument("--pallet-length", type=float, default=1.2)
    parser.add_argument("--pallet-width", type=float, default=1.0)
    parser.add_argument("--pallet-max-height", type=float, default=1.5)
    parser.add_argument("--max-stack-size", type=int, default=3)
    parser.add_argument("--output", type=Path, default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    boxes = load_task_boxes(args.task_file)
    pallet = PalletSpec(
        length=args.pallet_length,
        width=args.pallet_width,
        max_height=args.pallet_max_height,
    )
    result = plan_best_layer_pattern(
        boxes=boxes,
        pallet=pallet,
        max_stack_size=args.max_stack_size,
    )
    payload = result.as_dict()

    if args.output is None:
        print(json.dumps(payload, indent=2))
        return

    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved layer pattern to {args.output}")


if __name__ == "__main__":
    main()
