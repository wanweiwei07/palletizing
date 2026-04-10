from __future__ import annotations

import argparse
import json
from pathlib import Path

from palletizing import analyze_box_heights, load_task_boxes


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze box-height groups and stacked block candidates.")
    parser.add_argument("--task-file", type=Path, required=True, help="Path to a generated task JSON file.")
    parser.add_argument(
        "--max-combination-size",
        type=int,
        default=3,
        help="Maximum stacked box count to consider when matching target heights.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path. Prints to stdout when omitted.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    boxes = load_task_boxes(args.task_file)
    analysis = analyze_box_heights(boxes, max_combination_size=args.max_combination_size)
    payload = analysis.as_dict()

    if args.output is None:
        print(json.dumps(payload, indent=2))
        return

    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved height analysis to {args.output}")


if __name__ == "__main__":
    main()
