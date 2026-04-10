from __future__ import annotations

import argparse
import json

from .cpsat_model import Fill2DInstance, Fill2DItem, solve_fill2d


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve a simple 2D fill problem with CP-SAT.")
    parser.add_argument("--time-limit", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    instance = Fill2DInstance(
        pallet_width=12,
        pallet_height=10,
        items=(
            Fill2DItem("A", 6, 5, True),
            Fill2DItem("B", 6, 5, True),
            Fill2DItem("C", 6, 5, True),
            Fill2DItem("D", 6, 5, True),
            Fill2DItem("E", 4, 5, True),
            Fill2DItem("F", 4, 5, True),
        ),
    )
    result = solve_fill2d(instance, time_limit_seconds=args.time_limit, num_workers=args.workers)
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
