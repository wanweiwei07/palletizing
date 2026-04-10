from __future__ import annotations

import argparse
import json

from palletizing import BoxSpec, PalletSpec, plan_palletizing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan a dense palletizing layout for identical boxes.")
    parser.add_argument("--pallet-length", type=float, required=True)
    parser.add_argument("--pallet-width", type=float, required=True)
    parser.add_argument("--pallet-max-height", type=float, required=True)
    parser.add_argument("--box-length", type=float, required=True)
    parser.add_argument("--box-width", type=float, required=True)
    parser.add_argument("--box-height", type=float, required=True)
    parser.add_argument("--count", type=int, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    pallet = PalletSpec(
        length=args.pallet_length,
        width=args.pallet_width,
        max_height=args.pallet_max_height,
    )
    box = BoxSpec(
        length=args.box_length,
        width=args.box_width,
        height=args.box_height,
        count=args.count,
    )
    result = plan_palletizing(pallet, box)
    print(json.dumps(result.as_dict(), indent=2))


if __name__ == "__main__":
    main()
