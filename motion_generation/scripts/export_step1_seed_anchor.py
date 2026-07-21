#!/usr/bin/env python3
"""Export one observed causal multipart frame as a reusable Step 1 seed."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from utils.adaptive_anchor_tokens import validate_motion_payload  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a known neutral/current frame from exported causal body tokens"
    )
    parser.add_argument("--motion_token_json", type=Path, required=True)
    parser.add_argument("--frame_index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--description",
        type=str,
        default="User-selected neutral conversation-start seed",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.motion_token_json.resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    tokens = validate_motion_payload(payload, require_causal=True)
    index = args.frame_index
    if index < 0:
        index += len(tokens)
    if not 0 <= index < len(tokens):
        raise IndexError(f"frame_index {args.frame_index} is outside 0..{len(tokens) - 1}")
    output_payload = {
        "tokens": tokens[index],
        "layout": "body_16slot_512x4",
        "source_motion_token_json": str(source),
        "source_name": payload.get("name"),
        "source_frame_index": index,
        "description": args.description,
        "warning": "Verify this frame is visually neutral before using seed_mode=neutral or mixed_all.",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(output_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Exported seed:", output)
    print("Tokens:", output_payload["tokens"])


if __name__ == "__main__":
    main()
