#!/usr/bin/env python3
"""Verify and consolidate a sharded multipart motion-token export."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[2]
MODULE_DIR = PROJECT_DIR / "motion_generation"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from scripts.export_multipart_motion_tokens import (  # noqa: E402
    FORMAT_VERSION,
    assigned_shard_names,
    atomic_write_json,
    names_sha256,
    output_json_path,
    shard_manifest_name,
)
from utils.multipart_motion import load_name_list  # noqa: E402


CONSISTENT_KEYS = (
    "format_version",
    "split_names_sha256",
    "total_split_clips",
    "motion_fps",
    "motion_token_fps",
    "motion_token_unit_length",
    "codebook_size",
    "num_quantizers",
    "part_order",
    "tokens_per_frame",
    "token_layout",
    "causal_by_part",
    "body_causal",
    "checkpoint_fingerprints",
    "math_mode",
    "export_signature",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data_dir",
        type=Path,
        default=PROJECT_DIR / "SuSuInterActs" / "SuSuInterActs",
    )
    parser.add_argument("--split_file", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--allow_noncausal_body", action="store_true")
    parser.add_argument("--max_reported_errors", type=int, default=20)
    return parser.parse_args()


def load_json_mapping(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def validate_manifests(
    output_dir: Path,
    split_names: Sequence[str],
    num_shards: int,
    *,
    allow_noncausal_body: bool,
) -> tuple[list[dict], dict]:
    if num_shards < 1:
        raise ValueError("num_shards must be positive")
    manifests = []
    for shard_id in range(num_shards):
        path = output_dir / shard_manifest_name(num_shards, shard_id)
        if not path.is_file():
            raise FileNotFoundError(f"Missing shard manifest: {path}")
        manifest = load_json_mapping(path)
        manifest["_manifest_path"] = str(path)
        manifests.append(manifest)

    reference = manifests[0]
    for shard_id, manifest in enumerate(manifests):
        if manifest.get("shard_id") != shard_id:
            raise ValueError(
                f"Manifest {manifest['_manifest_path']} has shard_id={manifest.get('shard_id')}, "
                f"expected {shard_id}"
            )
        if manifest.get("num_shards") != num_shards:
            raise ValueError(f"Shard {shard_id} has inconsistent num_shards")
        if manifest.get("max_clips") is not None:
            raise ValueError(f"Shard {shard_id} is a truncated --max_clips export")
        assigned = assigned_shard_names(split_names, num_shards, shard_id)
        if manifest.get("assigned_clips") != len(assigned):
            raise ValueError(f"Shard {shard_id} assigned clip count is incorrect")
        if manifest.get("assigned_names_sha256") != names_sha256(assigned):
            raise ValueError(f"Shard {shard_id} assignment hash is incorrect")
        counts = manifest.get("counts")
        if not isinstance(counts, Mapping):
            raise ValueError(f"Shard {shard_id} has invalid counts")
        terminal_count = sum(
            int(counts.get(key, 0))
            for key in ("exported", "skipped_existing", "missing", "failed")
        )
        if terminal_count != len(assigned):
            raise ValueError(
                f"Shard {shard_id} terminal counts={terminal_count}, assigned={len(assigned)}"
            )
        if int(counts.get("missing", 0)) or int(counts.get("failed", 0)):
            raise ValueError(f"Shard {shard_id} contains missing/failed clips: {dict(counts)}")
        for key in CONSISTENT_KEYS:
            if manifest.get(key) != reference.get(key):
                raise ValueError(f"Shard {shard_id} differs from shard 0 for {key}")

    if reference.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"Export format {reference.get('format_version')} != expected {FORMAT_VERSION}"
        )
    if reference.get("total_split_clips") != len(split_names):
        raise ValueError("Shard manifests do not match the complete split size")
    if reference.get("split_names_sha256") != names_sha256(split_names):
        raise ValueError("Shard manifests do not match the complete split ordering")
    if not allow_noncausal_body and reference.get("body_causal") is not True:
        raise ValueError("Phase 1 export requires body_causal=true")
    math_mode = reference.get("math_mode")
    if not isinstance(math_mode, Mapping):
        raise ValueError("Missing strict inference math metadata")
    if math_mode.get("device_type") == "cuda":
        if math_mode.get("cudnn_allow_tf32") is not False:
            raise ValueError("CUDA export used cuDNN TF32")
        if math_mode.get("cuda_matmul_allow_tf32") is not False:
            raise ValueError("CUDA export used matmul TF32")

    aggregate_counts = Counter()
    for manifest in manifests:
        aggregate_counts.update({key: int(value) for key, value in manifest["counts"].items()})
    return manifests, dict(aggregate_counts)


def validate_token_payload(
    path: Path,
    name: str,
    reference: Mapping[str, object],
) -> tuple[int, int]:
    payload = load_json_mapping(path)
    expected_metadata = {
        "format_version": reference["format_version"],
        "export_signature": reference["export_signature"],
        "motion_token_fps": reference["motion_token_fps"],
        "motion_token_unit_length": reference["motion_token_unit_length"],
        "codebook_size": reference["codebook_size"],
        "num_quantizers": reference["num_quantizers"],
        "part_order": reference["part_order"],
        "tokens_per_frame": reference["tokens_per_frame"],
        "token_layout": reference["token_layout"],
        "causal_by_part": reference["causal_by_part"],
        "body_causal": reference["body_causal"],
    }
    if payload.get("name") != name:
        raise ValueError(f"payload name={payload.get('name')!r}, expected {name!r}")
    for key, expected in expected_metadata.items():
        if payload.get(key) != expected:
            raise ValueError(f"metadata mismatch for {key}")
    tokens = payload.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("tokens must be a non-empty list")
    slots = int(reference["tokens_per_frame"])
    codebook_size = int(reference["codebook_size"])
    for frame_index, frame in enumerate(tokens):
        if not isinstance(frame, list) or len(frame) != slots:
            raise ValueError(f"frame {frame_index} does not contain {slots} slots")
        if any(type(value) is not int or not 0 <= value < codebook_size for value in frame):
            raise ValueError(f"frame {frame_index} contains invalid code IDs")
    aligned_frames = payload.get("aligned_source_frames")
    if type(aligned_frames) is not int or aligned_frames < 1:
        raise ValueError("missing aligned_source_frames")
    expected_token_frames = aligned_frames // int(reference["motion_token_unit_length"])
    if len(tokens) != expected_token_frames:
        raise ValueError(
            f"token frames={len(tokens)}, expected {expected_token_frames} "
            f"from aligned source frames={aligned_frames}"
        )
    return len(tokens), slots * len(tokens)


def actual_token_names(output_dir: Path) -> set[str]:
    names = set()
    for path in output_dir.rglob("*.json"):
        if path.name.startswith("export_manifest"):
            continue
        names.add(path.relative_to(output_dir).with_suffix("").as_posix())
    return names


def verify_export(
    *,
    output_dir: Path,
    split_names: Sequence[str],
    num_shards: int,
    allow_noncausal_body: bool = False,
    max_reported_errors: int = 20,
) -> dict:
    if len(split_names) != len(set(split_names)):
        raise ValueError("Split file contains duplicate clip names")
    manifests, aggregate_counts = validate_manifests(
        output_dir,
        split_names,
        num_shards,
        allow_noncausal_body=allow_noncausal_body,
    )
    reference = manifests[0]
    expected_names = set(split_names)
    actual_names = actual_token_names(output_dir)
    missing_files = sorted(expected_names - actual_names)
    unexpected_files = sorted(actual_names - expected_names)
    if missing_files or unexpected_files:
        raise ValueError(
            f"Token file coverage mismatch: missing={missing_files[:10]}, "
            f"unexpected={unexpected_files[:10]}"
        )

    errors = []
    total_token_frames = 0
    total_ids = 0
    min_token_frames = None
    max_token_frames = 0
    for index, name in enumerate(split_names, start=1):
        path = output_json_path(output_dir, name)
        try:
            token_frames, ids = validate_token_payload(path, name, reference)
            total_token_frames += token_frames
            total_ids += ids
            min_token_frames = (
                token_frames if min_token_frames is None else min(min_token_frames, token_frames)
            )
            max_token_frames = max(max_token_frames, token_frames)
        except Exception as exc:  # report several corrupted files together
            if len(errors) < max_reported_errors:
                errors.append({"name": name, "error": f"{type(exc).__name__}: {exc}"})
        if index % 1_000 == 0:
            print(f"Verified {index}/{len(split_names)} token files; errors={len(errors)}")
    if errors:
        raise ValueError(f"Invalid token payloads (first {len(errors)}): {errors}")

    return {
        "format_version": FORMAT_VERSION,
        "status": "passed",
        "verified_clips": len(split_names),
        "num_shards": num_shards,
        "split_names_sha256": names_sha256(split_names),
        "export_signature": reference["export_signature"],
        "body_causal": reference["body_causal"],
        "causal_by_part": reference["causal_by_part"],
        "part_order": reference["part_order"],
        "tokens_per_frame": reference["tokens_per_frame"],
        "codebook_size": reference["codebook_size"],
        "num_quantizers": reference["num_quantizers"],
        "motion_token_fps": reference["motion_token_fps"],
        "checkpoint_fingerprints": reference["checkpoint_fingerprints"],
        "math_mode": reference["math_mode"],
        "aggregate_counts": aggregate_counts,
        "total_token_frames": total_token_frames,
        "total_token_ids": total_ids,
        "min_token_frames_per_clip": min_token_frames,
        "max_token_frames_per_clip": max_token_frames,
        "shard_manifests": [manifest["_manifest_path"] for manifest in manifests],
    }


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    split_file = (args.split_file or data_dir / "split" / "all_file_list.txt").resolve()
    output_dir = (
        args.output_dir or data_dir / "motion_token_data_multipart_causal_512x4"
    ).resolve()
    split_names = load_name_list(split_file)
    report = verify_export(
        output_dir=output_dir,
        split_names=split_names,
        num_shards=int(args.num_shards),
        allow_noncausal_body=bool(args.allow_noncausal_body),
        max_reported_errors=int(args.max_reported_errors),
    )
    report.update(
        {
            "split_file": str(split_file),
            "output_dir": str(output_dir),
        }
    )
    final_manifest = output_dir / "export_manifest.json"
    atomic_write_json(final_manifest, report, indent=2)
    print(json.dumps(report, indent=2))
    print(f"PASS: verified complete multipart token export -> {final_manifest}")


if __name__ == "__main__":
    main()
