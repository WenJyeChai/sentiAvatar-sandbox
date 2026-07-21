from __future__ import annotations

import json

import pytest

from export_multipart_motion_tokens import (
    FORMAT_VERSION,
    assigned_shard_names,
    atomic_write_json,
    names_sha256,
    output_json_path,
    shard_manifest_name,
)
from verify_multipart_motion_token_export import verify_export


def common_metadata(names: list[str]) -> dict:
    part_order = ["upper", "lower", "feet", "hands"]
    token_layout = [
        {"slot": slot, "part": part, "quantizer": quantizer}
        for slot, (part, quantizer) in enumerate(
            (part, quantizer) for part in part_order for quantizer in range(4)
        )
    ]
    return {
        "format_version": FORMAT_VERSION,
        "split_names_sha256": names_sha256(names),
        "total_split_clips": len(names),
        "motion_fps": 20.0,
        "motion_token_fps": 10.0,
        "motion_token_unit_length": 2,
        "codebook_size": 512,
        "num_quantizers": 4,
        "part_order": part_order,
        "tokens_per_frame": 16,
        "token_layout": token_layout,
        "causal_by_part": {part: True for part in part_order},
        "body_causal": True,
        "checkpoint_fingerprints": {
            part: {"path": f"{part}.pth", "sha256": part * 8} for part in part_order
        },
        "math_mode": {
            "device_type": "cuda",
            "cudnn_allow_tf32": False,
            "cuda_matmul_allow_tf32": False,
        },
        "export_signature": "canonical-signature",
    }


def build_valid_export(tmp_path, names: list[str], num_shards: int = 2) -> None:
    common = common_metadata(names)
    for shard_id in range(num_shards):
        assigned = assigned_shard_names(names, num_shards, shard_id)
        manifest = {
            **common,
            "split_file": "all_file_list.txt",
            "motion_dir": "motion_data",
            "output_dir": str(tmp_path),
            "shard_id": shard_id,
            "num_shards": num_shards,
            "assigned_clips": len(assigned),
            "assigned_names_sha256": names_sha256(assigned),
            "max_clips": None,
            "counts": {
                "exported": len(assigned),
                "skipped_existing": 0,
                "invalid_existing_rewritten": 0,
                "missing": 0,
                "failed": 0,
            },
        }
        atomic_write_json(
            tmp_path / shard_manifest_name(num_shards, shard_id), manifest, indent=2
        )
    for index, name in enumerate(names):
        tokens = [[index % 512] * 16, [(index + 1) % 512] * 16]
        payload = {
            **{
                key: common[key]
                for key in (
                    "format_version",
                    "export_signature",
                    "motion_token_fps",
                    "motion_token_unit_length",
                    "codebook_size",
                    "num_quantizers",
                    "part_order",
                    "tokens_per_frame",
                    "token_layout",
                    "causal_by_part",
                    "body_causal",
                )
            },
            "name": name,
            "aligned_source_frames": 4,
            "tokens": tokens,
        }
        atomic_write_json(output_json_path(tmp_path, name), payload)


def test_modulo_shards_are_disjoint_complete_and_stable() -> None:
    names = [f"source/session/clip_{index}" for index in range(17)]
    shards = [assigned_shard_names(names, 4, shard_id) for shard_id in range(4)]
    assert sum(len(shard) for shard in shards) == len(names)
    assert set().union(*(set(shard) for shard in shards)) == set(names)
    assert all(set(left).isdisjoint(right) for i, left in enumerate(shards) for right in shards[i + 1 :])
    assert shards[0] == names[0::4]
    with pytest.raises(ValueError, match="num_shards"):
        assigned_shard_names(names, 4, 4)


def test_verifier_accepts_complete_export_and_rejects_corruption(tmp_path) -> None:
    names = [f"source/session/clip_{index}" for index in range(5)]
    build_valid_export(tmp_path, names)
    report = verify_export(output_dir=tmp_path, split_names=names, num_shards=2)
    assert report["status"] == "passed"
    assert report["verified_clips"] == 5
    assert report["total_token_frames"] == 10

    broken_path = output_json_path(tmp_path, names[2])
    broken = json.loads(broken_path.read_text(encoding="utf-8"))
    broken["tokens"][0][3] = 512
    atomic_write_json(broken_path, broken)
    with pytest.raises(ValueError, match="Invalid token payloads"):
        verify_export(output_dir=tmp_path, split_names=names, num_shards=2)
