#!/usr/bin/env python3
"""Build deterministic, balanced, nested Step 1 training subsets.

The sampler uses only the official training split.  It balances capture source
and session, duration, text length, annotation availability, and a
codec-independent raw-motion dynamics score.  MSD and learned-codec outputs
are deliberately not required.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_DIR / "SuSuInterActs" / "SuSuInterActs"
DEFAULT_OUTPUT_DIR = PROJECT_DIR / "motion_generation" / "data_splits" / "step1_balanced_seed42"
TAG_PATTERN = re.compile(r"【(表情|动作)[：:]([^】]*)】")
MISSING_TAG = "<unannotated>"
METADATA_SCHEMA_VERSION = 3


@dataclass(frozen=True)
class ClipMetadata:
    name: str
    source: str
    session: str
    duration_seconds: float
    text_characters: int
    annotation_pattern: str
    expression: str
    action: str
    joint_speed_mean: float
    joint_speed_p90: float
    body_rotation_speed: float
    body_rotation_p90: float
    hand_rotation_speed: float
    hand_rotation_p90: float
    joint_positions_available: bool = True
    complexity_score: float = 0.0
    duration_bin: str = ""
    text_length_bin: str = ""
    complexity_bin: str = ""


BALANCE_SPECS: dict[str, tuple[float, float]] = {
    # attribute: (importance in the greedy score, frequency temperature)
    # temperature 1 preserves the dataset distribution; 0 is uniform.
    "source": (1.00, 0.75),
    "session": (0.75, 0.80),
    "duration_bin": (1.00, 0.00),
    "complexity_bin": (1.50, 0.00),
    "text_length_bin": (0.50, 0.00),
    "annotation_pattern": (1.00, 0.85),
    # Exact free-text labels are sparse, so they receive modest weights.  A
    # tempered target improves diversity without forcing a uniform taxonomy.
    "expression": (0.05, 0.75),
    "action": (0.05, 0.75),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--train_split", type=Path, default=None)
    parser.add_argument("--val_split", type=Path, default=None)
    parser.add_argument("--test_split", type=Path, default=None)
    parser.add_argument("--text_json", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--subset",
        action="append",
        default=None,
        metavar="NAME=SIZE",
        help="Nested subset to write; repeat as needed (default: smoke=512, pilot=2000, main=6000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--motion_fps", type=float, default=20.0)
    parser.add_argument("--quantile_bins", type=int, default=5)
    parser.add_argument("--metadata_cache", type=Path, default=None)
    parser.add_argument("--refresh_metadata", action="store_true")
    parser.add_argument("--progress_every", type=int, default=500)
    return parser.parse_args()


def normalize_name(value: str) -> str:
    return value.strip().replace("\\", "/")


def read_names(path: Path) -> list[str]:
    names = [normalize_name(line) for line in path.read_text(encoding="utf-8").splitlines()]
    names = [name for name in names if name]
    if len(names) != len(set(names)):
        duplicates = [name for name, count in Counter(names).items() if count > 1]
        raise ValueError(f"Duplicate clips in {path}: {duplicates[:5]}")
    return names


def parse_subsets(values: Sequence[str] | None, train_size: int) -> list[tuple[str, int]]:
    values = list(values or ("smoke=512", "pilot=2000", "main=6000"))
    parsed: list[tuple[str, int]] = []
    seen_names: set[str] = set()
    for value in values:
        if "=" not in value:
            raise ValueError(f"--subset must be NAME=SIZE, got {value!r}")
        label, raw_size = (piece.strip() for piece in value.split("=", 1))
        if not label or not re.fullmatch(r"[A-Za-z0-9_-]+", label):
            raise ValueError(f"Invalid subset label: {label!r}")
        size = int(raw_size)
        if not 0 < size <= train_size:
            raise ValueError(f"Subset {label!r} size must be in [1, {train_size}], got {size}")
        if label in seen_names:
            raise ValueError(f"Duplicate subset label: {label}")
        parsed.append((label, size))
        seen_names.add(label)
    sizes = [size for _, size in parsed]
    if len(sizes) != len(set(sizes)):
        raise ValueError("Subset sizes must be unique")
    return sorted(parsed, key=lambda item: item[1])


def split_hash(names: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for name in sorted(names):
        digest.update(name.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def extract_annotation(text: str) -> tuple[str, str, str, int]:
    expressions: list[str] = []
    actions: list[str] = []
    for kind, value in TAG_PATTERN.findall(text or ""):
        cleaned = " ".join(value.split()) or MISSING_TAG
        target = expressions if kind == "表情" else actions
        if cleaned not in target:
            target.append(cleaned)
    expression = " | ".join(expressions) if expressions else MISSING_TAG
    action = " | ".join(actions) if actions else MISSING_TAG
    if expressions and actions:
        pattern = "expression+action"
    elif expressions:
        pattern = "expression-only"
    elif actions:
        pattern = "action-only"
    else:
        pattern = "no-tags"
    transcript = TAG_PATTERN.sub("", text or "").strip()
    return expression, action, pattern, len(transcript)


def finite_nonnegative(value: float) -> float:
    value = float(value)
    return value if math.isfinite(value) and value >= 0.0 else 0.0


def motion_dynamics(
    path: Path, motion_fps: float
) -> tuple[float, float, float, float, float, float, float, bool]:
    payload = np.load(path, allow_pickle=True).item()
    if not isinstance(payload, Mapping):
        raise ValueError(f"Expected motion dictionary in {path}")
    required = ("body", "left", "right")
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"Motion file {path} is missing {missing}")
    body = np.asarray(payload["body"], dtype=np.float32)
    left = np.asarray(payload["left"], dtype=np.float32)
    right = np.asarray(payload["right"], dtype=np.float32)
    frame_count = min(len(body), len(left), len(right))
    if frame_count < 1:
        raise ValueError(f"Motion file has no frames: {path}")
    duration_seconds = frame_count / float(motion_fps)
    if frame_count == 1:
        return duration_seconds, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, "positions" in payload

    body_rotation = body[:frame_count, 3:]
    body_frame_speeds = np.linalg.norm(
        np.diff(body_rotation, axis=0).reshape(frame_count - 1, -1, 6), axis=-1
    )
    body_rotation_speed = finite_nonnegative(np.mean(body_frame_speeds))
    body_rotation_p90 = finite_nonnegative(np.percentile(body_frame_speeds, 90))
    hand_rotation = np.concatenate((left[:frame_count], right[:frame_count]), axis=-1)
    hand_frame_speeds = np.linalg.norm(
        np.diff(hand_rotation, axis=0).reshape(frame_count - 1, -1, 6), axis=-1
    )
    hand_rotation_speed = finite_nonnegative(np.mean(hand_frame_speeds))
    hand_rotation_p90 = finite_nonnegative(np.percentile(hand_frame_speeds, 90))
    positions_available = "positions" in payload
    if positions_available:
        positions = np.asarray(payload["positions"], dtype=np.float32)
        positions_available = (
            positions.ndim == 3
            and positions.shape[0] >= frame_count
            and positions.shape[1] >= 1
            and positions.shape[2] == 3
        )
    if positions_available:
        # Remove pelvis translation before measuring physical joint dynamics.
        # This is insensitive to the dataset's absolute/delta root eras.
        centered = positions[:frame_count] - positions[:frame_count, :1, :]
        joint_speeds = np.linalg.norm(np.diff(centered, axis=0), axis=-1)
        joint_speed_mean = finite_nonnegative(np.mean(joint_speeds))
        joint_speed_p90 = finite_nonnegative(np.percentile(joint_speeds, 90))
    else:
        # Released files may omit the documented positions array.
        # Keep them without fabricating positions; rotation dynamics are a
        # scale-compatible, codec-independent fallback for subset stratification.
        joint_speed_mean = body_rotation_speed
        joint_speed_p90 = finite_nonnegative(np.percentile(body_frame_speeds, 90))
    return (
        duration_seconds,
        joint_speed_mean,
        joint_speed_p90,
        body_rotation_speed,
        body_rotation_p90,
        hand_rotation_speed,
        hand_rotation_p90,
        bool(positions_available),
    )


def scan_metadata(
    names: Sequence[str],
    *,
    data_dir: Path,
    text_map: Mapping[str, str],
    motion_fps: float,
    progress_every: int,
) -> list[ClipMetadata]:
    rows: list[ClipMetadata] = []
    for index, name in enumerate(sorted(names)):
        pieces = name.split("/")
        if len(pieces) < 3:
            raise ValueError(f"Expected source/session/clip name, got {name!r}")
        if name not in text_map:
            raise KeyError(f"Missing text annotation for {name}")
        motion_path = data_dir / "motion_data" / f"{name}.npy"
        if not motion_path.is_file():
            raise FileNotFoundError(f"Missing motion file: {motion_path}")
        expression, action, pattern, text_characters = extract_annotation(text_map[name])
        (
            duration,
            joint_mean,
            joint_p90,
            body_speed,
            body_p90,
            hand_speed,
            hand_p90,
            positions_available,
        ) = motion_dynamics(motion_path, motion_fps)
        rows.append(
            ClipMetadata(
                name=name,
                source=pieces[0],
                session=f"{pieces[0]}/{pieces[1]}",
                duration_seconds=duration,
                text_characters=text_characters,
                annotation_pattern=pattern,
                expression=expression,
                action=action,
                joint_speed_mean=joint_mean,
                joint_speed_p90=joint_p90,
                body_rotation_speed=body_speed,
                body_rotation_p90=body_p90,
                hand_rotation_speed=hand_speed,
                hand_rotation_p90=hand_p90,
                joint_positions_available=positions_available,
            )
        )
        if progress_every > 0 and (index + 1) % progress_every == 0:
            print(f"Scanned raw motion: {index + 1}/{len(names)} clips", flush=True)
    return rows


def percentile_ranks(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if len(array) <= 1:
        return np.zeros(len(array), dtype=np.float64)
    order = np.argsort(array, kind="stable")
    ranks = np.empty(len(array), dtype=np.float64)
    ranks[order] = np.arange(len(array), dtype=np.float64) / (len(array) - 1)
    return ranks


def quantile_labels(values: Sequence[float], bins: int) -> list[str]:
    if bins < 2:
        raise ValueError("quantile_bins must be at least 2")
    array = np.asarray(values, dtype=np.float64)
    edges = np.quantile(array, np.linspace(0.0, 1.0, bins + 1))
    indices = np.searchsorted(edges[1:-1], array, side="right")
    return [f"q{int(index) + 1}of{bins}" for index in indices]


def enrich_metadata(rows: Sequence[ClipMetadata], bins: int) -> list[ClipMetadata]:
    dynamics = np.stack(
        [
            percentile_ranks([row.body_rotation_speed for row in rows]),
            percentile_ranks([row.body_rotation_p90 for row in rows]),
            percentile_ranks([row.hand_rotation_speed for row in rows]),
            percentile_ranks([row.hand_rotation_p90 for row in rows]),
        ],
        axis=1,
    )
    complexity = dynamics.mean(axis=1)
    duration_bins = quantile_labels([row.duration_seconds for row in rows], bins)
    text_bins = quantile_labels([row.text_characters for row in rows], bins)
    complexity_bins = quantile_labels(complexity.tolist(), bins)
    enriched: list[ClipMetadata] = []
    for index, row in enumerate(rows):
        payload = asdict(row)
        payload.update(
            complexity_score=float(complexity[index]),
            duration_bin=duration_bins[index],
            text_length_bin=text_bins[index],
            complexity_bin=complexity_bins[index],
        )
        enriched.append(ClipMetadata(**payload))
    return enriched


def load_cache(
    path: Path, expected_hash: str, *, quantile_bins: int, motion_fps: float
) -> list[ClipMetadata] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != METADATA_SCHEMA_VERSION:
        return None
    if payload.get("train_split_sha256") != expected_hash:
        return None
    if int(payload.get("quantile_bins", -1)) != int(quantile_bins):
        return None
    if not math.isclose(float(payload.get("motion_fps", -1.0)), float(motion_fps)):
        return None
    rows = payload.get("clips")
    if not isinstance(rows, list):
        return None
    return [ClipMetadata(**row) for row in rows]


def write_cache(
    path: Path,
    rows: Sequence[ClipMetadata],
    train_hash: str,
    *,
    quantile_bins: int,
    motion_fps: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": METADATA_SCHEMA_VERSION,
        "train_split_sha256": train_hash,
        "quantile_bins": int(quantile_bins),
        "motion_fps": float(motion_fps),
        "complexity_definition": (
            "mean percentile rank of body and hand 6D-rotation mean/p90 change; "
            "joint positions are recorded for audit but do not affect selection"
        ),
        "clips": [asdict(row) for row in rows],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def stable_jitter(name: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def balanced_priority(
    rows: Sequence[ClipMetadata],
    max_size: int,
    *,
    seed: int,
    balance_specs: Mapping[str, tuple[float, float]] = BALANCE_SPECS,
) -> list[int]:
    """Return a deterministic greedy ordering whose prefixes remain balanced."""
    if not 0 < max_size <= len(rows):
        raise ValueError(f"max_size must be in [1, {len(rows)}], got {max_size}")
    count = len(rows)
    group_state = []
    for attribute, (importance, temperature) in balance_specs.items():
        values = [str(getattr(row, attribute)) for row in rows]
        categories = sorted(set(values))
        category_to_index = {value: index for index, value in enumerate(categories)}
        indices = np.fromiter(
            (category_to_index[value] for value in values), dtype=np.int32, count=count
        )
        frequencies = np.bincount(indices, minlength=len(categories)).astype(np.float64)
        targets = np.power(frequencies, float(temperature))
        targets /= targets.sum()
        selected_counts = np.zeros(len(categories), dtype=np.float64)
        group_state.append((float(importance), indices, targets, selected_counts))

    jitter = np.asarray([stable_jitter(row.name, seed) for row in rows], dtype=np.float64)
    selected_mask = np.zeros(count, dtype=bool)
    ordering: list[int] = []
    for selection_index in range(max_size):
        scores = jitter * 1e-9
        for importance, indices, targets, selected_counts in group_state:
            desired_counts = (selection_index + 1) * targets
            relative_deficit = (desired_counts - selected_counts) / np.maximum(
                desired_counts, 1.0
            )
            scores = scores + importance * relative_deficit[indices]
        scores[selected_mask] = -np.inf
        chosen = int(np.argmax(scores))
        if selected_mask[chosen]:
            raise RuntimeError("Balanced sampler selected a duplicate index")
        selected_mask[chosen] = True
        ordering.append(chosen)
        for _, indices, _, selected_counts in group_state:
            selected_counts[indices[chosen]] += 1.0
    return ordering


def numeric_summary(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    if not len(array):
        return {"count": 0}
    return {
        "count": int(len(array)),
        "min": float(array.min()),
        "p10": float(np.percentile(array, 10)),
        "median": float(np.median(array)),
        "p90": float(np.percentile(array, 90)),
        "max": float(array.max()),
        "mean": float(array.mean()),
    }


def categorical_summary(rows: Sequence[ClipMetadata], attribute: str) -> dict[str, object]:
    counts = Counter(str(getattr(row, attribute)) for row in rows)
    return {
        "unique": len(counts),
        "counts": dict(sorted(counts.items(), key=lambda item: (-item[1], item[0]))),
    }


def sparse_tag_summary(
    selected: Sequence[ClipMetadata], all_rows: Sequence[ClipMetadata], attribute: str
) -> dict[str, object]:
    total_counts = Counter(str(getattr(row, attribute)) for row in all_rows)
    selected_counts = Counter(str(getattr(row, attribute)) for row in selected)
    annotated = sum(count for tag, count in selected_counts.items() if tag != MISSING_TAG)
    rare_clips = sum(
        count
        for tag, count in selected_counts.items()
        if tag != MISSING_TAG and total_counts[tag] <= 4
    )
    return {
        "annotated_clips": annotated,
        "unique_selected": len([tag for tag in selected_counts if tag != MISSING_TAG]),
        "unique_available": len([tag for tag in total_counts if tag != MISSING_TAG]),
        "clips_with_dataset_frequency_1_to_4": rare_clips,
        "top_selected": dict(selected_counts.most_common(20)),
    }


def subset_summary(rows: Sequence[ClipMetadata], all_rows: Sequence[ClipMetadata]) -> dict[str, object]:
    result: dict[str, object] = {
        "clips": len(rows),
        "duration_hours": sum(row.duration_seconds for row in rows) / 3600.0,
        "duration_seconds": numeric_summary(row.duration_seconds for row in rows),
        "complexity_score": numeric_summary(row.complexity_score for row in rows),
    }
    for attribute in (
        "source",
        "session",
        "duration_bin",
        "complexity_bin",
        "text_length_bin",
        "annotation_pattern",
    ):
        result[attribute] = categorical_summary(rows, attribute)
    result["expression"] = sparse_tag_summary(rows, all_rows, "expression")
    result["action"] = sparse_tag_summary(rows, all_rows, "action")
    return result


def validate_official_splits(
    train_names: Sequence[str], val_names: Sequence[str], test_names: Sequence[str]
) -> None:
    train, val, test = set(train_names), set(val_names), set(test_names)
    overlaps = {
        "train_val": sorted(train & val),
        "train_test": sorted(train & test),
        "val_test": sorted(val & test),
    }
    leaking = {key: values[:5] for key, values in overlaps.items() if values}
    if leaking:
        raise ValueError(f"Official dataset splits overlap: {leaking}")


def write_selected_metadata(path: Path, rows: Sequence[ClipMetadata], ranks: Sequence[int]) -> None:
    fields = ["selection_rank", *ClipMetadata.__dataclass_fields__.keys()]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for rank, row in zip(ranks, rows):
            writer.writerow({"selection_rank": rank, **asdict(row)})


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    train_path = (args.train_split or data_dir / "split" / "train_file_list.txt").resolve()
    val_path = (args.val_split or data_dir / "split" / "val_file_list.txt").resolve()
    test_path = (args.test_split or data_dir / "split" / "test_file_list.txt").resolve()
    text_path = (args.text_json or data_dir / "text_data" / "motion2text.json").resolve()
    output_dir = args.output_dir.expanduser().resolve()
    cache_path = (
        args.metadata_cache.expanduser().resolve()
        if args.metadata_cache is not None
        else data_dir / ".cache" / "step1_balanced_subset_metadata_v3.json"
    )

    train_names = read_names(train_path)
    val_names = read_names(val_path)
    test_names = read_names(test_path)
    validate_official_splits(train_names, val_names, test_names)
    subsets = parse_subsets(args.subset, len(train_names))
    train_digest = split_hash(train_names)
    rows = (
        None
        if args.refresh_metadata
        else load_cache(
            cache_path,
            train_digest,
            quantile_bins=int(args.quantile_bins),
            motion_fps=float(args.motion_fps),
        )
    )
    if rows is None:
        text_map_payload = json.loads(text_path.read_text(encoding="utf-8"))
        text_map = {normalize_name(str(key)): str(value) for key, value in text_map_payload.items()}
        rows = scan_metadata(
            train_names,
            data_dir=data_dir,
            text_map=text_map,
            motion_fps=float(args.motion_fps),
            progress_every=int(args.progress_every),
        )
        rows = enrich_metadata(rows, int(args.quantile_bins))
        write_cache(
            cache_path,
            rows,
            train_digest,
            quantile_bins=int(args.quantile_bins),
            motion_fps=float(args.motion_fps),
        )
        print(f"Wrote reusable metadata cache: {cache_path}")
    else:
        print(f"Loaded reusable metadata cache: {cache_path}")
    if {row.name for row in rows} != set(train_names):
        raise RuntimeError("Metadata cache does not exactly match the official training split")

    rows = sorted(rows, key=lambda row: row.name)
    max_size = max(size for _, size in subsets)
    ordering = balanced_priority(rows, max_size, seed=int(args.seed))
    selected_rows = [rows[index] for index in ordering]
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, object] = {
        "schema_version": 1,
        "seed": int(args.seed),
        "official_splits": {
            "train": len(train_names),
            "val": len(val_names),
            "test": len(test_names),
            "train_sha256": train_digest,
            "leakage": {"train_val": 0, "train_test": 0, "val_test": 0},
        },
        "uses_msd": False,
        "complexity_definition": (
            "Mean percentile rank of body and hand 6D-rotation mean/p90 change; "
            "joint positions are recorded for audit but do not affect selection."
        ),
        "raw_motion_quality": {
            "clips_with_positions": sum(row.joint_positions_available for row in rows),
            "clips_without_positions": sum(
                not row.joint_positions_available for row in rows
            ),
        },
        "balance_specs": {
            key: {"importance": value[0], "frequency_temperature": value[1]}
            for key, value in BALANCE_SPECS.items()
        },
        "full_train": subset_summary(rows, rows),
        "subsets": {},
    }
    previous_names: set[str] = set()
    for label, size in subsets:
        subset_rows = selected_rows[:size]
        names = [row.name for row in subset_rows]
        current_names = set(names)
        if len(current_names) != size:
            raise RuntimeError(f"Subset {label} contains duplicate clips")
        if not current_names <= set(train_names):
            raise RuntimeError(f"Subset {label} contains clips outside the training split")
        if previous_names and not previous_names <= current_names:
            raise RuntimeError(f"Subset {label} is not a superset of the preceding subset")
        if current_names & (set(val_names) | set(test_names)):
            raise RuntimeError(f"Subset {label} leaks validation/test clips")
        filename = f"train_step1_{label}_{size}.txt"
        (output_dir / filename).write_text("\n".join(names) + "\n", encoding="utf-8")
        report["subsets"][label] = {
            "size": size,
            "path": filename,
            "nested_superset_of_previous": previous_names <= current_names,
            **subset_summary(subset_rows, rows),
        }
        previous_names = current_names
        print(f"Wrote {label:>8}: {size:>5} clips -> {output_dir / filename}")

    write_selected_metadata(
        output_dir / "selected_main_metadata.csv",
        selected_rows,
        range(1, len(selected_rows) + 1),
    )
    report_path = output_dir / "subset_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote report: {report_path}")
    print("PASS: subsets are deterministic, nested, unique, train-only, and val/test-clean")


if __name__ == "__main__":
    main()
