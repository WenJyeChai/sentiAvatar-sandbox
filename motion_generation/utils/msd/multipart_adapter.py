"""SentiAvatar multipart codec and token-dataset adapter for MSD experiments."""

from __future__ import annotations

import gc
import json
import os
import pathlib
import sys
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
import torch


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[2]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

PART_ORDER = ("upper", "lower", "feet", "hands")


def torch_load_trusted(path: Path, map_location="cpu"):
    saved_posix_path = pathlib.PosixPath
    if os.name == "nt":
        pathlib.PosixPath = pathlib.WindowsPath
    try:
        try:
            return torch.load(path, map_location=map_location, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=map_location)
    finally:
        pathlib.PosixPath = saved_posix_path


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"CUDA is unavailable; using CPU instead of {requested}")
        return torch.device("cpu")
    return torch.device(requested)


def apply_speed_variant(tokens: torch.Tensor, speed: float) -> torch.Tensor:
    """Nearest-neighbor temporal resampling for token speed augmentation."""
    speed = float(speed)
    if speed <= 0:
        raise ValueError(f"speed must be positive, got {speed}")
    if abs(speed - 1.0) < 1e-6 or tokens.shape[0] <= 1:
        return tokens
    new_frames = max(2, int(round(tokens.shape[0] / speed)))
    indices = torch.linspace(
        0,
        tokens.shape[0] - 1,
        new_frames,
        device=tokens.device,
    ).round().long()
    return tokens[indices]


def cache_key(clip_id: str, speed: float) -> str:
    safe = clip_id.replace("/", "__").replace("\\", "__")
    return f"{safe}@speed{float(speed):.2f}"


@dataclass
class MultipartCodebookSet:
    """Lightweight multipart metadata and codebooks without encoder/decoder weights."""

    _codebooks: dict[str, torch.Tensor]
    checkpoint_paths: dict[str, Path]
    part_order: tuple[str, ...]
    codebook_size: int
    num_quantizers: int
    unit_length: int
    device: torch.device

    @classmethod
    def from_checkpoints(
        cls,
        checkpoints: Mapping[str, Path],
        device: torch.device,
        part_order: Sequence[str] = PART_ORDER,
    ) -> "MultipartCodebookSet":
        order = tuple(str(part) for part in part_order)
        books: dict[str, torch.Tensor] = {}
        paths = {part: Path(checkpoints[part]).resolve() for part in order}
        sizes: set[int] = set()
        quantizer_counts: set[int] = set()
        unit_lengths: set[int] = set()

        for part in order:
            path = paths[part]
            if not path.exists():
                raise FileNotFoundError(f"Part checkpoint not found: {path}")
            checkpoint = torch_load_trusted(path, map_location="cpu")
            args = checkpoint.get("args", {})
            model_config = checkpoint.get("model_config", {})
            state = checkpoint.get("model_state_dict")
            if not isinstance(state, dict):
                raise KeyError(f"model_state_dict not found in {path}")

            num_quantizers = int(
                args.get("num_quantizers", model_config.get("num_quantizers", 4))
            )
            part_books = []
            for quantizer in range(num_quantizers):
                key = f"quantizers.{part}.layers.{quantizer}.codebook"
                if key not in state:
                    candidates = [
                        name for name in state
                        if name.startswith(f"quantizers.{part}.") and name.endswith(".codebook")
                    ]
                    raise KeyError(
                        f"Codebook key '{key}' not found in {path}; candidates={candidates}"
                    )
                part_books.append(state[key].detach().float())
            stacked = torch.stack(part_books, dim=0)
            books[part] = stacked.to(device)
            sizes.add(int(stacked.shape[1]))
            quantizer_counts.add(num_quantizers)
            unit_lengths.add(int(args.get("down_t", 1)) * int(args.get("stride_t", 2)))
            del state, checkpoint, part_books, stacked
            gc.collect()

        if len(sizes) != 1 or len(quantizer_counts) != 1 or len(unit_lengths) != 1:
            raise ValueError(
                "Multipart checkpoint layouts differ: "
                f"sizes={sizes}, quantizers={quantizer_counts}, unit_lengths={unit_lengths}"
            )
        return cls(
            _codebooks=books,
            checkpoint_paths=paths,
            part_order=order,
            codebook_size=next(iter(sizes)),
            num_quantizers=next(iter(quantizer_counts)),
            unit_length=next(iter(unit_lengths)),
            device=device,
        )

    @property
    def tokens_per_frame(self) -> int:
        return len(self.part_order) * self.num_quantizers

    @property
    def codebooks(self) -> dict[str, torch.Tensor]:
        return self._codebooks

    def metadata(self) -> dict[str, object]:
        return {
            "codec_kind": "multipart",
            "part_order": list(self.part_order),
            "codebook_size": self.codebook_size,
            "num_quantizers": self.num_quantizers,
            "tokens_per_frame": self.tokens_per_frame,
            "unit_length": self.unit_length,
            "checkpoints": {part: str(self.checkpoint_paths[part]) for part in self.part_order},
        }


@dataclass
class MultipartCodecAdapter:
    codecs: dict[str, Any]
    part_order: tuple[str, ...]
    device: torch.device

    @classmethod
    def from_checkpoints(
        cls,
        checkpoints: Mapping[str, Path],
        device: torch.device,
        part_order: Sequence[str] = PART_ORDER,
    ) -> "MultipartCodecAdapter":
        # Keep checkpoint/model imports lazy so CLI help and cache-only readers
        # do not require the full motion-model environment.
        from scripts.export_multipart_motion_tokens import load_part_codec

        order = tuple(str(part) for part in part_order)
        missing = [part for part in order if part not in checkpoints]
        if missing:
            raise KeyError(f"Missing checkpoint paths for parts: {missing}")

        codecs = {
            part: load_part_codec(Path(checkpoints[part]), device)
            for part in order
        }
        for part, loaded in codecs.items():
            if loaded.part != part:
                raise ValueError(
                    f"Checkpoint for '{part}' loaded codec part '{loaded.part}': "
                    f"{loaded.checkpoint_path}"
                )

        codebook_sizes = {codec.codebook_size for codec in codecs.values()}
        quantizer_counts = {codec.num_quantizers for codec in codecs.values()}
        unit_lengths = {codec.unit_length for codec in codecs.values()}
        if len(codebook_sizes) != 1 or len(quantizer_counts) != 1 or len(unit_lengths) != 1:
            raise ValueError(
                "Multipart codecs must share codebook size, quantizer count, and unit length; "
                f"got sizes={codebook_sizes}, quantizers={quantizer_counts}, "
                f"unit_lengths={unit_lengths}"
            )
        return cls(codecs=codecs, part_order=order, device=device)

    @property
    def codebook_size(self) -> int:
        return int(next(iter(self.codecs.values())).codebook_size)

    @property
    def num_quantizers(self) -> int:
        return int(next(iter(self.codecs.values())).num_quantizers)

    @property
    def unit_length(self) -> int:
        return int(next(iter(self.codecs.values())).unit_length)

    @property
    def tokens_per_frame(self) -> int:
        return len(self.part_order) * self.num_quantizers

    @property
    def codebooks(self) -> dict[str, torch.Tensor]:
        return {
            part: self.codecs[part].model.quantizers[part].codebooks.detach().float()
            for part in self.part_order
        }

    @torch.no_grad()
    def decode_parts(self, tokens: torch.Tensor) -> dict[str, torch.Tensor]:
        """Decode flattened ``(T, P*Q)`` tokens to denormalized part features."""
        if tokens.ndim != 2 or tokens.shape[1] != self.tokens_per_frame:
            raise ValueError(
                f"Expected tokens shaped (T,{self.tokens_per_frame}), got {tuple(tokens.shape)}"
            )
        tokens = tokens.to(device=self.device, dtype=torch.long)
        decoded: dict[str, torch.Tensor] = {}
        for part_idx, part in enumerate(self.part_order):
            start = part_idx * self.num_quantizers
            indices = tokens[:, start : start + self.num_quantizers].unsqueeze(0)
            loaded = self.codecs[part]
            value = loaded.model.decode({part: indices})[part]
            value = loaded.normalizer.denormalize_tensor(part, value)
            decoded[part] = value.squeeze(0).float()
        return decoded

    @torch.no_grad()
    def decode_combined_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """Decode to the 381D non-duplicated multipart feature layout."""
        parts = self.decode_parts(tokens)
        frames = min(int(parts[part].shape[0]) for part in self.part_order)
        return torch.cat([parts[part][:frames] for part in self.part_order], dim=-1)

    @torch.no_grad()
    def decode_legacy_motion(self, tokens: torch.Tensor) -> dict[str, np.ndarray]:
        """Decode to legacy body/left/right arrays for downstream visualization."""
        from utils.multipart_motion import merge_parts_to_legacy_motion

        parts = {
            part: value.detach().cpu().numpy()
            for part, value in self.decode_parts(tokens).items()
        }
        return merge_parts_to_legacy_motion(parts)

    def metadata(self) -> dict[str, object]:
        return {
            "codec_kind": "multipart",
            "part_order": list(self.part_order),
            "codebook_size": self.codebook_size,
            "num_quantizers": self.num_quantizers,
            "tokens_per_frame": self.tokens_per_frame,
            "unit_length": self.unit_length,
            "checkpoints": {
                part: str(self.codecs[part].checkpoint_path)
                for part in self.part_order
            },
        }


class MultipartTokenDataset:
    """Read split names and exported multipart token JSON files."""

    def __init__(self, data_dir: Path, token_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir).resolve()
        self.token_dir = Path(
            token_dir or self.data_dir / "motion_token_data_multipart_512x4"
        ).resolve()
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.data_dir}")
        if not self.token_dir.exists():
            raise FileNotFoundError(
                f"Multipart token directory not found: {self.token_dir}. "
                "Run export_multipart_motion_tokens.py first."
            )
        self.manifest = self._load_manifest()

    def _load_manifest(self) -> dict[str, object]:
        path = self.token_dir / "export_manifest.json"
        if not path.exists():
            return {}
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object in token manifest: {path}")
        return payload

    def validate_layout(self, codec: MultipartCodecAdapter) -> None:
        expected = {
            "part_order": list(codec.part_order),
            "codebook_size": codec.codebook_size,
            "num_quantizers": codec.num_quantizers,
            "tokens_per_frame": codec.tokens_per_frame,
        }
        mismatches = {
            key: (self.manifest.get(key), value)
            for key, value in expected.items()
            if key in self.manifest and self.manifest.get(key) != value
        }
        if mismatches:
            raise ValueError(f"Token manifest does not match multipart codecs: {mismatches}")

    def split_names(self, split: str) -> list[str]:
        path = self.data_dir / "split" / f"{split}_file_list.txt"
        if not path.exists():
            raise FileNotFoundError(f"Split file not found: {path}")
        with open(path, "r", encoding="utf-8") as handle:
            return [line.strip().replace("\\", "/") for line in handle if line.strip()]

    def token_path(self, name: str) -> Path:
        parts = PurePosixPath(name.replace("\\", "/")).parts
        return self.token_dir / Path(*parts).with_suffix(".json")

    def load_tokens(self, name: str, expected_slots: int) -> torch.Tensor:
        path = self.token_path(name)
        if not path.exists():
            raise FileNotFoundError(path)
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        raw = payload.get("tokens") if isinstance(payload, dict) else payload
        tokens = torch.as_tensor(raw, dtype=torch.long)
        if tokens.ndim != 2 or tokens.shape[1] != expected_slots:
            raise ValueError(
                f"{name}: expected token shape (T,{expected_slots}), got {tuple(tokens.shape)}"
            )
        return tokens

    def iter_tokens(
        self,
        split: str,
        *,
        expected_slots: int,
        device: torch.device,
        speed: float = 1.0,
        limit: int | None = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        emitted = 0
        for name in self.split_names(split):
            try:
                tokens = self.load_tokens(name, expected_slots)
            except FileNotFoundError:
                continue
            if tokens.shape[0] < 2:
                continue
            tokens = apply_speed_variant(tokens.to(device), speed)
            yield name, tokens
            emitted += 1
            if limit is not None and emitted >= limit:
                break


class MultipartMSDCache:
    """Read combined and per-part MSD arrays produced by the cache command."""

    def __init__(self, cache_dir: Path) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        manifest_path = self.cache_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"MSD cache manifest not found: {manifest_path}")
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        if isinstance(manifest, list):
            self.metadata: dict[str, object] = {}
            entries = manifest
        elif isinstance(manifest, dict):
            self.metadata = dict(manifest.get("metadata") or {})
            entries = manifest.get("entries") or []
        else:
            raise ValueError(f"Unsupported MSD cache manifest format: {manifest_path}")
        self.entries = list(entries)
        self.by_key = {str(row["key"]): row for row in self.entries}

    def get(self, clip_id: str, speed: float = 1.0) -> dict[str, np.ndarray]:
        key = cache_key(clip_id, speed)
        if key not in self.by_key:
            raise KeyError(f"MSD cache miss: {key}")
        path = self.cache_dir / str(self.by_key[key]["path"])
        with np.load(path) as payload:
            return {name: payload[name] for name in payload.files}
