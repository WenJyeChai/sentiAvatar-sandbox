"""Canonical token and fixed-schedule contracts for the multipart Step 1 planner.

Phase 1 reuses the existing ``[body_N]`` tokens in the Qwen checkpoint.  The
logical body id is slot-offset encoded as ``slot * 512 + local_id``.  This keeps
the already-trained embedding rows while giving every multipart RVQ slot a
disjoint 512-token classifier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


BODY_PART_ORDER = ("upper", "lower", "feet", "hands")
QUANTIZERS_PER_PART = 4
BODY_CODEBOOK_SIZE = 512
BODY_SLOT_COUNT = len(BODY_PART_ORDER) * QUANTIZERS_PER_PART
BODY_TOKEN_COUNT = BODY_SLOT_COUNT * BODY_CODEBOOK_SIZE
MAX_GAP = 15

STEP1_ROLE_TOKEN = "[step1_planner]"
MOTION_START_TOKEN = "[motion_start]"
MOTION_END_TOKEN = "[motion_end]"
ANCHOR_TOKEN = "[anchor]"
MIMI_FRAME_TOKEN = "[mimi_frame]"
AUDIO_END_TOKEN = "[audio_end]"
SEED_NEUTRAL_TOKEN = "[seed_neutral]"
SEED_OBSERVED_TOKEN = "[seed_observed]"
SEED_PREVIOUS_TOKEN = "[seed_previous]"
GAP_TOKENS = tuple(f"[gap_{gap}]" for gap in range(MAX_GAP + 1))

STEP1_CONTROL_TOKENS = (
    STEP1_ROLE_TOKEN,
    MOTION_START_TOKEN,
    MOTION_END_TOKEN,
    ANCHOR_TOKEN,
    MIMI_FRAME_TOKEN,
    AUDIO_END_TOKEN,
    SEED_NEUTRAL_TOKEN,
    SEED_OBSERVED_TOKEN,
    SEED_PREVIOUS_TOKEN,
    *GAP_TOKENS,
)

SEED_TOKEN_BY_MODE = {
    "neutral": SEED_NEUTRAL_TOKEN,
    "observed": SEED_OBSERVED_TOKEN,
    "previous": SEED_PREVIOUS_TOKEN,
}


@dataclass(frozen=True)
class BodySlot:
    slot: int
    part: str
    quantizer: int

    @property
    def global_start(self) -> int:
        return self.slot * BODY_CODEBOOK_SIZE


BODY_SLOTS = tuple(
    BodySlot(
        slot=part_index * QUANTIZERS_PER_PART + quantizer,
        part=part,
        quantizer=quantizer,
    )
    for part_index, part in enumerate(BODY_PART_ORDER)
    for quantizer in range(QUANTIZERS_PER_PART)
)


@dataclass(frozen=True)
class SparseAnchorPlan:
    """Machine-readable sparse anchor plan with local 0..511 ids."""

    token_length: int
    times: tuple[int, ...]
    anchors: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        validate_sparse_plan(self.token_length, self.times, self.anchors)

    @property
    def gaps(self) -> tuple[int, ...]:
        return tuple(gap_from_anchor_times(left, right) for left, right in zip(self.times, self.times[1:]))

    def as_dict(self) -> dict[str, Any]:
        return {
            "token_length": self.token_length,
            "layout": "body_16slot_512x4",
            "anchors": [
                {"time": time, "tokens": list(tokens)}
                for time, tokens in zip(self.times, self.anchors)
            ],
            "gaps": list(self.gaps),
        }


def body_global_id(slot: int, local_id: int) -> int:
    if not 0 <= int(slot) < BODY_SLOT_COUNT:
        raise ValueError(f"slot must be in [0, {BODY_SLOT_COUNT - 1}], got {slot}")
    if not 0 <= int(local_id) < BODY_CODEBOOK_SIZE:
        raise ValueError(f"local_id must be in [0, {BODY_CODEBOOK_SIZE - 1}], got {local_id}")
    return int(slot) * BODY_CODEBOOK_SIZE + int(local_id)


def split_body_global_id(global_id: int) -> tuple[int, int]:
    if not 0 <= int(global_id) < BODY_TOKEN_COUNT:
        raise ValueError(f"global_id must be in [0, {BODY_TOKEN_COUNT - 1}], got {global_id}")
    return divmod(int(global_id), BODY_CODEBOOK_SIZE)


def body_token(slot: int, local_id: int) -> str:
    return f"[body_{body_global_id(slot, local_id)}]"


def parse_body_token(token: str) -> tuple[int, int]:
    if not token.startswith("[body_") or not token.endswith("]"):
        raise ValueError(f"Not a body token: {token!r}")
    try:
        global_id = int(token[len("[body_") : -1])
    except ValueError as exc:
        raise ValueError(f"Malformed body token: {token!r}") from exc
    return split_body_global_id(global_id)


def format_anchor(local_ids: Sequence[int]) -> tuple[str, ...]:
    validate_anchor(local_ids)
    return tuple(body_token(slot, local_id) for slot, local_id in enumerate(local_ids))


def validate_anchor(local_ids: Sequence[int]) -> None:
    if len(local_ids) != BODY_SLOT_COUNT:
        raise ValueError(f"Expected {BODY_SLOT_COUNT} body ids, got {len(local_ids)}")
    for slot, local_id in enumerate(local_ids):
        if not 0 <= int(local_id) < BODY_CODEBOOK_SIZE:
            raise ValueError(
                f"Anchor slot {slot} ({BODY_SLOTS[slot].part} q{BODY_SLOTS[slot].quantizer}) "
                f"is outside [0, {BODY_CODEBOOK_SIZE - 1}]: {local_id}"
            )


def gap_from_anchor_times(left: int, right: int) -> int:
    if int(right) <= int(left):
        raise ValueError(f"right anchor must follow left anchor, got {left}, {right}")
    gap = int(right) - int(left) - 1
    if gap > MAX_GAP:
        raise ValueError(f"gap {gap} exceeds Step 2 maximum {MAX_GAP}")
    return gap


def anchor_distance_from_gap(gap: int) -> int:
    if not 0 <= int(gap) <= MAX_GAP:
        raise ValueError(f"gap must be in [0, {MAX_GAP}], got {gap}")
    return int(gap) + 1


def fixed_anchor_times(token_length: int, gap: int = 3) -> tuple[int, ...]:
    """Return t=0, fixed-distance anchors, and an exact final anchor.

    The last interval is allowed to be shorter than the requested fixed gap.
    This is necessary because utterance duration is unknown online and body
    token lengths are not generally ``1 mod (gap + 1)``.
    """

    token_length = int(token_length)
    if token_length < 1:
        raise ValueError(f"token_length must be positive, got {token_length}")
    distance = anchor_distance_from_gap(gap)
    final_time = token_length - 1
    times = list(range(0, token_length, distance))
    if times[-1] != final_time:
        times.append(final_time)
    return tuple(times)


def causal_audio_boundaries(
    anchor_times: Sequence[int],
    *,
    audio_frames: int,
    audio_fps: float = 12.5,
    motion_fps: float = 10.0,
) -> tuple[int, ...]:
    """Map target anchors to nondecreasing causal audio-prefix boundaries.

    Boundary zero belongs to the seed at time zero.  Every non-final boundary
    is the ceiling of its physical timestamp on the audio grid.  The final
    anchor consumes all remaining audio, including rounding/tail frames.
    """

    if not anchor_times or int(anchor_times[0]) != 0:
        raise ValueError("anchor_times must start at zero")
    if audio_frames < 0 or audio_fps <= 0 or motion_fps <= 0:
        raise ValueError("audio_frames must be nonnegative and frame rates positive")
    boundaries = [0]
    for time in anchor_times[1:-1]:
        boundary = int(-(-int(time) * float(audio_fps) // float(motion_fps)))
        boundaries.append(min(audio_frames, max(boundaries[-1], boundary)))
    if len(anchor_times) > 1:
        boundaries.append(int(audio_frames))
    return tuple(boundaries)


def validate_sparse_plan(
    token_length: int,
    times: Sequence[int],
    anchors: Sequence[Sequence[int]],
) -> None:
    if int(token_length) < 1:
        raise ValueError("token_length must be positive")
    if len(times) != len(anchors) or not times:
        raise ValueError("times and anchors must have the same nonzero length")
    if int(times[0]) != 0 or int(times[-1]) != int(token_length) - 1:
        raise ValueError("plan must start at 0 and end at token_length - 1")
    previous = None
    for index, (time, anchor) in enumerate(zip(times, anchors)):
        time = int(time)
        if previous is not None:
            gap_from_anchor_times(previous, time)
        validate_anchor(anchor)
        previous = time
        if index and time >= int(token_length):
            raise ValueError(f"anchor time {time} is outside token length {token_length}")


def plan_from_dense_tokens(
    dense_tokens: Sequence[Sequence[int]],
    *,
    gap: int = 3,
) -> SparseAnchorPlan:
    times = fixed_anchor_times(len(dense_tokens), gap=gap)
    anchors = tuple(tuple(int(value) for value in dense_tokens[time]) for time in times)
    return SparseAnchorPlan(token_length=len(dense_tokens), times=times, anchors=anchors)


def ensure_step1_special_tokens(tokenizer: Any, model: Any | None = None) -> list[str]:
    """Add only missing Step 1 controls while preserving legacy special tokens."""

    missing = [token for token in STEP1_CONTROL_TOKENS if tokenizer.convert_tokens_to_ids(token) is None]
    if missing:
        # Transformers 4.57 renamed ``additional_special_tokens`` to
        # ``extra_special_tokens``. Support both APIs without replacing the
        # checkpoint's existing special-token roles.
        try:
            tokenizer.add_special_tokens(
                {"extra_special_tokens": missing},
                replace_extra_special_tokens=False,
            )
        except (TypeError, KeyError, AssertionError):
            tokenizer.add_special_tokens(
                {"additional_special_tokens": missing},
                replace_additional_special_tokens=False,
            )
        if model is not None:
            model.resize_token_embeddings(len(tokenizer))
    validate_body_tokenizer_contract(tokenizer)
    return missing


def validate_body_tokenizer_contract(tokenizer: Any) -> None:
    ids = []
    for global_id in range(BODY_TOKEN_COUNT):
        token = f"[body_{global_id}]"
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            raise ValueError(f"Qwen tokenizer is missing required token {token}")
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [token_id]:
            raise ValueError(f"{token} is not represented by exactly one tokenizer id: {encoded}")
        ids.append(int(token_id))
    if len(set(ids)) != BODY_TOKEN_COUNT:
        raise ValueError("The 8,192 multipart body tokens do not map to unique tokenizer ids")


def motion_token_id_table(tokenizer: Any) -> list[list[int]]:
    validate_body_tokenizer_contract(tokenizer)
    return [
        [int(tokenizer.convert_tokens_to_ids(body_token(slot, local_id))) for local_id in range(BODY_CODEBOOK_SIZE)]
        for slot in range(BODY_SLOT_COUNT)
    ]


def validate_motion_payload(payload: Mapping[str, Any], *, require_causal: bool = True) -> list[list[int]]:
    tokens = payload.get("tokens")
    if not isinstance(tokens, list) or not tokens:
        raise ValueError("Multipart motion payload has no nonempty 'tokens' list")
    if int(payload.get("codebook_size", -1)) != BODY_CODEBOOK_SIZE:
        raise ValueError(f"Expected codebook_size={BODY_CODEBOOK_SIZE}")
    if int(payload.get("num_quantizers", -1)) != QUANTIZERS_PER_PART:
        raise ValueError(f"Expected num_quantizers={QUANTIZERS_PER_PART}")
    if tuple(payload.get("part_order", ())) != BODY_PART_ORDER:
        raise ValueError(f"Expected part_order={BODY_PART_ORDER}, got {payload.get('part_order')}")
    if int(payload.get("tokens_per_frame", -1)) != BODY_SLOT_COUNT:
        raise ValueError(f"Expected tokens_per_frame={BODY_SLOT_COUNT}")
    if float(payload.get("motion_token_fps", -1)) != 10.0:
        raise ValueError(f"Expected motion_token_fps=10.0, got {payload.get('motion_token_fps')}")
    if require_causal and payload.get("body_causal") is not True:
        raise ValueError("Phase 1 requires motion tokens exported by the causal body codecs")
    normalized = []
    for frame in tokens:
        validate_anchor(frame)
        normalized.append([int(value) for value in frame])
    return normalized


def normalize_anchor(anchor: Iterable[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in anchor)
    validate_anchor(values)
    return values
