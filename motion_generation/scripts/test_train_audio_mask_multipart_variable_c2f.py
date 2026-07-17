from __future__ import annotations

import sys
from pathlib import Path
from types import MethodType

import torch


MOTION_GENERATION_DIR = Path(__file__).resolve().parents[1]
if str(MOTION_GENERATION_DIR) not in sys.path:
    sys.path.insert(0, str(MOTION_GENERATION_DIR))

from models.audio_motion_model import AudioMotionConfig, AudioMotionTransformer
from scripts.train_audio_mask_multipart_variable_c2f import (
    EpochResamplingLengthGroupedSampler,
    ResidualTargetBuilder,
    VariableGapC2FCollator,
    VariableGapC2FTrainer,
    VariableGapMaskDataset,
    VariableGapMaskExample,
    parse_args,
)
from transformers import TrainingArguments


def tiny_config(*, max_positions: int = 32) -> AudioMotionConfig:
    config = AudioMotionConfig(
        hidden_size=16,
        num_layers=1,
        num_heads=2,
        intermediate_size=32,
        max_position_embeddings=max_positions,
        vocab_size=17,
        codebook_size=4,
        audio_feat_dim=3,
        num_tokens_per_frame=4,
        num_frames=5,
        dropout=0.0,
        constrain_token_logits=True,
    )
    config.mask_token_id = 16
    config.num_quantizers_per_part = 2
    config.part_order = ["upper", "lower"]
    return config


def raw_frame(value: int) -> list[int]:
    return [value, value, value, value]


def synthetic_sequence(name: str, frames: int) -> dict:
    return {
        "name": name,
        "motion_tokens": [raw_frame(frame % 4) for frame in range(frames)],
        "audio_features": torch.randn(frames, 3).numpy(),
        "audio_fps": 10.0,
        "motion_token_fps": 10.0,
    }


def tiny_codebooks() -> dict[str, torch.Tensor]:
    return {
        part: torch.randn(2, 4, 8)
        for part in ("upper", "lower")
    }


def test_nested_yaml_config_loads_types_and_cli_overrides(tmp_path):
    config_path = tmp_path / "train.yaml"
    config_path.write_text(
        """
experiment:
  output_dir: checkpoints/test-run
masking:
  gap_bucket_weights: [0.2, 0.3, 0.5]
  stage_weights: [0.4, 0.3, 0.2, 0.1]
optimization:
  learning_rate: 1.0e-4
  bf16: true
  gradient_checkpointing: false
  ddp_find_unused_parameters: true
monitoring:
  wandb_tags: [yaml, c2f]
""".strip(),
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--config",
            str(config_path),
            "--learning_rate",
            "0.0002",
            "--no-bf16",
        ]
    )

    assert args.config == config_path
    assert args.output_dir == "checkpoints/test-run"
    assert args.gap_bucket_weights == "0.2,0.3,0.5"
    assert args.stage_weights == "0.4,0.3,0.2,0.1"
    assert args.wandb_tags == "yaml,c2f"
    assert args.learning_rate == 0.0002
    assert args.bf16 is False
    assert args.gradient_checkpointing is False
    assert args.ddp_find_unused_parameters is True


def test_training_windows_resample_reproducibly_and_remain_clip_balanced():
    sequences = [
        synthetic_sequence("long", 20),
        synthetic_sequence("medium", 10),
        synthetic_sequence("too_short", 2),
    ]
    dataset = VariableGapMaskDataset(
        sequences,
        min_gap_frames=1,
        max_gap_frames=7,
        windows_per_sequence=4,
        seed=17,
    )
    epoch0 = list(dataset.windows)
    dataset.resample(0)
    assert dataset.windows == epoch0
    assert len(dataset) == 8
    for sequence_idx in (0, 1):
        clip_windows = [
            (left, gap)
            for seq, left, gap in dataset.windows
            if seq == sequence_idx
        ]
        assert len(clip_windows) == 4
        assert len(set(clip_windows)) == 4
    assert all(sequence_idx != 2 for sequence_idx, _, _ in dataset.windows)

    dataset.resample(1)
    epoch1 = list(dataset.windows)
    assert epoch1 != epoch0
    assert len(epoch1) == len(epoch0)

    replica = VariableGapMaskDataset(
        sequences,
        min_gap_frames=1,
        max_gap_frames=7,
        windows_per_sequence=4,
        seed=17,
    )
    replica.resample(1)
    assert replica.windows == epoch1


def test_epoch_sampler_resamples_then_returns_every_dataset_index():
    dataset = VariableGapMaskDataset(
        [synthetic_sequence("a", 20), synthetic_sequence("b", 16)],
        min_gap_frames=1,
        max_gap_frames=7,
        windows_per_sequence=4,
        seed=23,
    )
    sampler = EpochResamplingLengthGroupedSampler(
        dataset,
        batch_size=4,
        tokens_per_frame=4,
        seed=23,
    )
    epoch0_windows = list(dataset.windows)
    epoch0_indices = list(iter(sampler))
    assert sorted(epoch0_indices) == list(range(len(dataset)))

    sampler.set_epoch(1)
    epoch1_indices = list(iter(sampler))
    assert dataset.windows != epoch0_windows
    assert sorted(epoch1_indices) == list(range(len(dataset)))


def test_variable_collator_pads_whole_frames_and_tracks_valid_masks():
    collator = VariableGapC2FCollator(tiny_config())
    short = VariableGapMaskExample(
        name="short",
        left_idx=0,
        right_idx=2,
        gap_frames=1,
        motion_tokens=[raw_frame(0), raw_frame(1), raw_frame(2)],
        audio_features=torch.randn(3, 3),
    )
    long = VariableGapMaskExample(
        name="long",
        left_idx=1,
        right_idx=5,
        gap_frames=3,
        motion_tokens=[raw_frame(i % 4) for i in range(5)],
        audio_features=torch.randn(5, 3),
    )

    batch = collator([short, long])

    assert batch["input_ids"].shape == (2, 20)
    assert batch["audio_features"].shape == (2, 5, 3)
    assert batch["attention_mask"].sum(dim=1).tolist() == [12, 20]
    assert batch["middle_mask"].sum(dim=1).tolist() == [4, 12]
    assert batch["input_ids"][0, 4:8].eq(16).all()
    assert batch["input_ids"][0, 12:].eq(16).all()
    assert not batch["attention_mask"][0, 12:].any()


def test_padding_mask_makes_valid_logits_invariant_to_padded_tail():
    torch.manual_seed(3)
    model = AudioMotionTransformer(tiny_config()).eval()
    short_ids = torch.tensor([[0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14]])
    short_audio = torch.randn(1, 3, 3)
    short_mask = torch.ones_like(short_ids, dtype=torch.bool)
    padded_ids = torch.cat([short_ids, torch.full((1, 8), 16, dtype=torch.long)], dim=1)
    padded_audio = torch.cat([short_audio, torch.randn(1, 2, 3)], dim=1)
    padded_mask = torch.cat(
        [short_mask, torch.zeros((1, 8), dtype=torch.bool)], dim=1
    )

    with torch.no_grad():
        short_logits = model(short_ids, audio_features=short_audio, attention_mask=short_mask)
        padded_logits = model(
            padded_ids, audio_features=padded_audio, attention_mask=padded_mask
        )

    assert torch.allclose(short_logits, padded_logits[:, :12], atol=1e-5, rtol=1e-5)


def test_padded_attention_supports_gradient_checkpointing_backward():
    torch.manual_seed(5)
    model = AudioMotionTransformer(tiny_config()).train()
    model.gradient_checkpointing_enable()
    input_ids = torch.tensor(
        [[0, 4, 8, 12, 16, 16, 16, 16, 2, 6, 10, 14, 16, 16, 16, 16]]
    )
    audio = torch.randn(1, 4, 3)
    attention = torch.tensor([[1] * 12 + [0] * 4], dtype=torch.bool)

    logits = model(input_ids, audio_features=audio, attention_mask=attention)
    loss = logits[:, :12].amax(dim=-1).mean()
    loss.backward()

    assert model.embed_tokens.weight.grad is not None
    assert torch.isfinite(model.embed_tokens.weight.grad).all()


def test_coarse_to_fine_generation_fills_q0_before_q1_and_skips_padding():
    config = tiny_config()
    model = AudioMotionTransformer(config).eval()

    def fake_forward(self, input_ids, labels=None, audio_features=None, attention_mask=None):
        del labels, audio_features, attention_mask
        logits = torch.full(
            (*input_ids.shape, self.config.vocab_size),
            -1000.0,
            device=input_ids.device,
        )
        ntpf = self.config.num_tokens_per_frame
        for position in range(input_ids.shape[1]):
            slot = position % ntpf
            q = slot % self.config.num_quantizers_per_part
            raw = 1
            if q == 1:
                previous_q0 = input_ids[:, position - 1]
                raw = torch.where(previous_q0.eq(16), 0, 2)
                logits[torch.arange(input_ids.shape[0]), position, slot * 4 + raw] = 0.0
            else:
                logits[:, position, slot * 4 + raw] = 0.0
        return logits

    model.forward = MethodType(fake_forward, model)
    anchors = torch.tensor([0, 4, 8, 12])
    input_ids = torch.cat(
        [anchors, torch.full((4,), 16), anchors, torch.full((4,), 16)]
    ).unsqueeze(0)
    attention = torch.tensor([[1] * 12 + [0] * 4], dtype=torch.bool)
    middle = torch.tensor([[0] * 4 + [1] * 4 + [0] * 8], dtype=torch.bool)
    audio = torch.randn(1, 4, 3)

    output = model.generate_quantizer_coarse_to_fine(
        input_ids,
        audio,
        middle_mask=middle,
        attention_mask=attention,
    )

    assert output[0, 4:8].tolist() == [1, 6, 9, 14]
    assert output[0, 12:].eq(16).all()
    assert torch.equal(output[0, :4], anchors)
    assert torch.equal(output[0, 8:12], anchors)


def test_adaptive_target_requantizes_residual_after_generated_q0():
    codebooks = {
        part: torch.tensor(
            [
                [[0.0], [2.0]],
                [[0.0], [-1.0]],
                [[0.0], [-1.0]],
            ]
        )
        for part in ("upper", "lower")
    }
    builder = ResidualTargetBuilder(
        codebooks,
        part_order=("upper", "lower"),
        codebook_size=2,
        num_quantizers=3,
    )
    offsets = torch.arange(6) * 2
    gt_ids = offsets.unsqueeze(0)
    current = gt_ids.clone()
    current[0, 0] = 1
    current[0, 3] = 7
    middle = torch.ones_like(gt_ids, dtype=torch.bool)

    q1_targets = builder.build_targets(
        gt_ids, current, middle, stage=1, adaptive=True
    )

    assert int(q1_targets[0, 1]) == 3
    assert int(q1_targets[0, 4]) == 9
    assert q1_targets.ne(-100).sum().item() == 2


def test_trainer_self_forced_stage_builds_a_finite_backward_loss(tmp_path):
    torch.manual_seed(11)
    config = tiny_config()
    model = AudioMotionTransformer(config).train()
    collator = VariableGapC2FCollator(config)
    examples = [
        VariableGapMaskExample(
            name="gap1",
            left_idx=0,
            right_idx=2,
            gap_frames=1,
            motion_tokens=[raw_frame(0), raw_frame(1), raw_frame(2)],
            audio_features=torch.randn(3, 3),
        ),
        VariableGapMaskExample(
            name="gap2",
            left_idx=0,
            right_idx=3,
            gap_frames=2,
            motion_tokens=[raw_frame(0), raw_frame(1), raw_frame(2), raw_frame(3)],
            audio_features=torch.randn(4, 3),
        ),
    ]
    batch = collator(examples)
    codebooks = {
        part: torch.randn(2, 4, 8)
        for part in ("upper", "lower")
    }
    builder = ResidualTargetBuilder(
        codebooks,
        part_order=("upper", "lower"),
        codebook_size=4,
        num_quantizers=2,
    )
    args = TrainingArguments(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        report_to="none",
        remove_unused_columns=False,
    )
    trainer = VariableGapC2FTrainer(
        model=model,
        args=args,
        data_collator=collator,
        target_builder=builder,
        stage_weights=(0.0, 1.0),
        self_forcing_warmup_ratio=0.0,
        self_forcing_ramp_ratio=0.0,
        self_forcing_max_prob=1.0,
        embedding_loss_weight=0.1,
        final_latent_loss_weight=0.1,
    )
    trainer.state.global_step = 1
    trainer.state.max_steps = 1

    assert not trainer._use_adaptive_targets(stage=0, self_forced=True)
    assert not trainer._use_adaptive_targets(stage=1, self_forced=False)
    assert trainer._use_adaptive_targets(stage=1, self_forced=True)
    trainer.adaptive_target_mode = "always"
    assert trainer._use_adaptive_targets(stage=1, self_forced=False)
    trainer.adaptive_target_mode = "never"
    assert not trainer._use_adaptive_targets(stage=1, self_forced=True)
    trainer.adaptive_target_mode = "self_forced"

    forward_training_modes = []
    original_forward = model.forward

    def tracked_forward(self, *args, **kwargs):
        forward_training_modes.append(self.training)
        return original_forward(*args, **kwargs)

    model.forward = MethodType(tracked_forward, model)
    loss = trainer.compute_loss(model, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert forward_training_modes == [False, True]
    assert model.training
    assert model.embed_tokens.weight.grad is not None
    assert torch.isfinite(model.embed_tokens.weight.grad).all()


def test_soft_recovery_uses_incorrect_prefix_pool_without_replacing_targets(tmp_path):
    torch.manual_seed(19)
    config = tiny_config()
    model = AudioMotionTransformer(config).train()
    collator = VariableGapC2FCollator(config)
    example = VariableGapMaskExample(
        name="gap2",
        left_idx=0,
        right_idx=3,
        gap_frames=2,
        motion_tokens=[raw_frame(0), raw_frame(1), raw_frame(2), raw_frame(3)],
        audio_features=torch.randn(4, 3),
    )
    batch = collator([example])
    builder = ResidualTargetBuilder(
        {part: torch.randn(2, 4, 8) for part in ("upper", "lower")},
        part_order=("upper", "lower"),
        codebook_size=4,
        num_quantizers=2,
    )
    trainer = VariableGapC2FTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            report_to="none",
            remove_unused_columns=False,
        ),
        target_builder=builder,
        stage_weights=(0.0, 1.0),
        self_forcing_warmup_ratio=0.0,
        self_forcing_ramp_ratio=0.0,
        self_forcing_max_prob=1.0,
        adaptive_target_mode="never",
        embedding_loss_weight=0.1,
        final_latent_loss_weight=0.1,
        soft_recovery_weight=0.1,
        soft_recovery_topk=2,
    )

    current = batch["input_ids"].clone()
    q0_mask = trainer._stage_mask(batch["middle_mask"], 0)
    current[q0_mask] = batch["gt_ids"][q0_mask]
    slots = torch.arange(current.shape[1]).remainder(config.num_tokens_per_frame)
    q0_positions = q0_mask & slots.view(1, -1).remainder(2).eq(0)
    offsets = slots.view(1, -1) * config.codebook_size
    raw = (current - offsets).remainder(config.codebook_size)
    current[q0_positions] = (
        offsets.expand_as(current)[q0_positions]
        + (raw[q0_positions] + 1).remainder(config.codebook_size)
    )
    logits = model(
        input_ids=current,
        audio_features=batch["audio_features"],
        attention_mask=batch["attention_mask"],
    )
    loss = trainer._soft_recovery_loss(
        logits,
        batch["gt_ids"],
        current,
        batch["middle_mask"],
        stage=1,
        metric_prefix="train_c2f",
    )
    canonical = builder.build_targets(
        batch["gt_ids"], current, batch["middle_mask"], stage=1, adaptive=False
    )

    assert torch.isfinite(loss) and float(loss.detach()) > 0
    assert torch.equal(
        canonical[trainer._stage_mask(batch["middle_mask"], 1)],
        batch["gt_ids"][trainer._stage_mask(batch["middle_mask"], 1)],
    )
    assert "train_c2f/q1_soft_recovery" in trainer._metric_sums
    assert trainer._metric_sums["train_c2f/q1_recovery_samples"] > 0
    loss.backward()
    assert torch.isfinite(model.embed_tokens.weight.grad).all()


def test_evaluation_loss_and_hard_latent_metrics_are_finite(tmp_path):
    torch.manual_seed(29)
    config = tiny_config()
    model = AudioMotionTransformer(config)
    collator = VariableGapC2FCollator(config)
    examples = [
        VariableGapMaskExample(
            name="gap1",
            left_idx=0,
            right_idx=2,
            gap_frames=1,
            motion_tokens=[raw_frame(0), raw_frame(1), raw_frame(2)],
            audio_features=torch.randn(3, 3),
        ),
        VariableGapMaskExample(
            name="gap2",
            left_idx=0,
            right_idx=3,
            gap_frames=2,
            motion_tokens=[raw_frame(0), raw_frame(1), raw_frame(2), raw_frame(3)],
            audio_features=torch.randn(4, 3),
        ),
    ]
    builder = ResidualTargetBuilder(
        {part: torch.randn(2, 4, 8) for part in ("upper", "lower")},
        part_order=("upper", "lower"),
        codebook_size=4,
        num_quantizers=2,
    )
    trainer = VariableGapC2FTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_eval_batch_size=2,
            prediction_loss_only=True,
            report_to="none",
            remove_unused_columns=False,
        ),
        eval_dataset=examples,
        data_collator=collator,
        target_builder=builder,
        stage_weights=(0.5, 0.5),
        self_forcing_warmup_ratio=0.0,
        self_forcing_ramp_ratio=0.0,
        self_forcing_max_prob=0.0,
        adaptive_target_mode="never",
        embedding_loss_weight=0.1,
        final_latent_loss_weight=0.1,
    )

    metrics = trainer.evaluate()
    logged = trainer.state.log_history[-1]

    assert torch.isfinite(torch.tensor(metrics["eval_loss"]))
    assert torch.isfinite(torch.tensor(logged["eval_c2f/hard_latent_rmse"]))
    assert "eval_c2f/hard_latent_rmse_q0_wrong" in logged
    assert "eval_c2f/hard_latent_rmse_gap_1_3" in logged


def test_audio_residual_posterior_is_audio_sensitive_and_slot_constrained():
    torch.manual_seed(37)
    config = tiny_config()
    model = AudioMotionTransformer(config).eval()
    books = tiny_codebooks()
    model.configure_audio_residual_posterior(
        books,
        config.part_order,
        config.num_quantizers_per_part,
        mode="residual_posterior",
        gate_init=2.0,
    )
    input_ids = torch.tensor(
        [[0, 4, 8, 12, 16, 16, 16, 16, 2, 6, 10, 14]]
    )
    attention = torch.ones_like(input_ids, dtype=torch.bool)
    audio = torch.randn(1, 3, 3)

    with torch.no_grad():
        first = model(
            input_ids,
            audio_features=audio,
            attention_mask=attention,
            audio_prior_stage=0,
            negative_audio_features=audio.roll(1, dims=1),
            return_audio_prior_details=True,
        )
        second = model(
            input_ids,
            audio_features=audio.flip(1),
            attention_mask=attention,
            audio_prior_stage=0,
        )

    assert set(first["audio_prior"]) == {(0, 0), (1, 0)}
    details = first["audio_prior"][(0, 0)]
    assert details["mu"].shape == (1, 3, 8)
    assert details["prior_logits"].shape == (1, 3, 4)
    assert details["negative_prior_logits"].shape == (1, 3, 4)
    assert details["gate"].min() > 0 and details["gate"].max() < 1
    assert not torch.allclose(first["logits"], second)
    valid = model._slot_valid_mask(input_ids.shape[1], input_ids.device)
    invalid_logits = first["logits"].masked_select(~valid.unsqueeze(0))
    assert invalid_logits.eq(torch.finfo(first["logits"].dtype).min).all()
    generated = model.generate_sbs(
        input_ids,
        audio,
        attention_mask=attention,
    )
    assert not generated.eq(config.mask_token_id).any()


def test_audio_residual_losses_train_posterior_on_self_forced_prefixes(tmp_path):
    torch.manual_seed(41)
    config = tiny_config()
    model = AudioMotionTransformer(config).train()
    collator = VariableGapC2FCollator(config)
    books = tiny_codebooks()
    model.configure_audio_residual_posterior(
        books,
        config.part_order,
        config.num_quantizers_per_part,
        mode="residual_posterior",
        gate_init=-2.0,
    )
    batch = collator(
        [
            VariableGapMaskExample(
                name="a",
                left_idx=0,
                right_idx=3,
                gap_frames=2,
                motion_tokens=[raw_frame(i) for i in range(4)],
                audio_features=torch.randn(4, 3),
            ),
            VariableGapMaskExample(
                name="b",
                left_idx=0,
                right_idx=3,
                gap_frames=2,
                motion_tokens=[raw_frame((i + 1) % 4) for i in range(4)],
                audio_features=torch.randn(4, 3),
            ),
        ]
    )
    trainer = VariableGapC2FTrainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path),
            per_device_train_batch_size=2,
            report_to="none",
            remove_unused_columns=False,
        ),
        target_builder=ResidualTargetBuilder(
            books,
            part_order=config.part_order,
            codebook_size=4,
            num_quantizers=2,
        ),
        stage_weights=(0.0, 1.0),
        self_forcing_warmup_ratio=0.0,
        self_forcing_ramp_ratio=0.0,
        self_forcing_max_prob=1.0,
        adaptive_target_mode="never",
        embedding_loss_weight=0.1,
        final_latent_loss_weight=0.1,
        audio_residual_soft_weight=0.1,
        audio_residual_nll_weight=0.05,
        audio_residual_rank_weight=0.05,
        audio_residual_topk=2,
        audio_residual_rank_margin=0.2,
    )
    trainer.state.global_step = 1
    trainer.state.max_steps = 1

    loss = trainer.compute_loss(model, batch)
    loss.backward()

    assert torch.isfinite(loss)
    assert "train_c2f/q1_audio_soft_nll" in trainer._metric_sums
    assert "train_c2f/q1_audio_negative_nll" in trainer._metric_sums
    assert "train_c2f/q1_audio_nll_margin" in trainer._metric_sums
    head = model.audio_residual_posterior.heads["part0_q1"]
    assert head.mu.weight.grad is not None
    assert torch.isfinite(head.mu.weight.grad).all()
    assert not model.audio_residual_posterior.codebook(0).requires_grad

    model.eval()
    with torch.no_grad():
        eval_loss = trainer.compute_loss(model, batch)
    assert torch.isfinite(eval_loss)
    assert "eval_c2f/q0_audio_soft_nll" in trainer._metric_sums
    assert "eval_c2f/q1_audio_nll_margin" in trainer._metric_sums


def test_audio_residual_posterior_round_trips_through_pretrained_checkpoint(tmp_path):
    torch.manual_seed(43)
    config = tiny_config()
    model = AudioMotionTransformer(config).eval()
    books = tiny_codebooks()
    model.configure_audio_residual_posterior(
        books,
        config.part_order,
        config.num_quantizers_per_part,
        mode="additive_residual_posterior",
        prior_weight=0.7,
        gate_init=-1.0,
    )
    input_ids = torch.tensor([[0, 4, 8, 12, 16, 16, 16, 16]])
    audio = torch.randn(1, 2, 3)
    with torch.no_grad():
        expected = model(input_ids, audio_features=audio, audio_prior_stage=1)

    model.save_pretrained(tmp_path)
    loaded = AudioMotionTransformer.from_pretrained(
        tmp_path, local_files_only=True
    ).eval()
    with torch.no_grad():
        actual = loaded(input_ids, audio_features=audio, audio_prior_stage=1)

    assert loaded.uses_audio_residual_posterior
    assert loaded.config.audio_conditioning_mode == "additive_residual_posterior"
    assert torch.equal(loaded.audio_residual_posterior.codebook(0), books["upper"])
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
