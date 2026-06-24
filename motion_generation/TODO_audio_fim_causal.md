# TODO: Compact AudioFIM Causal Step 2

This TODO tracks the small causal audio-aware infill transformer path.

Keep this path separate from:

- `models/audio_motion_model.py`: original BERT-style mask transformer.
- `models/vllm_infill_model.py`: Qwen/vLLM-compatible Step 2 infill.
- `scripts/train_vllm_infill.py`: Qwen/vLLM fine-tuning trainer.

Current model path:

- `models/audio_fim_causal_model.py`
- `scripts/train_audio_fim_causal.py`
- default checkpoint: `checkpoints/audio_fim_causal`

## Current Decisions

- Use continuous HuBERT layer9 features, not quantized audio tokens.
- Use compact old-Step-2 motion token IDs:
  - `res_1`: `0..511`
  - `res_2`: `512..1023`
  - `res_3`: `1024..1535`
  - `res_4`: `1536..2047`
- Keep `embed_tokens` and `out_head` untied.
- Match old Step 2 core size:
  - layers: 8
  - hidden: 512
  - heads: 16
  - FFN: 1536
  - params: about 24M
- First training target is the classic fixed gap:
  - left anchor at `t`
  - right anchor at `t+4`
  - predict `t+1`, `t+2`, `t+3`
- Keep `[LEN_N]` and `--step` explicit so variable gaps can be added later.
- Keep DDP/Accelerate launch on hold while debugging correctness.

## Known Good Sanity Signals

From the debug run:

- `Tie embeddings: False`
- `Shared emb/head: False`
- `zero-logit CE ~= log(2075) ~= 7.6377`
- initial `model CE ~= 7.875`
- supervised labels per example: `13`
- sequence length with 0 history: `32`
- sequence length with max sampled history: up to `64`

Important note:

- In the current debugpy multi-GPU/DataParallel mode, logged Trainer loss appears to be summed across 4 GPUs.
- A logged loss near `31` corresponds to true CE around `31 / 4 ~= 7.75`.
- For now, interpret debugpy multi-GPU loss with that scaling in mind.

## Immediate TODO

- [ ] Run a short single-GPU debug pass to confirm loss logs directly around `7-8`.
- [ ] Run a short multi-GPU debugpy pass and compare logged loss divided by GPU count.
- [ ] Keep `--debug_loss_sanity 4` available for correctness checks, but do not use it during future DDP runs until it is rank-aware.
- [ ] Confirm that `--motion_fps 20` and `--audio_fps 50` are used for the current SuSuInterActs preprocessing.
- [ ] Confirm the output folder is clean before serious training to avoid mixing checkpoints from the tied-weight version.

## Training TODO

- [ ] Start with a small smoke run:
  - `--max_samples 8`
  - `--max_windows_per_sequence 4`
  - `--debug_examples 1`
  - `--debug_loss_sanity 4`
- [ ] Run a medium overfit test on a small subset.
  - Goal: loss should drop clearly below random baseline.
  - Check generated middle frames manually.
- [ ] Run full debug-mode training after the overfit test looks sane.
- [ ] Add eval loss logging/checkpoint selection once the basic curve is trusted.
- [ ] Decide whether 100 epochs is useful or excessive after seeing the first full curve.

## Data And Collator TODO

- [ ] Profile whether collation remains a bottleneck with large batches.
  - Current profile: about `1.6-2.0s` for `4096` examples.
- [ ] Consider caching encoded compact FIM examples if collation becomes a real bottleneck.
- [ ] Verify audio alignment by printing several `(motion_idx -> audio_idx)` pairs.
- [ ] Check whether using one audio feature per motion frame is enough or whether middle frames should receive pooled local audio windows.
- [ ] Keep action text out for now; revisit only after audio/motion-only training is stable.

## Model TODO

- [ ] Add token-level accuracy on supervised positions.
- [ ] Add per-quantizer accuracy:
  - `res_1`
  - `res_2`
  - `res_3`
  - `res_4`
- [x] Add teacher-forced eval token/top-k/per-quantizer metrics.
- [x] Add free-running eval token/per-quantizer/frame/gap metrics.
- [ ] Consider masking logits by quantizer during training loss.
  - Current training predicts over the whole compact vocab.
  - Inference already restricts each generated token to the expected quantizer range.
- [ ] Consider segment/type embeddings if the model struggles to distinguish history/anchor/audio/target roles.
- [ ] Consider RoPE later; current model uses learned absolute positions initialized sinusoidally.

## Inference TODO

- [ ] Add a standalone inference script for `AudioFIMCausalLM`.
- [ ] Build a tiny demo that loads one val sample and predicts the middle 3 frames from GT anchors.
- [ ] Compare generated middle frames to GT tokens.
- [x] Add W&B token-space GT-vs-pred eval videos.
- [ ] Decode generated dense motion through RVQVAE and visually inspect motion.
- [x] Add W&B decoded skeleton GT-vs-pred eval videos.
- [ ] Add optional BVH export artifacts for selected eval videos.
- [ ] Add a rolling sequence inference helper:
  - use sparse keyframes
  - generate middle 3 frames for each gap
  - assemble dense tokens
- [ ] Add KV-cache only after the simple inference path is correct.

## On Hold

- [ ] DDP/Accelerate launch for real training.
- [ ] Rank-aware debug helpers for DDP.
- [ ] Variable gap length training.
- [ ] Scheduled sampling / generated history.
- [ ] Integration with realtime inference.

## Useful Debug Command Shape

```bash
python motion_generation/scripts/train_audio_fim_causal.py \
  --output_dir checkpoints/audio_fim_causal \
  --data_dir SuSuInterActs/SuSuInterActs \
  --train_split_file SuSuInterActs/SuSuInterActs/split/train_file_list.txt \
  --eval_split_file SuSuInterActs/SuSuInterActs/split/val_file_list.txt \
  --motion2text_json SuSuInterActs/SuSuInterActs/text_data/train.json \
  --audio_fps 50 \
  --motion_fps 20 \
  --step 4 \
  --min_history_frames 0 \
  --max_history_frames 8 \
  --bf16 \
  --gradient_checkpointing \
  --num_train_epochs 3 \
  --learning_rate 1e-4 \
  --per_device_train_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --max_samples 8 \
  --max_windows_per_sequence 4 \
  --debug_examples 1 \
  --debug_loss_sanity 4 \
  --profile_startup \
  --profile_collator_batches 5 \
  --logging_steps 5
```
