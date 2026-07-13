# Multipart MSD Experiments

The MSD implementation supports both the original single-body RVQ API and the
new SentiAvatar multipart codec. The multipart path uses four independent
codecs in this order:

```text
upper q0..q3 | lower q0..q3 | feet q0..q3 | hands q0..q3
```

RVQ vectors are summed within each part. The four resulting part embeddings are
concatenated before computing the combined MSD. They must not be summed across
parts because the part codecs were trained in independent latent spaces.

Every experiment also reports or caches per-part descriptors so that a good
combined score cannot hide a failing part.

## Prerequisites

Run commands from the repository root. The default paths expect:

```text
SuSuInterActs/SuSuInterActs/motion_token_data_multipart_512x4
checkpoints/multipart_rvqvae/rvq_upper_512x4_bs256_cosine/model/best.pth
checkpoints/multipart_rvqvae/rvq_lower_512x4_bs256_cosine/model/best.pth
checkpoints/multipart_rvqvae/rvq_feet_512x4_bs256_cosine/model/best.pth
checkpoints/multipart_rvqvae/rvq_hands_512x4_bs256_cosine/model/best.pth
```

If multipart token JSONs do not exist yet:

```bash
python motion_generation/scripts/export_multipart_motion_tokens.py \
  --data_dir SuSuInterActs/SuSuInterActs \
  --output_dir SuSuInterActs/SuSuInterActs/motion_token_data_multipart_512x4 \
  --upper_ckpt checkpoints/multipart_rvqvae/rvq_upper_512x4_bs256_cosine/model/best.pth \
  --lower_ckpt checkpoints/multipart_rvqvae/rvq_lower_512x4_bs256_cosine/model/best.pth \
  --feet_ckpt checkpoints/multipart_rvqvae/rvq_feet_512x4_bs256_cosine/model/best.pth \
  --hands_ckpt checkpoints/multipart_rvqvae/rvq_hands_512x4_bs256_cosine/model/best.pth \
  --device cuda:0
```

## 1. Integration Check

Load all checkpoints, validate the token manifest, and decode five clips:

```bash
python motion_generation/utils/msd/precompute_msd.py inspect \
  --device cuda:0 --split val --n 5
```

Do not continue if checkpoint parts, token slots, or decoded shapes disagree.

## 2. Token/Decoded Agreement Gate

Run the quick study first:

```bash
python motion_generation/utils/msd/precompute_msd.py agreement \
  --device cuda:0 --split val --n 50 \
  --out motion_generation/utils/msd/outputs/multipart/agreement_val_50.csv
```

Then run the 500-clip gate:

```bash
python motion_generation/utils/msd/precompute_msd.py agreement \
  --device cuda:0 --split train --n 500 \
  --out motion_generation/utils/msd/outputs/multipart/agreement_train_500.csv
```

The command reports median Spearman agreement for the combined descriptor and
for upper, lower, feet, and hands separately. The default gate is `0.9` for all
five readings. If one part fails, inspect it before using combined token MSD as
a training weight.

## 3. Visual Checks

Plot the first five validation clips:

```bash
python motion_generation/utils/msd/precompute_msd.py heatmaps \
  --device cuda:0 --split val --n 5
```

Or select exact split names:

```bash
python motion_generation/utils/msd/precompute_msd.py heatmaps \
  --device cuda:0 --split val \
  --clips fbx_to_json_data_susu_retarget_maya/20250910/Human_0908_188-9_01
```

Plots contain the combined spectral heatmap and combined/per-part Omega curves.

## 4. Decoded-Motion Energy Floor

The energy floor is only for decoded-motion MSD, not token MSD:

```bash
python motion_generation/utils/msd/precompute_msd.py calibrate-floor \
  --device cuda:0 --split train --n 200
```

Pass the suggested value to `agreement --energy-floor VALUE` and confirm that
the agreement gate still passes.

## 5. Cache For Training

The current Step 2 dataset does not use speed augmentation, so cache only
`1.0` unless the experiment explicitly trains with other speeds:

```bash
python motion_generation/utils/msd/precompute_msd.py cache \
  --device cuda:0 \
  --splits train val test \
  --speeds 1.0 \
  --out motion_generation/utils/msd/outputs/multipart/cache_token_msd
```

For the original speed-augmentation study:

```bash
python motion_generation/utils/msd/precompute_msd.py cache \
  --device cuda:0 \
  --splits train val test \
  --speeds 0.9 1.0 1.1 \
  --out motion_generation/utils/msd/outputs/multipart/cache_token_msd_speed_aug
```

Each NPZ contains:

```text
phi, omega, weight
phi_upper, omega_upper, weight_upper
phi_lower, omega_lower, weight_lower
phi_feet, omega_feet, weight_feet
phi_hands, omega_hands, weight_hands
```

Load a cache entry in training code with:

```python
import sys

sys.path.insert(0, "motion_generation")
from utils.msd.multipart_adapter import MultipartMSDCache

cache = MultipartMSDCache("motion_generation/utils/msd/outputs/multipart/cache_token_msd")
arrays = cache.get(clip_name, speed=1.0)
frame_weights = arrays["weight"]
```

`frame_weights` is on the token timeline. Repeat each selected frame weight over
the 16 token slots before applying it to per-token cross entropy. For Step 2,
apply loss only at masked middle positions, as before.

## Other Files

`task2_label_calibration.py` computes thresholds from raw motion and FK, so it
does not depend on either codec and is run unchanged:

```bash
python motion_generation/utils/msd/task2_label_calibration.py
```

`verify_msd_susuinteracts.ipynb` remains the historical old-codec walkthrough.
Use the CLI above for multipart results so old and new outputs cannot be mixed in
one notebook state.

Run synthetic core tests from the MSD directory. This avoids the repository's
legacy `motion_generation.utils` package-import assumptions:

```bash
cd motion_generation/utils/msd
pytest test_msd.py -v
```
