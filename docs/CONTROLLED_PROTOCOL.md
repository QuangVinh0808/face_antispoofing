# Controlled MSU-MFSD ↔ SiW-Mv2 protocol

## Purpose

The primary claim is cross-dataset generalization under a controlled E0-E6
ablation. It is not an official within-dataset SiW-Mv2 Protocol-I score
for `M_to_S`, because the model is trained on MSU-MFSD.

## Data partitions

- `M_to_S`: source = MSU-MFSD official 15-subject training partition; target =
  SiW-Mv2 Protocol-I test partition.
- `S_to_M`: source = SiW-Mv2 Protocol-I training partition; target = MSU-MFSD
  official 20-subject test partition.
- A deterministic 20% source-subject holdout (seed 42) supplies source-dev when
  the source has no official development split.
- Holdout is stratified by each subject's class signature. Train and dev must
  both contain live and attack samples.
- No subject, video or path may overlap across train/dev/test.

## Preprocessing

1. Uniform deterministic selection of 32 frames per raw video.
2. Largest-face MTCNN detection.
3. `face_context` crop expanded by 1.3×.
4. Lossless RGB PNG at 256×256.
5. Model input resized to 224×224 and mapped from `[0,1]` to `[-1,1]`.
6. After source-dev exists: 16 source-train, 32 source-dev and 32 target-test
   frames per video.
7. Classifier FFT normalization is computed from every final source-train row.
8. A separate DDPM normalization is computed from source-train attacks only.
   Manifest fingerprints prevent silently reusing either statistic from
   another split.
9. Spectral-gate centroids and empirical bounds are calibrated from
   source-train only.

## E0-E6 definitions

| Variant | Definition | Scientific comparison |
|---|---|---|
| E0 | RGB full image + cross-entropy | RGB-only baseline |
| E1 | FFT log-amplitude + cross-entropy | FFT-only baseline |
| E2 | RGB + FFT + cross-entropy | E2−E0: added FFT branch |
| E3 | E2 + patch training stream | E3−E2: patch contribution |
| E4 | E3 + frequency supervised contrastive loss | E4−E3: contrastive contribution |
| E5 | E4 + center loss | E5−E4: center contribution |
| E6 | E5 + D2 spoof-amplitude DDPM | E6−E5: DDPM contribution |

E1 is an alternative modality baseline rather than a cumulative successor to
E0. Therefore the effect of adding FFT to RGB is measured by E2−E0, not E2−E1.

## Fairness controls

All variants use the same manifest, frame indices, seed, fixed class-balanced
classifier batch size, model-input size, classifier optimizer, source-dev checkpoint rule,
video aggregation and frozen threshold. E0 performs no FFT computation and
does not load FFT statistics. E1-E6 use the same source-train-only FFT
normalization. E0-E5 disable Stage-6. For E6, D2 is pretrained separately on
source-train attacks for a fixed update budget. K=20/50/100 are compared using
source-train quality statistics; the selected DDPM checkpoint is frozen before
classifier optimization. Four real attack slots are replaced by four synthetic
attacks: 8 live + 4 real attacks + 4 synthetic attacks. Thus E5
and E6 both process 16 classifier samples with the same 50/50 class prior.
The DDPM is never updated during classifier training, making the generated
distribution stable and reproducible.

## E6 synthetic quality control

Quality thresholds are fitted once from source-train only. Source-dev and
target-test are never reference distributions for the gate.

1. DDPM normalized log-amplitude is clamped to `[-3, 3]` before inverse-log.
2. Generated amplitude is projected onto the conjugate-symmetric Fourier
   subspace required by real-valued IFFT reconstruction.
3. The synthesis rule is exactly `A_syn=A_DDPM`; phase remains the parent live
   phase. No `A_mix` interpolation is used.
4. A candidate must be finite, have the configured tensor shape, clip at most
   1% of pixels, fall inside source-calibrated attack PSD distance/margin
   bounds, and fall inside the source-attack high-frequency interval.
5. Failed slots are retried at most three times. Remaining slots fall back to
   real source attacks, so the classifier still sees 8 live and 8 attack.
6. If acceptance over 100 consecutive active batches falls below 80%,
   synthetic augmentation is disabled for the rest of the run. Non-finite
   DDPM loss or generated pixels stops the run instead of silently training on
   invalid data.

`synthetic_quality.csv` records requested/generated/accepted/rejected counts,
retry and real-fallback counts, rejection reasons, pixel clipping, attack
margin, attack distance, high-frequency ratio and rolling
acceptance.

Structure/face checks are not hard online gates because the DDPM models
amplitude rather than a direct RGB face. The offline evaluator reads only
source-train and reports the online-gate acceptance rate, MTCNN detection
(optional), FaceNet identity cosine
(optional), radial-PSD MMD to real attacks, spectral diversity, perceptual-hash
duplicates and nearest-source-attack PSD distance. These support the generator
analysis; the primary method claim remains the paired E6-vs-E5 target-test
HTER/AUC/EER and macro/worst APCER over seeds 42/43/44.

## Evaluation

- Label convention: 0 = live, 1 = attack; larger score means more likely attack.
- Aggregate frame attack probabilities with an arithmetic mean per video.
- Choose the threshold at EER on source-dev only; never recalibrate on target.
- Report video-level HTER, AUC, EER, APCER, BPCER and ACER.
- For SiW-Mv2, also report per-attack, macro and worst-case APCER after the
  model and threshold are frozen. These are Protocol-I target slices, not an
  official Protocol-II retraining experiment.
- Run seeds 42, 43 and 44 and report mean ± population standard deviation.
- The full experiment contains 42 runs: 2 directions × 7 variants × 3 seeds.
- EMA is not used in this version; it is reserved for a later controlled
  ablation if raw-weight DDPM sampling remains unstable.

## Target-leakage prohibition

Target data cannot affect optimization, epoch/checkpoint choice, threshold,
aggregation, frame policy, augmentation settings, normalization or any
hyperparameter. If a decision is changed after seeing target results, the
result is exploratory and must not be presented as the locked benchmark.
