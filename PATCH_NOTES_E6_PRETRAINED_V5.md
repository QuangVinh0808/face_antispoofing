# E6 pretrained-DDPM patch v5 (EMA disabled)

## Locked method changes

- `A_syn = A_DDPM`; no `A_mix` interpolation.
- Structure cosine is removed from the online hard gate.
- MTCNN/FaceNet are offline diagnostics only (the original dataset face crop
  still uses MTCNN during preprocessing).
- Classifier normalization uses all source-train images; DDPM normalization
  uses source-train attacks only.
- DDPM is pretrained independently, then frozen during E6 classifier training.
- Generated shifted amplitude is projected to conjugate symmetry before IFFT.
- K=20/50/100 are compared with source-only statistics; the runner injects the
  selected K into E6.
- Spectral thresholds are empirical source-train attack quantiles and are
  fingerprinted to the exact manifest.
- EMA is explicitly disabled and recorded as `ema: false`.

## Files modified relative to the previous quality-controlled version

1. `configs/ms_e0_e6.yaml`
2. `src/improved_ddpm/experiments/stage6.py`
3. `src/improved_ddpm/fas/amplitude_ddpm.py`
4. `src/improved_ddpm/fas/frequency_normalization.py`
5. `src/improved_ddpm/fas/frequency_utils.py`
6. `src/improved_ddpm/fas/synthetic_quality.py`
7. `src/improved_ddpm/scripts/evaluate_synthetic_quality.py`
8. `src/improved_ddpm/scripts/export_stage6_samples.py`
9. `src/improved_ddpm/scripts/prepare_cross_dataset_ms.py`
10. `src/improved_ddpm/scripts/run_cross_dataset_ms.py`
11. `src/improved_ddpm/scripts/train.py`
12. `tests/test_cross_dataset_e0_e6_runner.py`
13. `tests/test_preprocessing_stage3.py`
14. `tests/test_stage6_ddpm.py`
15. `tests/test_synthetic_quality.py`
16. `README.md`
17. `docs/CONTROLLED_PROTOCOL.md`
18. `PROJECT_FILES.md`

## Files added

1. `src/improved_ddpm/scripts/pretrain_amplitude_ddpm.py`
2. `src/improved_ddpm/scripts/evaluate_ddpm_sampling_k.py`

## Required migration

Do not delete processed MSU/SiW images. Rebuild only the controlled benchmark
metadata so the two new source-only files are produced:

```bash
python -m improved_ddpm.scripts.prepare_cross_dataset_ms \
  --datasets-config configs/datasets_ms.example.yaml \
  --settings all \
  --source-dev-policy holdout \
  --source-dev-fraction 0.20 \
  --seed 42 \
  --train-frames 16 \
  --dev-frames all \
  --test-frames all \
  --output-root data/benchmarks/cross_dataset_ms \
  --force
```

This creates, per setting:

- `normalization_224.json` — classifier, all source-train rows.
- `normalization_attack_224.json` — DDPM, source-train attacks only.
- `spectral_gate_source_train_224.json` — source-calibrated gate.

## Smoke test

```bash
python -m improved_ddpm.scripts.run_cross_dataset_ms \
  --base-config configs/ms_e0_e6.yaml \
  --benchmark-root data/benchmarks/cross_dataset_ms \
  --settings M_to_S \
  --variants E6 \
  --seeds 42 \
  --max-steps 20 \
  --ddpm-pretrain-steps 20 \
  --sampling-k-candidates 20 50 100 \
  --sampling-quality-count 8 \
  --output-root data/experiments/smoke_e6_pretrained_v5
```

The 20-step pretrain is only a pipeline test, not a quality experiment.

## Full benchmark

```bash
python -m improved_ddpm.scripts.run_cross_dataset_ms \
  --base-config configs/ms_e0_e6.yaml \
  --benchmark-root data/benchmarks/cross_dataset_ms \
  --settings M_to_S S_to_M \
  --variants all \
  --seeds 42 43 44 \
  --max-steps 3000 \
  --ddpm-pretrain-steps 3000 \
  --sampling-k-candidates 20 50 100 \
  --augmentation-ratio 0.5 \
  --output-root data/experiments/cross_dataset_ms_v5
```

Use a new output root so old E6 results cannot be silently reused.
