# MSU-MFSD ↔ SiW-Mv2 controlled face PAD benchmark

This is the cleaned research package for the complete E0-E6 ablation:

| Variant | RGB | FFT | Patch | Contrastive | Center | DDPM |
|---|---:|---:|---:|---:|---:|---:|
| E0 | ✓ |  |  |  |  |  |
| E1 |  | ✓ |  |  |  |  |
| E2 | ✓ | ✓ |  |  |  |  |
| E3 | ✓ | ✓ | ✓ |  |  |  |
| E4 | ✓ | ✓ | ✓ | ✓ |  |  |
| E5 | ✓ | ✓ | ✓ | ✓ | ✓ |  |
| E6 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

E0 is the RGB-only baseline. E5 is the complete discriminative classifier
without DDPM. E6 changes only the spoof-amplitude DDPM intervention relative
to E5. The spoof-amplitude DDPM is pretrained separately on source-train
attacks, selected with source-only diagnostics, and frozen during classifier
training. The classifier batch stays fixed at
`8 live + 4 real attack + 4 synthetic attack`.

E6 now fails closed: generated normalized log-amplitude is clipped to ±3σ,
projected to conjugate-symmetric Fourier amplitude, combined directly with
the parent live phase (`A_syn=A_DDPM`), and checked using numerical and
source-calibrated spectral gates. A
rejected slot is retried up to three times; if it still fails, a real attack
fills that slot. The classifier therefore always remains 16 samples with an
8:8 live/attack prior. If rolling acceptance stays below 80% for 100 batches,
synthetic augmentation is disabled while DDPM/classifier training continues
with real attacks.

Only MSU-MFSD and SiW-Mv2 are supported. OULU-NPU, legacy automatic
`root_dir/val_ratio` splitting, MCIO wrappers, notebooks, caches and duplicate
documentation were intentionally removed.

## Locked benchmark

| Setting | Source train/dev | Frozen target test |
|---|---|---|
| `M_to_S` | MSU official train; 20% subject-disjoint source-dev | SiW-Mv2 Protocol-I test |
| `S_to_M` | SiW-Mv2 Protocol-I train; 20% subject-disjoint source-dev | MSU official test |

The target never participates in training, checkpoint selection, threshold
selection, FFT normalization or hyperparameter tuning. Each video is first
sampled to 32 deterministic frames. After the source-dev split, the benchmark
uses 16/32/32 frames per source-train/source-dev/target-test video. Video scores
are arithmetic means. The threshold is selected by EER on source-dev and frozen.

## Installation

```bash
python -m pip install -r requirements.txt
python -m pip install --no-deps facenet-pytorch==2.6.0
python -m pip install -e .
```

FFmpeg (`ffmpeg` and `ffprobe`) must be available on `PATH` so corrupt video
packets can be skipped without truncating temporal sampling.

## 1. Prepare MSU-MFSD

Your existing successful processed directory can be reused:

```text
data/processed/MSU-MFSD/official/face_context
```

For a fresh run:

```bash
python prepare_msu_benchmark.py \
  --raw-root raw_data/MSU-MFSD-Publish
```

## 2. Prepare SiW-Mv2 Protocol I

Required raw layout:

```text
SiW-Mv2/
├── Live/
└── Spoof/
    ├── Print/
    ├── Replay/
    └── ...
```

The official protocol directory must contain:

```text
trainlist_all.txt
trainlist_live.txt
testlist_all.txt
testlist_live.txt
```

Run:

```bash
python prepare_siwmv2_benchmark.py \
  --raw-root raw_data/SiW-Mv2 \
  --protocol-dir raw_data/SiW-Mv2/source_SiW_Mv2/pro_3_text
```

If the protocol source is outside the data directory, pass its actual path.
Repeated official list lines are removed from the manifest and recorded in
`siwmv2_adapter_report.json`; class balancing is applied later by the sampler.

## 3. Build both cross-dataset settings

```bash
cp configs/datasets_ms.example.yaml configs/datasets_ms.yaml

python -m improved_ddpm.scripts.prepare_cross_dataset_ms \
  --datasets-config configs/datasets_ms.yaml \
  --settings all \
  --source-dev-policy holdout \
  --source-dev-fraction 0.20 \
  --seed 42 \
  --train-frames 16 \
  --dev-frames 32 \
  --test-frames 32 \
  --image-size 224
```

## 4. Debug before the full experiment

One short run covering RGB-only, complete classifier and DDPM paths:

```bash
python -m improved_ddpm.scripts.run_cross_dataset_ms \
  --base-config configs/ms_e0_e6.yaml \
  --settings M_to_S \
  --variants E0 E5 E6 \
  --seeds 42 \
  --max-steps 20 \
  --ddpm-pretrain-steps 20 \
  --sampling-quality-count 8 \
  --output-root data/experiments/smoke_test
```

This is a pipeline check, not a publication result.

## 5. Full controlled comparison

```bash
python -m improved_ddpm.scripts.run_cross_dataset_ms \
  --base-config configs/ms_e0_e6.yaml \
  --settings M_to_S S_to_M \
  --variants all \
  --seeds 42 43 44 \
  --max-steps 3000 \
  --augmentation-ratio 0.5 \
  --ddpm-pretrain-steps 3000 \
  --sampling-k-candidates 20 50 100 \
  --output-root data/experiments/cross_dataset_ms
```

This creates 42 runs (`2 directions × 7 variants × 3 seeds`). Outputs are:

```text
data/experiments/cross_dataset_ms/comparison.csv
data/experiments/cross_dataset_ms/ablation_summary.csv
data/experiments/cross_dataset_ms/ablation_paired_deltas.csv
data/experiments/cross_dataset_ms/benchmark_plan.json
data/experiments/cross_dataset_ms/ddpm_pretrain/<setting>/seed_<seed>/ddpm_final.pt
data/experiments/cross_dataset_ms/ddpm_pretrain/<setting>/seed_<seed>/sampling_k_evaluation/sampling_k_selection.json
```

`comparison.csv` contains every seed. `ablation_summary.csv` contains mean and
population standard deviation. `ablation_paired_deltas.csv` compares matching
seeds for RGB-vs-FFT, adding FFT, patch, contrastive, center, DDPM, and the full
method versus RGB-only. `benchmark_plan.json` records the exact 42-run matrix
before training starts.

Every run also stores the frozen threshold, frame/video predictions, ROC files,
HTER/AUC/EER/APCER/BPCER/ACER and per-attack APCER. SiW-Mv2 per-attack numbers
are explicitly reported as Protocol-I target stratification, not as official
Protocol-II scores.

For E6, online quality statistics are written to:

```text
data/experiments/cross_dataset_ms/<setting>/E6/seed_<seed>/logs/synthetic_quality.csv
```

The runner automatically pretrains one source-only DDPM per direction/seed,
compares K=20/50/100, chooses K without target data, freezes the selected DDPM,
and then trains E6. EMA is intentionally disabled in this version.

After an E6 run, create a fixed source-train-only quality report (no target
data is read):

```bash
python -m improved_ddpm.scripts.evaluate_synthetic_quality \
  --config data/experiments/cross_dataset_ms/M_to_S/E6/seed_42/resolved_config.yaml \
  --checkpoint data/experiments/cross_dataset_ms/M_to_S/E6/seed_42/checkpoints/best.pt \
  --output-dir data/experiments/cross_dataset_ms/M_to_S/E6/seed_42/synthetic_quality \
  --count 256 \
  --face-metrics
```

This produces `quality_summary.json`, `synthetic_quality_samples.csv` and
`live_vs_synthetic_grid.png`. Add `--identity-metrics` only when FaceNet
VGGFace2 weights are already available or the machine can download them.

See [docs/CONTROLLED_PROTOCOL.md](docs/CONTROLLED_PROTOCOL.md) for the exact
scientific rules and [PROJECT_FILES.md](PROJECT_FILES.md) for the retained-file
inventory.
