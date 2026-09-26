# Retained files

This package is intentionally small. The retained groups are:

- `prepare_msu_benchmark.py`, `prepare_siwmv2_benchmark.py`: raw-data preparation.
- `configs/`: one dataset map and one shared E0-E6 experiment config.
- `src/improved_ddpm/data/`: manifests, deterministic frame sampling, MTCNN
  preprocessing, dataset reader and balanced samplers.
- `src/improved_ddpm/data/adapters/`: only MSU-MFSD and SiW-Mv2 adapters.
- `src/improved_ddpm/protocols/`: only the controlled M→S/S→M builder.
- `src/improved_ddpm/models/`, `diffusion/`, `fas/`, `experiments/`: E0-E6 model
  and spoof-amplitude DDPM implementation.
- `src/improved_ddpm/fas/synthetic_quality.py`: source-only E6 gates, rolling
  acceptance control, spectral descriptors and offline quality metrics.
- `src/improved_ddpm/scripts/`: build, preprocess, verify, train, evaluate and
  the two controlled benchmark commands.
- `src/improved_ddpm/scripts/evaluate_synthetic_quality.py`: fixed source-train
  synthetic-quality report and qualitative grid.
- `src/improved_ddpm/scripts/pretrain_amplitude_ddpm.py`: independent
  source-train attack-only DDPM pretraining (EMA intentionally disabled).
- `src/improved_ddpm/scripts/evaluate_ddpm_sampling_k.py`: source-only
  K=20/50/100 quality comparison and deterministic K selection.
- `tests/`: unit/integration tests for retained functionality.

Not included: OULU/CASIA/Replay-Attack adapters, old SiW adapter, MCIO scripts,
legacy `root_dir/val_ratio` dataset splitting, demos, notebooks, caches,
generated data, checkpoints, outputs, redundant reports and old ZIP files.
