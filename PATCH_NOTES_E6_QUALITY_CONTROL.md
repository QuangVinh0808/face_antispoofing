# E6 synthetic-quality-control patch

This patch is based on the step-scheduled E0-E6 package (3000 maximum steps,
300-step E6 warm-up, 8L+4R+4S after warm-up).

## Training behavior

- Keeps classifier batch size at 16 and class prior at 8 live / 8 attack.
- Clips generated normalized log-amplitude to ±3σ.
- Mixes generated and parent-live amplitude with strength sampled from
  `[0.5, 1.0]`; parent-live phase is retained.
- Applies source-train-only numerical, structure and attack-spectrum gates.
- Retries rejected synthetic slots up to three times.
- Fills unresolved slots with real attacks.
- Disables synthetic augmentation after a sustained rolling acceptance rate
  below 80% over 100 active batches.
- Stops immediately on non-finite DDPM loss or generated values.

## Outputs

- Online interval log: `logs/synthetic_quality.csv` inside each E6 run.
- Offline source-train report: `quality_summary.json`.
- Offline per-sample data: `synthetic_quality_samples.csv`.
- Qualitative panel: `live_vs_synthetic_grid.png`.

## Offline command

```bash
python -m improved_ddpm.scripts.evaluate_synthetic_quality \
  --config data/experiments/cross_dataset_ms/M_to_S/E6/seed_42/resolved_config.yaml \
  --checkpoint data/experiments/cross_dataset_ms/M_to_S/E6/seed_42/checkpoints/best.pt \
  --output-dir data/experiments/cross_dataset_ms/M_to_S/E6/seed_42/synthetic_quality \
  --count 256 \
  --face-metrics
```

All selection and gate references remain source-only. Target-test data is used
only by the existing frozen-policy evaluation command.
