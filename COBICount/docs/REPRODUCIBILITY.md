# Reproducibility and protocol checklist

## Strict source-only boundary

A run is protocol-compliant only when all of the following are true:

- optimization uses one annotated source domain;
- checkpoint selection uses source-domain validation only;
- no target image, annotation, pseudo-label, feature statistic, count statistic, metric, visualization, or hard-negative mining result influences training;
- the source checkpoint is frozen before target-domain evaluation;
- the same frozen checkpoint is used across all reported target subsets unless the protocol explicitly states otherwise.

The public `cobicount-train` module imports only `RSOCBuildingDataset`. DOTA loaders reside in the separate evaluation entry point. The protocol test in `tests/test_protocol.py` checks this separation.

## Default configuration

- epochs: 80;
- batch size: 6;
- optimizer: AdamW;
- initial learning rate: `1.8e-4`;
- weight decay: `1e-4`;
- AMP: enabled on CUDA;
- gradient clipping: 5.0;
- EMA decay: 0.999;
- seed: 3407;
- input/patch size: 512;
- score stride: 4.

The relative and logarithmic count losses warm up over the first eight epochs. Audit supervision starts at epoch 10.

## Sources of variation

Exact values may differ due to:

- GPU architecture and CUDA/cuDNN versions;
- PyTorch kernel changes;
- nondeterministic CUDA operations;
- dataset preprocessing or split differences;
- image decoding libraries;
- annotation parsing differences;
- full-image inference scale and file-list differences.

Use `--deterministic` for stronger repeatability when debugging. It can reduce speed and does not guarantee identical results across software/hardware stacks.

## What to publish with a checkpoint

For each public weight file, publish:

- filename and SHA-256 checksum;
- source-domain split definition;
- exact command line;
- `run_config.json`;
- package commit hash and release tag;
- PyTorch/CUDA versions;
- whether EMA or raw model weights are evaluated;
- evaluation file lists and class filters;
- reported image resizing rules.

Weights should be attached to a versioned GitHub Release or an archival repository rather than committed into Git history.
