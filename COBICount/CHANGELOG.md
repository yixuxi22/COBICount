# Changelog

All notable changes to the public research-code release are documented here.

## [0.1.0] - 2026-07-28

### Added

- installable `cobicount` Python package;
- strict source-only RSOC training entry point;
- independent frozen-checkpoint RSOC/DOTA evaluation entry point;
- single-image inference;
- source-domain checkpoint selection only;
- 80-epoch manuscript-aligned default configuration;
- CPU smoke tests and source-only protocol test;
- GitHub Actions CI, issue templates, citation metadata, model card, and documentation.

### Changed

- removed all machine-specific absolute paths;
- separated training, evaluation, visualization, and shared utilities;
- renamed public API to `COBICount` while retaining the legacy checkpoint-compatible class name;
- added correctly spelled `origin_score_density` output aliases while keeping legacy keys.

### Removed

- target-domain DOTA loader construction from the training process;
- target-domain metric logging and target-selected checkpoint generation during training;
- `best_dota_proxy_v5_23_bic.pth` from the official workflow.
