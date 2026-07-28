# Migration from the original research scripts

The uploaded research code has been reorganized without changing the core model equations or checkpoint parameter names.

| Original file | Public location | Main change |
|---|---|---|
| `model_redcount_v5_23_bic.py` | `src/cobicount/model.py` | Public `COBICount` alias; legacy class retained; output spelling alias added |
| `loss_redcount_v5_23_bic.py` | `src/cobicount/losses.py` | Packaged API and current autocast compatibility |
| `datasets_redcount_v5_23.py` | `src/cobicount/data.py` | Source/target roles documented; no machine-specific paths |
| `train_redcount_v5_23_bic_rsoc_dota.py` | `src/cobicount/cli/train.py` | DOTA loading/evaluation/checkpoint selection removed; 80-epoch default |
| `test_redcount_v5_23_bic_rsoc_dota.py` | `src/cobicount/cli/evaluate.py` | Independent post-training dataset selection and result export |

Shared functions previously imported from the training script are now in:

- `src/cobicount/engine.py`;
- `src/cobicount/visualization.py`.

## Legacy checkpoint compatibility

The model parameter structure is unchanged. `REDCountV523BIC` remains available, while `COBICount` is an alias. The checkpoint loader supports both old metadata keys (`model_cfg`, `best_rsoc_mae`) and cleaned-release keys (`model_config`, `best_source_val_mae`).

Old checkpoints should be evaluated with:

```bash
cobicount-evaluate --checkpoint /path/to/old_checkpoint.pth ...
```

Do not use the original `best_dota_proxy_v5_23_bic.pth` as the official source-only selected checkpoint. Use the source-selected RSOC checkpoint that produced the manuscript results.
