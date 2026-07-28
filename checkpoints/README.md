# Checkpoints

No model weights are included in Git history.

For each released checkpoint, add a table like:

| Version | Source split | File | SHA-256 | Download |
|---|---|---|---|---|
| v0.1.0 | RSOC Building | `cobicount_rsoc_building_ema.pth` | `TO_BE_ADDED` | GitHub Release asset |

Generate a checksum:

```bash
# Linux/macOS
sha256sum cobicount_rsoc_building_ema.pth

# Windows PowerShell
Get-FileHash cobicount_rsoc_building_ema.pth -Algorithm SHA256
```

Do not load untrusted PyTorch checkpoints.
