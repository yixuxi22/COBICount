# COBICount

Official research-code release for **COBICount: Candidate-Origin Bias Isolation for Single-Source Generalizable Remote Sensing Object Counting in Unseen Scenarios**.

COBICount factorizes the final counting score into:

- **Candidate Evidence (CE):** permissive dense response generation;
- **Component Acceptance (CA):** center-aligned local response retention;
- **Bias Isolation (BI):** suppression of repeated structural responses;
- an auxiliary **audit interface** used for diagnosis only.

> The audit channels are not calibrated target-category predictions and do not gate the final count. The continuous predicted count is the spatial integral of the score map and is not constrained to equal the number of extracted diagnostic points.

## Source-only protocol

The cleaned training entry point enforces a strict source-only boundary:

- `cobicount-train` imports and constructs **RSOC source-domain datasets only**;
- target-domain loaders are available only in `cobicount-evaluate`;
- target-domain metrics cannot be used by the training loop for optimization, checkpoint selection, early stopping, or visualization;
- the default training schedule is **80 epochs**, matching the manuscript configuration.

The legacy model class name `REDCountV523BIC` is retained for checkpoint compatibility. The public paper-facing alias is `COBICount`.

## Repository layout

```text
COBICount/
├── src/cobicount/
│   ├── model.py              # CE/CA/BI model and diagnostic point extraction
│   ├── losses.py             # source-derived training objective
│   ├── data.py               # RSOC and post-training DOTA datasets
│   ├── engine.py             # shared training/evaluation utilities
│   ├── visualization.py      # diagnostic maps and result export
│   └── cli/
│       ├── train.py          # strict source-only training
│       ├── evaluate.py       # frozen-checkpoint evaluation
│       └── infer.py          # single-image inference
├── tests/                    # CPU smoke and protocol tests
├── docs/                     # dataset, reproducibility, and release notes
├── .github/                  # CI and contribution templates
├── CITATION.cff
├── LICENSE
└── pyproject.toml
```

## Installation

Python 3.10–3.12 is recommended.

For CUDA systems, install the PyTorch build appropriate for your CUDA environment first. Then install the repository:

```bash
git clone https://github.com/OWNER/COBICount.git
cd COBICount
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
# source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pytest -q
```

## Dataset preparation

Datasets are not redistributed. Obtain them from their official sources and comply with their licenses.

Expected RSOC layout:

```text
RSOC_BUILDING_ROOT/
├── train_data/
│   ├── images/
│   └── ground_truth/
└── test_data/
    ├── images/
    └── ground_truth/
```

Expected DOTA evaluation layout:

```text
DOTA_IMAGE_ROOT/
└── *.png | *.jpg | *.tif

DOTA_ANNOTATION_ROOT/
└── *.txt | *.json
```

See [`docs/DATASETS.md`](docs/DATASETS.md) for annotation parsing details and protocol cautions.

## Training

The following command trains only on RSOC Building and uses the specified source-domain validation split for checkpoint selection:

```bash
cobicount-train \
  --rsoc-root /path/to/RSOC_building/building \
  --output-dir runs/cobicount_rsoc_building \
  --epochs 80 \
  --batch-size 6 \
  --device cuda
```

Resume from the latest epoch checkpoint:

```bash
cobicount-train \
  --rsoc-root /path/to/RSOC_building/building \
  --output-dir runs/cobicount_rsoc_building \
  --resume auto
```

The default RSOC directory layout exposes `test_data` rather than a dedicated validation directory. The CLI therefore defaults to `--val-split test_data` for compatibility with the original implementation. For a new study, use a disjoint source-domain validation split and pass it through `--val-split`.

Training outputs include:

- `best_source_val.pth` — checkpoint selected by source-domain validation MAE;
- `epoch_XXX.pth` — resumable periodic checkpoints;
- `metrics.csv` and `training_curves.png`;
- source-validation diagnostic visualizations;
- `run_config.json`.

No target-domain checkpoint is created.

## Post-training evaluation

Freeze the selected source checkpoint before target-domain evaluation.

### RSOC source-domain reference

```bash
cobicount-evaluate \
  --checkpoint runs/cobicount_rsoc_building/best_source_val.pth \
  --dataset rsoc \
  --rsoc-root /path/to/RSOC_building/building \
  --output-dir outputs/rsoc_reference
```

### DOTA full-image evaluation

```bash
cobicount-evaluate \
  --checkpoint runs/cobicount_rsoc_building/best_source_val.pth \
  --dataset dota-full \
  --dota-image-root /path/to/DOTA/images \
  --dota-annotation-root /path/to/DOTA/annotations \
  --class-filter "small vehicle" \
  --file-list /path/to/test_small_vehicle.txt \
  --max-side 1440 \
  --output-dir outputs/dota_small_vehicle
```

### DOTA positive-crop proxy evaluation

```bash
cobicount-evaluate \
  --checkpoint runs/cobicount_rsoc_building/best_source_val.pth \
  --dataset dota-proxy \
  --dota-image-root /path/to/DOTA/images \
  --dota-annotation-root /path/to/DOTA/annotations \
  --class-filter "ship" \
  --proxy-crops-per-image 4 \
  --output-dir outputs/dota_ship_proxy
```

The positive-crop proxy excludes images without selected-class instances by default. Its MAE/RMSE is therefore not directly comparable with full-image DOTA evaluation.

## Single-image inference

```bash
cobicount-infer \
  --checkpoint runs/cobicount_rsoc_building/best_source_val.pth \
  --image /path/to/image.png \
  --output-dir outputs/example
```

This exports:

- `prediction.json`;
- `score_map.png`;
- `diagnostic_points.png`.

## Reproducibility

The principal defaults are:

| Setting | Value |
|---|---:|
| Input/patch size | 512 × 512 |
| Epochs | 80 |
| Batch size | 6 |
| Optimizer | AdamW |
| Learning rate | 1.8 × 10⁻⁴ |
| Weight decay | 1 × 10⁻⁴ |
| EMA decay | 0.999 |
| Gradient clipping | 5.0 |
| Seed | 3407 |
| Score stride | 4 |

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for the source-only checklist, expected differences across hardware, and checkpoint-release guidance.

## Checkpoints

Model weights are intentionally not committed to Git history. Add released weights as GitHub Release assets or host them on an archival service, then update [`checkpoints/README.md`](checkpoints/README.md) with the download link and checksum.

Only load checkpoints from trusted sources because PyTorch checkpoints may contain serialized Python objects.

## Citation

Until the article receives its final bibliographic record, cite the software as:

```bibtex
@software{zheng2026cobicount,
  author  = {Junjing Zheng and Zhiyi Zhou},
  title   = {COBICount: Candidate-Origin Bias Isolation for Source-Only Remote Sensing Object Counting},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/OWNER/COBICount}
}
```

Update the repository URL and citation after creating the public repository and after the paper is published. GitHub will also render the root-level [`CITATION.cff`](CITATION.cff).

## License

The code is released under the [MIT License](LICENSE). Dataset licenses, pretrained weights derived from third-party datasets, and third-party implementations remain subject to their own terms.

## Responsible use and limitations

COBICount is research software for remote-sensing counting experiments. It is not validated for safety-critical deployment. Performance can degrade under large category, scale, density, sensor, and background shifts, particularly for very small dense objects. See [`MODEL_CARD.md`](MODEL_CARD.md).
