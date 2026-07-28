# Dataset preparation

This repository does not redistribute RSOC, DOTA, DIOR, or any derived imagery/annotations. Obtain each dataset from its official provider and comply with its license and citation requirements.

## RSOC Building

The source loader expects image/annotation pairs under:

```text
ROOT/
├── train_data/
│   ├── images/
│   └── ground_truth/
└── test_data/
    ├── images/
    └── ground_truth/
```

Each MATLAB annotation file must contain a `center` array with point coordinates in `(x, y)` order. The loader accepts both `GT_<image-stem>.mat` and `<image-stem>.mat` names.

The original RSOC layout does not expose a dedicated validation directory. The training CLI defaults to `test_data` for source-domain checkpoint selection to remain compatible with the original research implementation. A newly designed experiment should instead create a disjoint source validation split and pass its directory name using `--val-split`.

## DOTA

DOTA is used only by the post-training evaluation command. Supported annotations are:

- standard text lines: `x1 y1 x2 y2 x3 y3 x4 y4 class difficulty`;
- JSON objects containing common polygon/point/bounding-box fields.

The evaluation point is the arithmetic mean of the polygon vertices. Box size and orientation are not supplied to the model.

`--class-filter` accepts comma-separated labels, for example:

```text
--class-filter "large vehicle"
--class-filter "small vehicle"
--class-filter "ship"
```

For reproducible subset evaluation, provide a fixed `--file-list` containing one image filename per line.

## Positive-crop proxy

`dota-proxy` samples deterministic target-containing crops and excludes empty images unless `--keep-empty` is set. This protocol intentionally changes the image distribution and should be treated as a diagnostic proxy, not as a full-image benchmark.

## Data privacy and repository hygiene

Do not commit:

- dataset images or annotations;
- local absolute paths;
- institutional credentials or access tokens;
- generated crops containing data that cannot legally be redistributed.

The root `.gitignore` excludes common dataset and run directories, but contributors remain responsible for verifying staged files before pushing.
