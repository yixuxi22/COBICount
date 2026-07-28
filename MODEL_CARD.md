# COBICount model card

## Model description

COBICount is a point-supervised remote-sensing object counter designed for single-source generalization. It predicts a non-negative score map whose spatial sum is the continuous image-level count.

The score is factorized into Candidate Evidence, Component Acceptance, Bias Isolation, and an effective-valid mask. Diagnostic points are extracted after inference using deterministic peak selection and non-maximum suppression.

## Intended use

- academic research on remote-sensing object counting;
- analysis of source-only and single-source generalization;
- visualization of local counting responses and structured-background failures;
- controlled benchmarking on properly licensed datasets.

## Out-of-scope use

- safety-critical surveillance or autonomous decision making;
- individual identification or tracking;
- claims of calibrated target-category recognition from audit channels;
- deployment without domain-specific validation;
- use that violates dataset, privacy, or applicable legal requirements.

## Training data

The principal configuration uses RSOC Building as the only supervised source. The repository does not include training images, annotations, or pretrained weights.

## Limitations

Performance can degrade under large shifts in category, object support scale, density, sensor characteristics, and background layout. Very small dense targets can produce insufficient candidate evidence, while repeated roads, parking patterns, roof boundaries, and water/harbor structures can produce false responses.

The audit channels are auxiliary source-derived responses. They are not calibrated semantic probabilities and must not be used as target-category labels.

The number of diagnostic points is not the predicted count. The count is the score-map integral; point extraction is a separate diagnostic procedure.

## Evaluation

Use count MAE/RMSE together with spatial candidate diagnostics. Scalar count errors can hide cancellation between missed targets and false structural responses.
