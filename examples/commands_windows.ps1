$RSOC_ROOT = "F:\path\to\RSOC_building\building"
$DOTA_IMAGE_ROOT = "F:\path\to\DOTA\images"
$DOTA_ANNOTATION_ROOT = "F:\path\to\DOTA\annotations"

cobicount-train `
  --rsoc-root $RSOC_ROOT `
  --output-dir "runs\cobicount_rsoc_building" `
  --epochs 80 `
  --batch-size 6 `
  --device cuda

cobicount-evaluate `
  --checkpoint "runs\cobicount_rsoc_building\best_source_val.pth" `
  --dataset dota-full `
  --dota-image-root $DOTA_IMAGE_ROOT `
  --dota-annotation-root $DOTA_ANNOTATION_ROOT `
  --class-filter "ship" `
  --output-dir "outputs\dota_ship"
