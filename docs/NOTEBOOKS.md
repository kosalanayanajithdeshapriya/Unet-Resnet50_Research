# Notebook guide

Notebooks live in [`../notebooks/`](../notebooks) and are numbered in the order they were built.
All of them set their root directory (`BASE_DIR` / `UNET_PROJECT_ROOT`) in the config cell to
`d:/Reasearch/Unet` — change it to your local clone path. Metrics and figures are written to `results/`.

Dependency chain: `01 → 02` (re-run with extra evaluation), `01 → 03` (same U-Net recipe, k-fold), `01 → 04` (exported U-Net feeds the pipeline).

## 01 — `01_unet_pipeline.ipynb`

The original end-to-end U-Net notebook.

- **Task 2:** COCO polygons → binary masks, merged into `processed/{train,valid,test}/{images,masks}` plus stage-mapping CSVs, with an overlay sanity check.
- **Task 3:** custom U-Net (`base_channels=32`, 7.77M parameters), BCE + Dice loss, flip/rotation augmentation, best checkpoint by validation Dice.
- **Task 4:** held-out test evaluation (overall and per growth stage), qualitative examples.
- **Task 5:** Leaf Pixel Fraction (LPF) extraction for all 982 images.
- **Needs:** `tomato final dataset/` (Roboflow COCO export).
- **Writes:** `checkpoints/unet_{best,final}.pth`; `results/training_curves.png`, `test_set_per_image_scores.csv`, `test_scores_by_stage.csv`, `lpf_results.csv`, `qualitative_examples/test_examples.png`.

## 02 — `02_unet_pipeline_v2.ipynb`

Same notebook as 01 plus two extra cells at the end: a pixel-level confusion matrix and precision/recall/F1 on the test set.

- **Needs:** the same inputs as 01.
- **Writes:** `results/confusion_matrix.png` (plus everything 01 writes).

## 03 — `03_unet_cross_validation.ipynb`

Long-running 5-fold study, in two parts.

- **Part A:** retrain the U-Net from scratch on 5 stratified folds of train+valid (822 images); score pixel metrics on each fold's held-out subset and on the fixed 160-image test set.
- **Part B:** run the crop-aligned segment → mask → classify pipeline for each fold's U-Net with a fixed masked ResNet50 classifier.
- **Needs:** `processed/` (from 01) and the masked ResNet50 checkpoint `checkpoints_masked/resnet50_masked_best.pt` from the sibling `Resnet50test` project.
- **Writes:** `checkpoints/cv_folds/unet_fold{0-4}.pth`; `results/cross_validation/unet_segmentation_cv_results.csv`, `fusion_pipeline_cv_results.csv`, `cv_summary.png`.

## 04 — `04_unet_resnet_fusion_pipeline.ipynb`

Full pipeline demo and evaluation with no retraining: the exported U-Net segments the photo, the background is blacked out, and the masked ResNet50 classifies the growth stage on the 160-image test set.

- **Needs:** `export/unet_tomato_final.pth`, `export/unet_lpf_extractor.py` (loaded by file path), the sibling project's `checkpoints_masked/resnet50_masked_best.pt` and `tomato final dataset/test`.
- **Writes:** `results/fusion_pipeline/full_pipeline_test_results.csv`, `full_pipeline_confusion_matrix.png`, `pipeline_walkthrough.png`.

## Scripts

| Script | Purpose |
|---|---|
| [`scripts/build_results_report.py`](../scripts/build_results_report.py) | Rebuilds `reports/segmentation_and_pipeline_results_report.{md,docx}` from the CSVs in `results/` (and DeepLabV3 CSVs from the sibling `Resnet50test` repo). |
| [`scripts/add_confusion_matrices.py`](../scripts/add_confusion_matrices.py) | Draws `results/pipeline_confusion_matrices.png` and inserts it into `reports/Segmentation_Comparison_Report_with_CM.docx`. Takes the source `.docx` path as its first argument. |
