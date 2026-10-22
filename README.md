# Tomato Leaf Segmentation with U-Net — Branch 01 of the Growth-Stage Pipeline

Research project (2026) on classifying tomato plant photos into four growth stages
(**seeding, developing, flowering, fruiting**). This repository is the **U-Net segmentation branch**:
a from-scratch U-Net separates leaves from background, and the result feeds a masked ResNet50 classifier.

## Research question

Can a U-Net trained from scratch on 982 self-collected tomato images segment leaves well enough that the
sequential **segment → mask → classify** pipeline predicts growth stage on new, unannotated photos — and how
stable is that result across training splits (5-fold cross-validation)?

## Pipeline

```
photo ──► U-Net (leaf vs. background) ──► black out background ──► masked ResNet50 ──► growth stage
              │
              └──► Leaf Pixel Fraction (LPF) = leaf pixels / total pixels
```

## Key results

All numbers are computed from the files in [`results/`](results) (test set = 160 held-out images).

| Experiment | Result | Source |
|---|---|---|
| U-Net segmentation, fixed split — Dice / IoU (mean per image) | 0.757 / 0.665 | `results/test_set_per_image_scores.csv` |
| U-Net segmentation, fixed split — leaf F1 (mean per image) | 0.785 | `results/test_set_pixel_classification_metrics.csv` |
| U-Net segmentation, 5-fold CV — leaf F1 on fixed test set | 0.773 ± 0.008 | `results/cross_validation/unet_segmentation_cv_results.csv` |
| U-Net segmentation, 5-fold CV — pixel accuracy | 92.8% ± 0.5% | same |
| Full pipeline, fixed split — accuracy | 89.4% (143/160) | `results/fusion_pipeline/full_pipeline_test_results.csv` |
| Full pipeline, 5 fold-specific U-Nets — accuracy | 85.6% ± 2.4% | `results/cross_validation/fusion_pipeline_cv_results.csv` |
| Full pipeline, 5 fold-specific U-Nets — macro F1 | 0.844 ± 0.024 | same |

Per-stage results on the fixed split:

| Stage | Dice | IoU | Pipeline recall | n |
|---|---|---|---|---|
| seeding | 0.857 | 0.776 | 100.0% | 38 |
| developing | 0.774 | 0.687 | 95.1% | 41 |
| flowering | 0.729 | 0.631 | 85.0% | 60 |
| fruiting | 0.625 | 0.519 | 71.4% | 21 |

`fruiting` is the weakest stage (smallest class, leaves occluded by fruit); most pipeline errors are
flowering ↔ fruiting confusions. Mean LPF is 0.24 (developing), 0.19 (flowering), 0.14 (fruiting) and
0.03 (seeding) (`results/lpf_results.csv`).

± values are sample standard deviations across the 5 folds. The fold U-Nets share one fixed classifier and the
same 160-image test set, so the CV spread reflects segmenter variability only. Full tables, the DeepLabV3
comparison and caveats are in
[reports/segmentation_and_pipeline_results_report.md](reports/segmentation_and_pipeline_results_report.md).

## Repository layout

```
.
├── notebooks/    Training & evaluation notebooks, numbered in the order they were built
├── scripts/      Report / figure builders
├── results/      Metrics (CSV) and figures produced by the notebooks
├── reports/      Written results (md / docx / pdf)
├── export/       Standalone U-Net package: metadata.json, lpf_full_dataset.csv, unet_lpf_extractor.py
├── docs/         Per-notebook documentation (NOTEBOOKS.md)
└── requirements.txt
```

Not tracked in git (too large; regenerate with the notebooks): `checkpoints/`, `export/*.pth`,
`tomato final dataset/`, `processed/`.

## Notebooks

| # | Notebook | Trains | Purpose |
|---|---|---|---|
| 01 | [01_unet_pipeline.ipynb](notebooks/01_unet_pipeline.ipynb) | U-Net | COCO → masks, training, test evaluation, LPF extraction |
| 02 | [02_unet_pipeline_v2.ipynb](notebooks/02_unet_pipeline_v2.ipynb) | U-Net | Notebook 01 plus a pixel confusion matrix / precision-recall-F1 cell |
| 03 | [03_unet_cross_validation.ipynb](notebooks/03_unet_cross_validation.ipynb) | 5 × U-Net | Segmentation CV and fold-wise full-pipeline CV |
| 04 | [04_unet_resnet_fusion_pipeline.ipynb](notebooks/04_unet_resnet_fusion_pipeline.ipynb) | — | End-to-end U-Net → masked ResNet50 evaluation |

Details: [docs/NOTEBOOKS.md](docs/NOTEBOOKS.md).

## Reproducing

```bash
pip install -r requirements.txt
jupyter lab notebooks/
```

1. Place the Roboflow COCO export under `tomato final dataset/{train,valid,test}/{fruiting,flowering,developing,seeding}/`.
2. Each notebook sets its root in the config cell (`BASE_DIR` / `UNET_PROJECT_ROOT`, currently
   `d:/Reasearch/Unet`) — change it to your local clone path.
3. Run notebooks 01 → 04 in order. Outputs go to `results/`; checkpoints to `checkpoints/`.
4. Notebooks 03 (Part B) and 04 also need the masked ResNet50 checkpoint `checkpoints_masked/resnet50_masked_best.pt`
   and the test images from the sibling `Resnet50test` project (`d:/Reasearch/Resnet50test`).
5. Copy the final weights to `export/unet_tomato_final.pth` before running notebook 04.
6. Rebuild reports: `python scripts/build_results_report.py` (needs the sibling `Resnet50test` repo checked out next to this one).

### Using the exported model

```python
from unet_lpf_extractor import load_unet, compute_lpf
model = load_unet("export/unet_tomato_final.pth")
lpf, mask = compute_lpf(model, "path/to/image.jpg")
```

## License

See [LICENSE](LICENSE).
