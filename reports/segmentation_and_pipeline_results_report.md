# Tomato Leaf Segmentation and Growth-Stage Classification

Consolidated Results Report

Prepared: 30 September 2026

This document separates segmentation-model performance, complete sequential pipeline performance, and cross-validation results. It consolidates saved results from the U-Net and Resnet50test projects. No models were retrained or inference rerun for this report. Fold summaries and U-Net fixed-test classification metrics were recalculated from saved CSV files. DeepLabV3 fixed-split macro metrics were transcribed from saved notebook output. Source identifiers [S1–S10] are resolved in Section 8.

## 1. Study scope and experimental organization

The segmentation task labels pixels as leaf/plant foreground or background. The classification task assigns an image to developing, flowering, fruiting, or seeding. The sequential pipeline predicts a mask, applies it to the RGB image, and sends the masked image to a ResNet50 classifier. DeepLabV3–ResNet50 names the segmentation network and its backbone; the downstream ResNet50 classifier is a separate model. These results concern sequential masking, not learned concatenation of LPF and classifier features.

| Experiment | Evaluation design | What is measured |
| --- | --- | --- |
| Fixed-split segmentation | Saved trained segmenter; existing test split | Pixel overlap and pixel classification |
| Fixed-split pipeline | Saved segmenter and classifier; 160 test images | Four-stage image classification |
| U-Net five-fold study | 822 development images; five U-Nets; fixed 160-image test set | Segmenter variation and downstream variation with a fixed classifier |
| DeepLabV3 segmenter-only CV | 979 annotated images; five outer test folds | Generalization of the segmentation training procedure |
| DeepLabV3 full-pipeline CV | 979 annotated images; both models retrained per outer fold | Generalization of the sequential two-model training procedure |

The original U-Net split contains 687 training, 135 validation, and 160 test images (982 total). DeepLabV3 CV records contain 979 usable annotated images after excluding samples without usable annotations. The exact three-image difference and cross-project identity should be reconciled before describing the CV studies as matched experiments. [S1, S6, S7]

### 1.1 Metric conventions

IoU and Dice are reported on the 0–1 scale. Classification accuracy, macro precision, macro recall, and macro F1 are percentages. Macro classification metrics average the four stages equally. Segmentation macro metrics average background and foreground equally. Foreground IoU is TP/(TP+FP+FN); binary Dice is 2TP/(2TP+FP+FN). Soft Dice uses probabilities and is a different measurement. Binary Dice equals foreground F1 only when masks and aggregation are identical.

All fold summaries use an unweighted arithmetic mean and sample standard deviation (denominator K−1). A percentage metric’s SD is in percentage points. SD describes fold/run spread, not a confidence interval. DeepLabV3 CSV values were saved to four decimal places; derived summaries inherit this precision limit. Mean IoU cannot generally be converted into mean Dice after averaging.

## 2. Segmentation-model results: fixed-split evaluation

### 2.1 Saved comparison using binary masks

Table 1 reports the saved cross-model comparison artifact. Foreground IoU and foreground Dice use the same class definition within this table. Preprocessing variants remain separate. These are single-checkpoint results and have no fold-level SD. [S2]

| Segmentation model / preprocessing | Foreground IoU | Foreground Dice | Both-class mean IoU |
| --- | --- | --- | --- |
| DeepLabV3-ResNet50 | 0.6230 | 0.7565 | 0.7507 |
| U-Net (custom, from scratch) | 0.6646 | 0.7844 | 0.7926 |
| U-Net (crop-aligned, DeepLabV3-style preprocessing) | 0.6674 | 0.7827 | 0.7753 |

In this saved comparison, U-Net’s foreground overlap was higher than DeepLabV3’s. This is a descriptive checkpoint comparison, not evidence of a statistically significant or cross-validated advantage. Both-class mean IoU includes background and is not interchangeable with foreground IoU.

### 2.2 Original U-Net standalone evaluation

| Split | Images | Binary IoU | Soft Dice |
| --- | --- | --- | --- |
| Validation | 135 | 0.7241 | 0.8127 |
| Test | 160 | 0.6650 | 0.7572 |

These values come from the original export metadata and use direct image resizing. Dice here is probability-based soft Dice; it must not be substituted for the binary Dice in Table 1. The difference between test soft Dice 0.7572 and binary Dice 0.7844 is therefore not a transcription error. [S1, S10]

| Test growth stage | Images | Binary IoU | Soft Dice |
| --- | --- | --- | --- |
| Developing | 41 | 0.6868 | 0.7736 |
| Flowering | 60 | 0.6313 | 0.7289 |
| Fruiting | 21 | 0.5186 | 0.6250 |
| Seeding | 38 | 0.7756 | 0.8575 |

Fruiting had the lowest overlap in this evaluation; seeding had the highest. The saved results identify the performance pattern but do not establish its cause.

### 2.3 Additional DeepLabV3 segmentation evaluations

The fixed-split training notebook records test both-class mean IoU of 0.7548. A later saved fixed-split metrics cell records pixel accuracy 0.9010, macro precision 0.8466, macro recall 0.8626, and macro F1 0.8542; foreground F1/binary Dice is 0.7715. These are separate saved evaluations from the comparison artifact’s foreground Dice of 0.7565 and both-class IoU of 0.7507. They are retained here for provenance, not assembled into a synthetic result row. [S7, cells 17, 24 and 34; S2]

## 3. Full sequential pipeline results: fixed-split evaluation

### 3.1 Main fixed-split classification results

Each row below describes its own saved evaluation of the segment → mask → classify sequence. The U-Net row is recalculated from 160 per-image predictions. The DeepLabV3 row uses the later saved standalone recomputation and its matching macro-metric output. [S3, S7 cell 34, S8 cell 12]

| Pipeline | Accuracy (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) |
| --- | --- | --- | --- | --- |
| U-Net → ResNet50 | 89.38 | 87.44 | 87.89 | 87.65 |
| DeepLabV3 → ResNet50 (recomputed) | 87.50 | 85.46 | 88.57 | 86.12 |

U-Net correctly classified 143 of 160 images; DeepLabV3’s recomputed accuracy corresponds to 140 of 160. These evaluations reuse separately trained checkpoints. They do not establish a controlled architecture comparison or statistical superiority.

### 3.2 Stage-wise fixed-split classification

| Pipeline | Stage | Precision (%) | Recall (%) | F1 (%) | Test images |
| --- | --- | --- | --- | --- | --- |
| U-Net | Developing | 95.12 | 95.12 | 95.12 | 41 |
| U-Net | Flowering | 86.44 | 85.00 | 85.71 | 60 |
| U-Net | Fruiting | 68.18 | 71.43 | 69.77 | 21 |
| U-Net | Seeding | 100.00 | 100.00 | 100.00 | 38 |
| DeepLabV3 | Developing | 95.35 | 100.00 | 97.62 | 41 |
| DeepLabV3 | Flowering | 91.67 | 73.33 | 81.48 | 60 |
| DeepLabV3 | Fruiting | 54.84 | 80.95 | 65.38 | 21 |
| DeepLabV3 | Seeding | 100.00 | 100.00 | 100.00 | 38 |

Both fixed-split evaluations classified all 38 seeding images correctly. Fruiting had the lowest class F1. This describes the observed test set, not guaranteed deployment performance.

### 3.3 U-Net fixed-split confusion matrix

| True / predicted | Developing | Flowering | Fruiting | Seeding |
| --- | --- | --- | --- | --- |
| Developing | 39 | 2 | 0 | 0 |
| Flowering | 2 | 51 | 7 | 0 |
| Fruiting | 0 | 6 | 15 | 0 |
| Seeding | 0 | 0 | 0 | 38 |

Rows are true labels and columns are predictions. Thirteen of the 17 U-Net pipeline errors were exchanges between flowering and fruiting; four were exchanges between developing and flowering. [S3]

### 3.4 Earlier preprocessing comparison: separate result series

| Saved comparison variant | Pipeline accuracy (%) |
| --- | --- |
| DeepLabV3-ResNet50 | 84.38 |
| U-Net (custom, from scratch) | 78.75 |
| U-Net (crop-aligned, DeepLabV3-style preprocessing) | 89.38 |

This comparison records DeepLabV3 accuracy 84.38%, native-preprocessing U-Net accuracy 78.75%, and crop-aligned U-Net accuracy 89.38%. Do not attach the later DeepLabV3 macro metrics for the 87.50% run to the earlier 84.38% run. The crop-aligned and native U-Net results show that pipeline preprocessing matters alongside segmentation overlap. The comparison CSV stores rounded accuracies; the 89.38% value in Section 3.1 is independently calculated from exact predictions. [S2, S3]

## 4. Cross-validation: segmentation models

### 4.1 U-Net training and evaluation protocol

Five growth-stage-stratified folds were formed from the 822 training-plus-validation images (seed 42). Each fresh U-Net trained for 30 epochs. Folds 1–2 used 657 training and 165 validation images; folds 3–5 used 658 and 164. The best validation-soft-Dice checkpoint was selected per fold. Each checkpoint was then evaluated on the fixed 160-image test set. The baseline used 60 epochs and a learning-rate scheduler, whereas the CV loop used 30 epochs without that scheduler. [S4, S9]

The following foreground IoU values are derived per fold from saved pooled leaf F1 as IoU = F1/(2−F1), then summarized. They were not directly saved as IoU measurements. Binary Dice is the saved leaf F1 because both refer to the same pooled thresholded pixels. These values differ in aggregation from the per-image fixed-split metrics in Section 2.

| Fold | Test foreground IoU (derived) | Test foreground Dice/F1 | Test pixel accuracy (%) |
| --- | --- | --- | --- |
| 1 | 0.6394 | 0.7801 | 93.07 |
| 2 | 0.6225 | 0.7673 | 92.68 |
| 3 | 0.6189 | 0.7646 | 91.99 |
| 4 | 0.6273 | 0.7710 | 92.77 |
| 5 | 0.6438 | 0.7833 | 93.47 |
| Mean ± SD | 0.6304 ± 0.0108 | 0.7732 ± 0.0081 | 92.80 ± 0.54 |

| U-Net segmentation metric | Fold-validation subsets: mean ± SD | Fixed test: mean ± SD |
| --- | --- | --- |
| Foreground precision | 0.7453 ± 0.0341 | 0.7288 ± 0.0335 |
| Foreground recall | 0.8278 ± 0.0263 | 0.8258 ± 0.0302 |
| Foreground Dice/F1 | 0.7835 ± 0.0126 | 0.7732 ± 0.0081 |
| Pixel accuracy | 0.9454 ± 0.0060 | 0.9280 ± 0.0054 |

Fold-validation subsets also selected the checkpoint, so their scores are model-selection measurements. Fixed-test results measure variation among five trained segmenters on the same images; they are not five disjoint outer test folds.

### 4.2 DeepLabV3 segmenter-only five-fold CV

This separate experiment used ordinary, non-stratified KFold on 979 annotated images, with 10 epochs per fold and a 15% inner validation split. Folds 1–4 used 665/118/196 training/validation/test images; fold 5 used 666/118/195. Test IoU averages background and plant IoU within a batch and then averages batches equally. Pixel precision, recall, and F1 are macro averages over the two classes. [S5, S7 Section 9]

| Fold | Both-class IoU | Pixel accuracy (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.7565 | 91.77 | 85.07 | 86.20 | 85.62 |
| 2 | 0.7390 | 91.19 | 86.00 | 83.85 | 84.86 |
| 3 | 0.7390 | 90.39 | 85.07 | 83.15 | 84.06 |
| 4 | 0.7341 | 91.30 | 84.72 | 83.17 | 83.91 |
| 5 | 0.7645 | 91.30 | 85.49 | 86.09 | 85.78 |
| Mean ± SD | 0.7466 ± 0.0131 | 91.19 ± 0.50 | 85.27 ± 0.49 | 84.49 ± 1.54 | 84.85 ± 0.86 |

Foreground-only Dice was not saved in this CV CSV. The recorded macro F1 is the mean of background and foreground F1, not foreground Dice. It must not be relabeled as leaf Dice or derived from the averaged IoU.

### 4.3 DeepLabV3 segmentation within full-pipeline CV

The full-pipeline study uses a different set of stratified fold runs. Both the segmenter and classifier were retrained; segmentation results from those runs are reported separately below. [S6]

| Metric | Mean ± SD |
| --- | --- |
| Both-class, batch-averaged IoU | 0.7489 ± 0.0037 |
| Pixel accuracy (%) | 91.11 ± 0.26 |
| Macro pixel precision (%) | 85.44 ± 1.10 |
| Macro pixel recall (%) | 83.85 ± 1.08 |
| Macro pixel F1 (%) | 84.57 ± 0.18 |

The 0.7466 segmenter-only CV IoU and 0.7489 full-pipeline CV segmentation IoU are both recorded results from different runs. Neither is directly comparable with U-Net’s derived pooled foreground IoU without aligning the class definition, aggregation, data splits, and training protocol.

## 5. Cross-validation: full sequential pipeline results

### 5.1 Protocol differences

U-Net: each of the five fold segmenters was paired with the same unchanged masked-image ResNet50 and evaluated on the same 160 test images. This is downstream robustness under segmenter retraining, not full two-model cross-validation. [S9]

DeepLabV3: five stratified outer folds were formed from all 979 usable annotated images, including images from the original test partition. Each fold reserved 15% of its outer training partition for inner validation. A fresh pretrained segmenter trained for 10 epochs and a fresh pretrained classifier trained for 12 epochs. The classifier trained and validated on ground-truth-masked images; the outer test pipeline used predicted masks. Both models were selected using inner validation. This is full sequential training-procedure CV, although the networks were trained separately rather than jointly. [S6, S7 Sections 10]

### 5.2 Pipeline summary table

| Pipeline | Accuracy (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) |
| --- | --- | --- | --- | --- |
| U-Net → fixed ResNet50 | 85.62 ± 2.38 | 84.00 ± 2.39 | 85.89 ± 2.11 | 84.41 ± 2.37 |
| DeepLabV3 → ResNet50; both retrained | 82.23 ± 2.16 | 77.12 ± 6.74 | 75.80 ± 5.27 | 74.08 ± 4.30 |

Table note: values are mean ± sample SD across five runs/folds. Evaluation protocols differ; this table is a descriptive summary and does not establish a matched CV ranking. Segmentation IoU is intentionally reported separately in Section 4.

### 5.3 U-Net pipeline results by fold

| Fold | Accuracy (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) |
| --- | --- | --- | --- | --- |
| 1 | 85.00 | 83.34 | 85.36 | 83.86 |
| 2 | 86.25 | 85.45 | 85.61 | 85.25 |
| 3 | 83.12 | 81.61 | 85.27 | 82.28 |
| 4 | 89.38 | 87.39 | 89.44 | 88.09 |
| 5 | 84.38 | 82.20 | 83.78 | 82.54 |

U-Net pipeline accuracy ranged from 83.13% to 89.38%. The five evaluations contain 160 unique test images, not 800 independent test cases. The mean accuracy is lower than the fixed-split baseline, but the different training budgets and training subsets prevent attributing the difference solely to cross-validation.

### 5.4 DeepLabV3 full-pipeline results by fold

| Fold | Test images | Accuracy (%) | Macro precision (%) | Macro recall (%) | Macro F1 (%) |
| --- | --- | --- | --- | --- | --- |
| 1 | 196 | 83.16 | 78.87 | 83.20 | 79.16 |
| 2 | 196 | 83.16 | 76.28 | 78.93 | 76.90 |
| 3 | 196 | 82.14 | 76.32 | 72.50 | 71.88 |
| 4 | 196 | 78.57 | 67.65 | 70.00 | 68.13 |
| 5 | 195 | 84.10 | 86.48 | 74.38 | 74.33 |

DeepLabV3 full-pipeline accuracy ranged from 78.57% to 84.10%. The 979 images are distributed across outer test folds, unlike U-Net’s repeated fixed-test evaluations. Fold-level per-class pipeline metrics and per-image predictions are not contained in the saved DeepLabV3 CV summary CSV.

### 5.5 U-Net pipeline stage-wise fold summary

| Stage | Precision (%) | Recall (%) | F1 (%) |
| --- | --- | --- | --- |
| Developing | 94.74 ± 4.64 | 91.71 ± 5.34 | 93.05 ± 3.15 |
| Flowering | 84.55 ± 3.32 | 75.67 ± 7.32 | 79.66 ± 4.15 |
| Fruiting | 56.71 ± 6.68 | 76.19 ± 5.83 | 64.91 ± 5.86 |
| Seeding | 100.00 ± 0.00 | 100.00 ± 0.00 | 100.00 ± 0.00 |

These class-wise values are available for U-Net’s fixed-classifier evaluations only. They should not be paired with DeepLabV3’s single-run class metrics as if both were cross-validated class results.

## 6. Interpretation and limits

Segmentation and classification measure different outcomes. U-Net’s stronger foreground overlap in the saved comparison did not automatically yield stronger classification with native preprocessing: the native pipeline scored 78.75%, whereas the crop-aligned variant scored 89.38%. Thus, the saved experiments support evaluating input framing and mask/classifier compatibility in addition to overlap metrics.

The fixed-split pipeline records show 89.38% accuracy for the main U-Net run and 87.50% for the later DeepLabV3 recomputation. Earlier DeepLabV3 comparison results show 84.38%. These are distinct saved evaluations; selecting one score without its protocol would conceal run differences. The report preserves them rather than treating them as duplicates.

The CV studies answer different questions. U-Net quantifies segmenter and downstream variation with a fixed classifier. DeepLabV3 full-pipeline CV quantifies the complete training procedure with both models retrained. Their numerical difference cannot establish a cross-validated architecture advantage because the splits, test exposure, training schedules, and retraining scope differ.

Image-level splits have not been established here as independent by plant, collection session, or near-duplicate group. The original test set was pooled into DeepLabV3 CV and is therefore not an untouched external test for that study. Repeated U-Net fixed-test scoring must not be used for test-guided model selection while claiming untouched evaluation. Cross-project data separation, preprocessing implementation, and duplicate handling require a dedicated audit before stronger generalization claims.

Some saved evaluations differ in resize/crop geometry, soft versus binary Dice, foreground versus both-class metrics, and per-image versus pooled-pixel or batch aggregation. The earlier U-Net report also identifies a mask-resizing concern in the pipeline notebook. These differences should be resolved by a common evaluation implementation before a definitive segmentation comparison. No significance test or external validation is claimed here.

## 7. Conclusions and next reporting steps

Both U-Net and DeepLabV3 have saved segmentation IoU and Dice results in fixed-split evaluations. Both sequential pipelines have saved growth-stage classification results. U-Net has five-fold segmenter training with fixed-classifier downstream evaluation; DeepLabV3 has both segmenter-only CV and full-pipeline CV with both networks retrained.

For presentation, use Section 2 for standalone segmentation, Section 3 for fixed-split pipelines, Section 4 for segmentation CV, and Section 5 for pipeline CV. Keep the protocol note below the pipeline CV table. Do not mix a fixed-split IoU with a cross-validated pipeline score in a row implying one experiment.

To obtain a matched comparison, use identical eligible images and outer folds, harmonize image geometry and overlap definitions, retrain both pipeline models within each fold, select checkpoints only on inner validation, and retain per-image predictions and pixel confusion counts. This is future work, not a completed experiment. DeepLabV3 foreground Dice per CV fold requires additional saved confusion data or reevaluation; it is not recoverable from the current macro summary alone.

## 8. Evidence index and reproducibility

S1. Original U-Net metadata: D:/Reasearch/Unet/export/metadata.json

S2. Fixed-split cross-model comparison: D:/Reasearch/Resnet50test/unet_vs_deeplabv3_comparison_v2.csv

S3. U-Net fixed-split pipeline predictions: D:/Reasearch/Unet/outputs/fusion_pipeline/full_pipeline_test_results.csv

S4. U-Net segmentation fold metrics: D:/Reasearch/Unet/outputs/cross_validation/unet_segmentation_cv_results.csv

S5. DeepLabV3 segmenter-only CV: D:/Reasearch/Resnet50test/checkpoints_segmentation/deeplabv3_kfold_cv_results.csv

S6. DeepLabV3 full-pipeline CV: D:/Reasearch/Resnet50test/deeplabv3_pipeline_kfold_cv_results.csv

S7. DeepLabV3 training and CV notebook: D:/Reasearch/Resnet50test/leaf_segmentation.ipynb

S8. DeepLabV3 standalone pipeline recomputation: D:/Reasearch/Resnet50test/deeplabv3_pipeline_evaluation.ipynb

S9. U-Net CV notebook: D:/Reasearch/Unet/unet_cross_validation.ipynb

S10. Earlier U-Net methods/results audit: D:/Reasearch/Unet/reports/research_workload_report.md

S11. U-Net pipeline fold metrics: D:/Reasearch/Unet/outputs/cross_validation/fusion_pipeline_cv_results.csv

Notebook cell numbers in this report are zero-based indices in the saved notebook JSON. Section 5 U-Net numerical tables use S11. The companion build_results_report.py regenerates the CSV-based calculations and both document formats. Some fixed-split values are explicitly transcribed from the identified saved notebook outputs. No external literature claims are required for this results-only report.
