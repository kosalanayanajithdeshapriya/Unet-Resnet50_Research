# Tomato Growth-Stage Classification Using U-Net Leaf Segmentation and a Fixed ResNet50 Classifier

## Research workload report and manuscript drafting guide

Prepared: 15 September 2026. Evidence: notebook source, saved notebook outputs, exported metadata, processed-data counts, checkpoint presence, and result CSVs in this workspace. Training and inference were not rerun for this report. Numerical summaries were checked from the saved CSVs. External references support architecture background and one preprocessing detail; they do not validate this project's results.

**Central finding:** the project has completed baseline U-Net training, LPF extraction, a sequential U-Net-to-ResNet50 inference pipeline, and five-fold U-Net cross-validation with downstream evaluation through a fixed ResNet50. It has not completed cross-validation that retrains both models within every fold, and it has not implemented LPF–ResNet feature concatenation in the four reviewed notebooks.

---

## 1. Suggested paper title and scope

**Suggested title:** *Tomato Growth-Stage Classification with U-Net Leaf Segmentation: Five-Fold Segmenter Evaluation Using a Fixed ResNet50 Classifier*

The study is an application and evaluation of existing model families. Do not claim a new neural-network architecture or state-of-the-art performance from the available experiments. The strongest supported contributions are:

1. A leaf-mask preparation, segmentation-training, evaluation, and LPF-export workflow for a 982-image tomato dataset.
2. Integration of a trained segmenter and a separately trained four-class classifier through a shared image coordinate frame.
3. Evaluation of segmentation quality and downstream classification variability across five independently initialized U-Net fold models.
4. Class-specific analysis identifying fruiting and flowering as the principal classification challenge in the evaluated test set.

## 2. Draft abstract

This study evaluates a sequential leaf-segmentation and growth-stage classification workflow for tomato images. A dataset of 982 annotated images spanning developing, flowering, fruiting, and seeding stages was used. A custom U-Net was trained from scratch to predict binary leaf masks, which were applied to RGB images before classification by a separately trained ResNet50. A baseline U-Net was trained for 60 epochs using 687 training images and 135 validation images. The resulting pipeline correctly classified 143 of 160 test images, achieving 89.38% accuracy. To investigate sensitivity to segmenter training data, five-fold stratified cross-validation was performed on the combined 822-image training and validation pool, with each U-Net trained for 30 epochs. Each fold model was evaluated on its validation subset and the fixed test set, and was subsequently paired with the same unchanged ResNet50 classifier. Mean fixed-test leaf-pixel F1 was 0.7732 ± 0.0081, while downstream classification accuracy and macro F1 were 85.63% ± 2.38 percentage points and 0.8441 ± 0.0237, respectively. Seeding achieved an F1 of 1.0000 across all five pipeline evaluations, whereas fruiting achieved a mean F1 of 0.6491. These results characterize downstream variability under U-Net retraining. They do not establish full pipeline cross-validation, because classifier training was held fixed. Further evaluation should retrain both stages within each fold, verify data separation across projects, and assess generalization on independently collected images.

**Keywords:** tomato growth stage; leaf segmentation; U-Net; ResNet50; stratified cross-validation; leaf pixel fraction.

## 3. Introduction and research questions

The project investigates whether leaf-focused image representations can support classification of tomato growth stages. Its workflow predicts leaf regions, suppresses the remaining image pixels, and classifies the masked image. This makes segmentation an intermediate part of the classification process as well as a separately evaluated task.

U-Net combines a contracting path with an expanding path and connections between corresponding resolutions, providing the basis for the project's segmentation implementation. The implementation here is a custom variant, not an exact reproduction of all details of the original network. [Ronneberger et al., 2015](https://arxiv.org/abs/1505.04597).

ResNet uses residual learning for image recognition and provides the architecture family for the downstream classifier. The project's inference code instantiates ResNet50 and replaces its output layer with a four-class linear layer. [He et al., 2015](https://arxiv.org/abs/1512.03385).

The completed experiments address three questions:

- **RQ1:** How accurately does the baseline U-Net identify leaf pixels on the existing test set?
- **RQ2:** What growth-stage classification performance is obtained when its predicted mask is passed to a fixed ResNet50 through a masked RGB image?
- **RQ3:** How much do segmentation and downstream classification results vary when U-Net is retrained across five stratified folds?

They do not yet answer whether masking improves over a matched RGB-only classifier, whether LPF adds information to ResNet features, or how well the full training procedure generalizes to a new site or collection period.

**Literature-review boundary:** the two architecture references above are starting references, not a systematic review of agricultural classification research. A submission still needs a focused review of tomato growth-stage classification, plant segmentation, and comparable evaluation protocols. No superiority claims against that literature are made here.

## 4. Dataset and annotation preparation

### 4.1 Dataset composition

The repository describes 982 self-collected tomato images annotated using Roboflow and exported in COCO format. Counts below were verified against `export/lpf_full_dataset.csv`; processed image and mask counts also match the split totals.

| Growth stage | Training | Validation | Test | Total |
|---|---:|---:|---:|---:|
| Developing | 142 | 20 | 41 | 203 |
| Flowering | 210 | 30 | 60 | 300 |
| Fruiting | 71 | 9 | 21 | 101 |
| Seeding | 264 | 76 | 38 | 378 |
| **Total** | **687** | **135** | **160** | **982** |

The original split is approximately 70.0% training, 13.7% validation, and 16.3% test. Class frequencies are unequal. Fruiting is the smallest class overall and in the test set. The repository label is `seeding`; the author should confirm whether the intended scientific term is “seedling stage” before standardizing manuscript terminology.

The full pipeline's saved output reports 160 test samples with annotations and zero skipped test samples.

### 4.2 Mask conversion

The preprocessing code reads COCO image and annotation records, gathers the polygons belonging to each image, rounds polygon coordinates, and fills them into a binary foreground mask. All stage categories are merged into a single leaf foreground class. Thus, segmentation is binary even though image classification has four classes.

Processed masks are stored as PNG files using 0 for background and 255 for leaves. Stage mappings are saved separately. The notebook provides overlays for visual inspection of image-mask alignment. Presence of this inspection code is not a substitute for documenting an annotation quality-control procedure.

### 4.3 Dataset information still needed for a paper

The reviewed files do not establish camera specifications, collection dates and locations, cultivar, number of distinct plants, image acquisition distance, growth-stage labeling rules, annotator counts, inter-annotator agreement, or the independence of images from the same plant/session. These details must come from actual collection records. Do not invent them.

## 5. Model and preprocessing methods

### 5.1 U-Net

| Component | Implemented configuration |
|---|---|
| Input | 224 × 224 RGB image |
| Encoder block outputs | 32, 64, 128, 256 channels |
| Bottleneck output | 512 channels |
| Downsampling | 2 × 2 max pooling |
| Convolution block | Two repetitions of 3 × 3 convolution, batch normalization, ReLU |
| Upsampling | 2 × 2 transposed convolution, stride 2 |
| Skip connections | Concatenation of encoder and decoder features |
| Output | 1 × 1 convolution producing one-channel logits |
| Probability conversion | Sigmoid outside the model |
| Binary threshold | Probability > 0.5 |
| Initialization | Training from scratch; no pretrained encoder |
| Parameter count | 7,765,985, as recorded when the exported model was loaded |

The bottleneck maps 256 channels to 512. Export metadata incorrectly describes a 512-to-1024 bottleneck and lists 7,771,891 parameters. Use the implemented architecture and verified count when reconciling documentation.

### 5.2 Standalone segmentation preprocessing

Images are loaded with OpenCV, converted from BGR to RGB, and resized directly to 224 × 224 using linear interpolation. Masks are resized using nearest-neighbor interpolation and thresholded into binary targets. Image values are scaled to [0, 1] and normalized using channel means [0.485, 0.456, 0.406] and standard deviations [0.229, 0.224, 0.225].

Training augmentation applies horizontal flipping with probability 0.5, vertical flipping with probability 0.5, and a rotation selected from 90°, 180°, and 270° with probability 0.5. The same transformation is applied to image and mask. Validation and test datasets are not augmented.

### 5.3 Loss function

The loss is:

`L = 0.5 × BCEWithLogitsLoss + 0.5 × (1 − soft Dice)`

For predicted probabilities p_i and binary targets y_i within one image:

`soft Dice = (2 Σ_i p_i y_i + ε) / (Σ_i p_i + Σ_i y_i + ε)`

The implementation uses ε = 10^-6 and averages the per-image soft Dice within a batch. This probability-based Dice differs from Dice calculated after thresholding the mask.

### 5.4 ResNet50 and pipeline inference

The classifier is loaded from the sibling project's `checkpoints_masked/resnet50_masked_best.pt`. The local notebook describes it as trained on ground-truth-masked images. Its class order is developing, flowering, fruiting, and seeding. It is not trained in the reviewed notebooks.

The actual pipeline is:

```text
RGB image → shared resize and center crop → U-Net → predicted binary leaf mask
                          │                              │
                          └──────── pixel multiplication ┘
                                         ↓
                               masked RGB image
                                         ↓
                          normalization → fixed ResNet50
                                         ↓
                              four growth-stage scores
```

The code uses `Resize(int(224 × 1.14))`, which evaluates to **255**, followed by `CenterCrop(224)`. The notebook prose describes this approximately as resizing to 256; the exact implemented short-edge size is 255. An integer Resize size preserves aspect ratio by setting the shorter edge to that size. [Torchvision 0.21 Resize documentation](https://docs.pytorch.org/vision/0.21/generated/torchvision.transforms.Resize.html).

U-Net receives the normalized crop. Its predicted binary mask is multiplied into the same crop in unnormalized [0, 1] RGB space. The masked image is then normalized for ResNet50. Softmax converts classifier logits into class scores, and the highest-scoring class is returned. These scores have not been evaluated for probability calibration.

This is a sequential segmentation–classification pipeline. There is no ResNet backbone inside this U-Net, no joint optimization of both models, and no LPF feature concatenation in the reviewed pipeline.

### 5.5 Leaf Pixel Fraction

`LPF = number of predicted leaf pixels / total number of image pixels`

LPF was exported for all 982 images using the standalone direct-resize preprocessing. It represents projected image occupancy, not physical leaf area, biomass, or a validated continuous growth measurement. Camera distance, framing, and occlusion can influence it.

## 6. Experimental design

### 6.1 Experiment A: baseline segmentation

| Setting | Value |
|---|---|
| Training / validation / test | 687 / 135 / 160 images |
| Epochs | 60 |
| Batch size | 8 |
| Optimizer | Adam |
| Initial learning rate | 0.0001 |
| Learning-rate scheduler | ReduceLROnPlateau on validation Dice; factor 0.5, patience 5 |
| Best checkpoint | Epoch 58, selected using validation soft Dice |
| Mixed precision | CUDA autocast and gradient scaling |
| DataLoader workers | 0 |

The saved training output records best validation Dice of approximately 0.8132. The export metadata separately records recomputed validation Dice of 0.8127 and IoU of 0.7241. Report their provenance rather than treating them as identical measurements. This review did not rerun evaluation to reconcile the small difference.

### 6.2 Experiment B: baseline combined pipeline

The exported baseline U-Net and fixed masked-image ResNet50 are loaded without retraining. The full prediction path is run on the 160 test images. Outputs include per-image predictions, a four-class confusion matrix, and a visual walkthrough.

### 6.3 Experiment C: five-fold U-Net cross-validation

The original training and validation images are merged into an 822-image pool. `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` assigns folds according to growth-stage labels.

| Fold, shown as 1–5 | Training images | Validation images | Validation developing / flowering / fruiting / seeding |
|---|---:|---:|---|
| 1 | 657 | 165 | 33 / 48 / 16 / 68 |
| 2 | 657 | 165 | 33 / 48 / 16 / 68 |
| 3 | 658 | 164 | 32 / 48 / 16 / 68 |
| 4 | 658 | 164 | 32 / 48 / 16 / 68 |
| 5 | 658 | 164 | 32 / 48 / 16 / 68 |

Each fold creates a fresh U-Net, trains it for 30 epochs with Adam at 0.0001, and saves the checkpoint with highest fold-validation soft Dice. Batch size, loss, normalization, and augmentation follow the baseline. **The CV training loop does not use the baseline's learning-rate scheduler.** Therefore training budget, training-set size, and scheduler differ from the baseline.

The five fold checkpoints exist under `checkpoints/cv_folds/`. Filenames and CSV fold IDs are zero-based (`fold0` through `fold4`); this report presents them as folds 1–5.

Each best fold model is scored on its validation subset and on the fixed 160-image test set. Because the validation subset is also used for epoch selection, those validation scores are model-selection results rather than a completely untouched outer-fold evaluation.

The configuration comments refer to an earlier eight-epoch trial as undertrained. The current saved result tables represent the 30-epoch runs. The earlier trial is not treated as an independently verified quantitative experiment here.

### 6.4 Experiment D: fixed-classifier robustness across U-Net folds

Each fold's U-Net is connected to exactly the same ResNet50 and evaluated on exactly the same test images. Only the segmenter changes. This assesses the downstream effect of segmenter training variability, with variation also including random initialization and stochastic optimization.

It is not five disjoint test folds for the complete classifier pipeline. There are 160 unique test images, not 800 independent test images. The five accuracies describe repeated model evaluations on a common test set.

## 7. Evaluation metrics and reporting conventions

For a binary target class:

- `Precision = TP / (TP + FP)`
- `Recall = TP / (TP + FN)`
- `F1 = 2TP / (2TP + FP + FN)`
- `IoU = TP / (TP + FP + FN)`
- `Pixel accuracy = (TP + TN) / (TP + TN + FP + FN)`

For growth-stage classification, accuracy is the fraction of images assigned the correct stage. Macro precision, recall, and F1 are unweighted means over the four classes. Class-wise recall equals the fraction correctly classified within each true stage.

**Metric distinctions that must remain explicit:**

| Evaluation | Dice/F1 calculation | Image frame | Aggregation |
|---|---|---|---|
| Baseline U-Net Dice | Soft probabilities | Direct 224 × 224 resize | Mean over images at test evaluation |
| Baseline U-Net IoU | Thresholded mask | Direct resize | Mean over images |
| Pipeline segmentation Dice/IoU | Thresholded mask | Shared resize and center crop | Mean over images |
| CV leaf F1 | Thresholded mask | Direct resize | All pixels pooled within each evaluated subset |

Binary Dice and F1 are equivalent when calculated on the same thresholded predictions with the same aggregation. That equivalence does not make the different rows above interchangeable.

All fold summaries use the arithmetic mean and **sample standard deviation**, with denominator K − 1. Standard deviation measures spread across these five runs; it is not a confidence interval. Pixel-level sample counts must not be treated as independent biological observations for uncertainty estimation.

## 8. Results

### 8.1 Baseline segmentation

| Evaluation | Soft Dice | Binary IoU |
|---|---:|---:|
| Validation, export metadata recomputation | 0.8127 | 0.7241 |
| Test, 160 images | 0.7572 | 0.6650 |

| Test stage | Images | Mean soft Dice | Mean binary IoU |
|---|---:|---:|---:|
| Developing | 41 | 0.7736 | 0.6868 |
| Flowering | 60 | 0.7289 | 0.6313 |
| Fruiting | 21 | 0.6250 | 0.5186 |
| Seeding | 38 | 0.8575 | 0.7756 |

Source: `outputs/test_set_per_image_scores.csv`, `outputs/test_scores_by_stage.csv`, and `export/metadata.json`.

The pipeline notebook separately records cropped-frame binary Dice of 0.7843 and IoU of 0.6698. Treat these as a separate evaluation protocol. They also carry the mask-resizing concern described in Section 10.

### 8.2 Baseline growth-stage classification

The combined baseline pipeline achieved **143/160 correct predictions: 89.375% accuracy**, rounded to 89.38%.

| Stage | Precision | Recall | F1 | Test images |
|---|---:|---:|---:|---:|
| Developing | 0.9512 | 0.9512 | 0.9512 | 41 |
| Flowering | 0.8644 | 0.8500 | 0.8571 | 60 |
| Fruiting | 0.6818 | 0.7143 | 0.6977 | 21 |
| Seeding | 1.0000 | 1.0000 | 1.0000 | 38 |

Confusion matrix reconstructed from the saved per-image CSV; rows are true labels and columns are predictions:

| True / Predicted | Developing | Flowering | Fruiting | Seeding |
|---|---:|---:|---:|---:|
| Developing | 39 | 2 | 0 | 0 |
| Flowering | 2 | 51 | 7 | 0 |
| Fruiting | 0 | 6 | 15 | 0 |
| Seeding | 0 | 0 | 0 | 38 |

Thirteen of the 17 errors were exchanges between flowering and fruiting. The other four were exchanges between developing and flowering. This identifies the main observed confusion pattern; it does not establish its biological or visual cause.

Source: `outputs/fusion_pipeline/full_pipeline_test_results.csv`.

### 8.3 U-Net cross-validation: fold-validation subsets

| Metric | Mean ± sample SD |
|---|---:|
| Leaf precision | 0.7453 ± 0.0341 |
| Leaf recall | 0.8278 ± 0.0263 |
| Leaf F1 | 0.7835 ± 0.0126 |
| Pixel accuracy | 94.54% ± 0.60 percentage points |

These scores use each fold's own validation images, which also determine its best epoch.

### 8.4 U-Net cross-validation: fixed test set

| Fold | Leaf precision | Leaf recall | Leaf F1 | Pixel accuracy |
|---|---:|---:|---:|---:|
| 1 | 0.7379 | 0.8274 | 0.7801 | 93.07% |
| 2 | 0.7273 | 0.8120 | 0.7673 | 92.68% |
| 3 | 0.6787 | 0.8753 | 0.7646 | 91.99% |
| 4 | 0.7277 | 0.8198 | 0.7710 | 92.77% |
| 5 | 0.7724 | 0.7945 | 0.7833 | 93.47% |
| **Mean ± SD** | **0.7288 ± 0.0335** | **0.8258 ± 0.0302** | **0.7732 ± 0.0081** | **92.80% ± 0.54 pp** |

Background F1 averaged 0.9572 ± 0.0036. High pixel accuracy must be read alongside leaf F1 because background and leaf pixels contribute unequally to overall accuracy.

Source for both segmentation CV tables: `outputs/cross_validation/unet_segmentation_cv_results.csv`.

### 8.5 Pipeline performance with fixed ResNet50

| Fold | Accuracy | Macro precision | Macro recall | Macro F1 |
|---|---:|---:|---:|---:|
| 1 | 85.00% | 0.8334 | 0.8536 | 0.8386 |
| 2 | 86.25% | 0.8545 | 0.8561 | 0.8525 |
| 3 | 83.13% | 0.8161 | 0.8527 | 0.8228 |
| 4 | 89.38% | 0.8739 | 0.8944 | 0.8809 |
| 5 | 84.38% | 0.8220 | 0.8378 | 0.8254 |
| **Mean ± SD** | **85.63% ± 2.38 pp** | **0.8400 ± 0.0239** | **0.8589 ± 0.0211** | **0.8441 ± 0.0237** |

Pipeline accuracy ranged from 83.125% to 89.375%, a spread of 6.25 percentage points. The mean is 3.75 percentage points below the single-split baseline. This is a descriptive comparison, not a controlled estimate of a cross-validation effect, because the training protocols differ.

| Stage | Mean precision ± SD | Mean recall ± SD | Mean F1 ± SD |
|---|---:|---:|---:|
| Developing | 0.9474 ± 0.0464 | 0.9171 ± 0.0534 | 0.9305 ± 0.0315 |
| Flowering | 0.8455 ± 0.0332 | 0.7567 ± 0.0732 | 0.7966 ± 0.0415 |
| Fruiting | 0.5671 ± 0.0668 | 0.7619 ± 0.0583 | 0.6491 ± 0.0586 |
| Seeding | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |

Source: `outputs/cross_validation/fusion_pipeline_cv_results.csv`.

### 8.6 LPF export and descriptive statistics

| Stage | Images across all splits | Mean predicted LPF | Sample SD |
|---|---:|---:|---:|
| Developing | 203 | 0.2416 | 0.0782 |
| Flowering | 300 | 0.1927 | 0.0601 |
| Fruiting | 101 | 0.1443 | 0.0418 |
| Seeding | 378 | 0.0310 | 0.0198 |

Source: `export/lpf_full_dataset.csv`. These summarize predictions across training, validation, and test images together. They are descriptive dataset statistics, not independent test performance or evidence that LPF improves classification. No LPF-versus-ground-truth occupancy error analysis was established in this review.

## 9. Discussion

### 9.1 What is demonstrated

The project has a functioning route from an RGB image to a predicted growth stage using a learned intermediate leaf mask. The baseline achieves 89.38% accuracy on the recorded test set. Across five U-Net fold models, the same classifier achieves 85.63% mean accuracy. The small leaf-F1 standard deviation indicates limited spread in that particular segmentation metric across the five tested runs, while classification accuracy varies more visibly.

These observations apply to this dataset, this fixed classifier, and these training settings. They do not establish robustness to other cameras, farms, cultivars, seasons, or acquisition conditions.

### 9.2 Stage-dependent errors

Seeding is easiest in the observed evaluations. Fruiting has the weakest baseline segmentation and the lowest mean classification F1 across the fold pipelines. Its comparatively low classification precision indicates frequent false-positive fruiting predictions.

The smaller fruiting class and leaf occlusion are plausible contributors. However, occlusion severity was not quantified and no controlled class-balancing experiment was performed. Present these as possible explanations, not demonstrated causal findings. Also consider the hypothesis that removing non-leaf pixels discards useful flowers or fruit; an RGB-versus-masked ablation is needed to assess this.

### 9.3 Segmentation and classification measure different things

Fold 5 has the highest fixed-test leaf F1, whereas fold 4 has the highest pipeline accuracy. This shows that their rankings differ in the current results. It does not prove a general absence of association, particularly because segmentation and pipeline classification use different image preprocessing and only five models were tested.

### 9.4 What “fusion” means in this repository

The original project narrative anticipates combining LPF with ResNet features. The implemented combined notebook instead transfers information through a masked image. A paper should use “sequential segmentation–classification pipeline” or explicitly define its usage of “fusion.” It should not describe a learned feature-fusion head that is absent from these experiments.

## 10. Limitations and publication corrections

1. **Fixed classifier:** ResNet50 is reused unchanged across folds. Results characterize segmenter retraining, not the complete two-model training procedure.
2. **Classifier provenance:** the sibling checkpoint exists and was successfully loaded in the saved run, but its full training data, initialization, augmentation, optimization, and split history were not audited here. `weights=None` at loading does not establish whether its original training used pretrained weights.
3. **Split independence:** stage stratification does not enforce separation by plant, session, or near-duplicate image. Verify these relationships and use grouped splits if appropriate. This report does not assert that leakage occurred or that it has been ruled out.
4. **Validation reuse:** fold-validation subsets select checkpoints and provide reported validation scores. A stricter outer-fold estimate needs inner model selection or a training budget fixed without consulting outer-fold results.
5. **Repeated fixed-test evaluation:** all five pipelines share the same 160 test images. Do not describe them as 800 independent test cases or choose a deployment fold based on its test accuracy while continuing to call that score untouched evaluation. The notebook narrative also references previous test comparisons, so the full history of test-guided choices needs documenting.
6. **Unequal training protocols:** baseline training uses 60 epochs and a scheduler; CV uses 30 epochs and no scheduler. Differences in outcomes cannot be attributed solely to fold composition.
7. **Metric mismatch:** soft Dice, thresholded mean-image Dice, and pooled leaf F1 are different reporting protocols here. Label them precisely and recompute a consistent set if direct comparisons are needed.
8. **Cropped ground-truth resizing:** `gt_mask_cropped` says nearest-neighbor in its docstring but calls the shared `Resize` transform without an interpolation override, then thresholds at 127. Torchvision's default Resize interpolation is bilinear. Thus the saved cropped-mask segmentation scores reflect that implementation. Introduce an explicit nearest-neighbor mask transform and rerun those segmentation metrics before treating them as the intended nearest-neighbor evaluation. The label-based classification results do not directly depend on this ground-truth mask helper. [Torchvision documentation](https://docs.pytorch.org/vision/0.21/generated/torchvision.transforms.Resize.html).
9. **Small and imbalanced test classes:** seeding's 38/38 result does not establish perfect performance in the population; fruiting has only 21 test examples.
10. **Missing comparisons:** no matched RGB-only, ground-truth-mask oracle, LPF-only, or LPF-feature-fusion ablation is established by these four notebooks. Narrative references to other variants are not used as verified comparison results here.
11. **No external validation or deployment measurements:** inference latency, throughput, memory usage, calibration, and independent collection performance were not established.
12. **Reproducibility details:** several dependencies are unpinned, CV fold assignments are not exported as a dedicated manifest, and fold checkpoints contain weights without full training metadata. Baseline checkpoints contain weights and basic metadata, not a complete resumable optimizer state. Preserve the environment and run configuration before further experiments.
13. **Documentation inconsistencies:** reconcile the parameter count, bottleneck description, 255-versus-256 resize wording, and differing validation-metric provenance. Update the README to cover the later notebooks.

These are report findings; source notebooks and existing result files were not changed as part of preparing this document.

## 11. Workload completion inventory

| Work item | Evidence and status |
|---|---|
| COCO conversion | Complete; 982 processed image/mask pairs |
| Baseline U-Net training | Complete; best and final checkpoints, saved training outputs |
| Baseline segmentation evaluation | Complete; 160 per-image records and stage breakdown |
| LPF extraction | Complete; 982 rows in exported CSV |
| Standalone U-Net export | Complete; weights, Python loader, metadata |
| Additional v2 confusion-matrix cell | Implemented; no saved execution for the added cell |
| Combined U-Net → ResNet inference | Complete; saved model loads and visual walkthrough |
| Baseline pipeline test evaluation | Complete; 160 prediction records and confusion matrix |
| Five-fold U-Net training | Complete; five checkpoints and results |
| Pipeline evaluation for each U-Net fold | Complete; five result rows with fixed classifier |
| ResNet retraining within folds | Not implemented in these notebooks |
| LPF + ResNet feature-fusion training | Not implemented in these notebooks |
| Full data-leakage audit | Not established by this review |
| Independent external test evaluation | Not established by this review |

### Notebook responsibilities

- `unet_pipeline.ipynb`: annotation conversion, baseline U-Net training, segmentation evaluation, and LPF extraction.
- `unet_pipeline_v2.ipynb`: same source workflow plus an appended pixel-level confusion-matrix and precision/recall/F1 cell. It is not an independent architectural experiment.
- `unet_resnet_fusion_pipeline.ipynb`: load both trained models, align image preprocessing, predict growth stages, and evaluate the combined baseline.
- `unet_cross_validation.ipynb`: retrain five U-Nets, evaluate segmentation, and evaluate each with the fixed classifier.

## 12. Recommended next experiments

### 12.1 Complete training-procedure cross-validation

1. Audit image identities and plant/session groups across both projects and freeze a split manifest.
2. Reserve an independent final test set where possible, especially if the current test set has informed method selection.
3. Define outer folds on the remaining development data, using groups when required.
4. Within each outer training fold, select settings using only inner training/validation data.
5. Train a fresh U-Net and a fresh ResNet50 within that outer training fold; clearly specify whether classifier training uses ground-truth or predicted masks.
6. If classifier training uses predicted masks, consider inner out-of-fold mask generation so training images are not segmented by a U-Net trained on those same images.
7. Evaluate the complete pipeline on the excluded outer fold and save image-level predictions, fold IDs, and group IDs.
8. Summarize outer-fold and pooled out-of-fold results. After choices are fixed, train the final system and evaluate the reserved test set.

Both models may still be trained separately; full pipeline CV does not require differentiating through both networks jointly.

### 12.2 Ablations that would strengthen the paper

| Comparison | Research question |
|---|---|
| RGB-only ResNet vs predicted-mask ResNet | Does segmentation help classification under matched splits and training budgets? |
| Ground-truth-mask vs predicted-mask inference | What performance gap is associated with imperfect segmentation? |
| LPF-only vs ResNet features vs their combination | Does LPF contribute useful additional predictive information? |
| Matched 30- and 60-epoch segmenters | How much is explained by training budget? |
| Matched direct-resize vs aligned-crop pipelines | How does framing influence results under controlled conditions? |
| Balanced training or targeted fruiting augmentation | Can the weakest stage improve without harming others? |

These are proposed experiments, not completed results.

## 13. Draft conclusion

This work implemented and evaluated a tomato growth-stage classification pipeline in which U-Net leaf predictions mask RGB images before classification by a fixed ResNet50. The baseline pipeline achieved 89.38% accuracy on 160 test images. Five-fold U-Net training, followed by repeated evaluation with the unchanged classifier, produced a mean test leaf-pixel F1 of 0.7732 and a mean pipeline accuracy of 85.63%. Results varied by stage: seeding was classified correctly across all five pipeline evaluations, while fruiting remained the weakest class. The experiments demonstrate a working sequential pipeline and quantify its sensitivity to segmenter retraining. Full training-procedure cross-validation, consistent segmentation reporting, cross-project data auditing, and independent external evaluation remain necessary to support broader generalization claims.

## 14. Paper assembly guide

| Manuscript section | Material to use |
|---|---|
| Abstract | Section 2; revise after any new experiments |
| Introduction | Section 3 plus a domain-specific literature review |
| Dataset | Section 4 plus real collection and annotation records |
| Methods | Sections 5–7 |
| Results | Section 8, maintaining metric and evaluation-set labels |
| Discussion | Section 9 |
| Limitations | Section 10 |
| Conclusion | Section 13 |
| Supplementary material | Fold tables, per-image predictions, environment, split manifest, notebook inventory |

Recommended existing figures and captions:

1. **Baseline training curves** — `outputs/training_curves.png`: “Training and validation loss, soft Dice, and binary IoU over the 60-epoch baseline U-Net run.”
2. **Qualitative segmentation examples** — `outputs/qualitative_examples/test_examples.png`: “Test images, annotated leaf masks, and U-Net predictions.” State that these are illustrative examples rather than an error-rate estimate.
3. **Pipeline walkthrough** — `outputs/fusion_pipeline/pipeline_walkthrough.png`: “Shared image framing, predicted leaf masks, masked RGB inputs, and resulting growth-stage predictions.”
4. **Baseline classification confusion matrix** — `outputs/fusion_pipeline/full_pipeline_confusion_matrix.png`: “Predicted versus annotated growth stages for the 160-image test set.”
5. **Fold variability** — `outputs/cross_validation/cv_summary.png`: “Fixed-test segmentation leaf F1 and downstream accuracy for five U-Net fold models paired with one fixed ResNet50.”

Inspect figure text, class order, font sizes, and export quality before submission. This report verified numeric records but did not perform a visual publication-quality review of each saved figure.

### Claims suitable for the paper

- “Five-fold stratified cross-validation was performed for U-Net using an 822-image development pool.”
- “Each fold's segmenter was evaluated with the same fixed classifier on a common 160-image test set.”
- “Mean pipeline accuracy was 85.63% with a sample standard deviation of 2.38 percentage points.”

### Claims not supported by the current evidence

- “Both models underwent full five-fold cross-validation.”
- “A learned LPF–ResNet feature-fusion model achieved these results.”
- “Segmentation significantly improved performance over an RGB baseline.”
- “The system generalizes to all tomato varieties or field conditions.”
- “The five test evaluations represent 800 independent test images.”
- “The method has perfect seeding accuracy in deployment.”

## 15. Evidence index and references

### Local evidence

- [Original training notebook](../unet_pipeline.ipynb)
- [v2 notebook](../unet_pipeline_v2.ipynb)
- [Combined inference notebook](../unet_resnet_fusion_pipeline.ipynb)
- [Cross-validation notebook](../unet_cross_validation.ipynb)
- [Baseline per-image segmentation results](../outputs/test_set_per_image_scores.csv)
- [Baseline stage segmentation results](../outputs/test_scores_by_stage.csv)
- [Baseline pipeline predictions](../outputs/fusion_pipeline/full_pipeline_test_results.csv)
- [Segmentation CV results](../outputs/cross_validation/unet_segmentation_cv_results.csv)
- [Pipeline CV results](../outputs/cross_validation/fusion_pipeline_cv_results.csv)
- [Full-dataset LPF export](../export/lpf_full_dataset.csv)
- [Standalone model loader](../export/unet_lpf_extractor.py)
- [Export metadata, with discrepancies noted above](../export/metadata.json)
- [Dependency declarations](../requirements.txt)

### Verified external starting references

1. Ronneberger, O., Fischer, P., and Brox, T. (2015). *U-Net: Convolutional Networks for Biomedical Image Segmentation*. [arXiv:1505.04597](https://arxiv.org/abs/1505.04597).
2. He, K., Zhang, X., Ren, S., and Sun, J. (2015). *Deep Residual Learning for Image Recognition*. [arXiv:1512.03385](https://arxiv.org/abs/1512.03385).
3. PyTorch contributors. *Torchvision 0.21: Resize*. [Official API documentation](https://docs.pytorch.org/vision/0.21/generated/torchvision.transforms.Resize.html). Accessed 15 September 2026.

The references above use the verified arXiv records and official documentation. Convert them to the target journal's bibliography style and add relevant agricultural literature before submission.
