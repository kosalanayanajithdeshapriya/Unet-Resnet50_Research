"""Add full-pipeline confusion matrices to Segmentation_Comparison_Report.docx.

Sources for the three 160-image test-set matrices (Table 6 variants):
  - U-Net (crop-aligned): recomputed from results/fusion_pipeline/full_pipeline_test_results.csv
  - U-Net (native resize): Resnet50test/unet_pipeline_confusion_matrix.png (acc 0.7875)
  - DeepLabV3 (acc 0.8438 run used in Table 6): reconstructed from the saved
    classification_report in Resnet50test/unet_leaf_pipeline_comparison.ipynb (cell 24).
    Supports + 2-dp recall fix each TP uniquely; the 4-dp per-class F1 (cell 28) then fixes
    each predicted-column total, and the off-diagonal system has a single non-negative solution.
    It reproduces all Table 7 F1 values and 135/160 correct.
"""
from pathlib import Path
import copy
import csv
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports'
SRC_DOCX = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / 'Downloads/Segmentation_Comparison_Report.docx'
OUT_DOCX = OUT / 'Segmentation_Comparison_Report_with_CM.docx'
FIG = ROOT / 'results' / 'pipeline_confusion_matrices.png'

CLASSES = ['developing', 'flowering', 'fruiting', 'seeding']
LABELS = ['Developing', 'Flowering', 'Fruiting', 'Seeding']

with open(ROOT / 'results/fusion_pipeline/full_pipeline_test_results.csv', encoding='utf-8-sig') as f:
    preds = list(csv.DictReader(f))
unet_crop = [[sum(r['true_stage'] == a and r['pred_stage'] == b for r in preds) for b in CLASSES] for a in CLASSES]

MATRICES = [
    ('DeepLabV3-ResNet50', [[36, 5, 0, 0], [1, 46, 13, 0], [0, 6, 15, 0], [0, 0, 0, 38]]),
    ('U-Net (native resize)', [[39, 2, 0, 0], [2, 31, 27, 0], [0, 2, 18, 1], [0, 0, 0, 38]]),
    ('U-Net (crop-aligned)', unet_crop),
]


def f1_scores(cm):
    out = []
    for i in range(4):
        tp = cm[i][i]
        out.append(2 * tp / (sum(cm[i]) + sum(row[i] for row in cm)))
    return out


def accuracy(cm):
    return sum(cm[i][i] for i in range(4)) / sum(map(sum, cm))


# Sanity checks against the report's Tables 6 and 7.
expected = {
    'DeepLabV3-ResNet50': (0.8438, [0.9231, 0.7863, 0.6122, 1.0]),
    'U-Net (native resize)': (0.7875, [0.9512, 0.6526, 0.5455, 0.9870]),
    'U-Net (crop-aligned)': (0.8938, [0.95, 0.86, 0.70, 1.0]),  # Table 7 shows 2-dp values here
}
for name, cm in MATRICES:
    acc, f1 = expected[name]
    assert sum(map(sum, cm)) == 160 and [sum(r) for r in cm] == [41, 60, 21, 38], name
    assert round(accuracy(cm), 4) == acc, (name, accuracy(cm))
    digits = 2 if name == 'U-Net (crop-aligned)' else 4
    assert [round(x, digits) for x in f1_scores(cm)] == f1, (name, f1_scores(cm))

# --- Figure: one sequential blue ramp, shared 0-100% row-normalised scale -----------------
INK, MUTED, SURFACE = '#1f1f1d', '#6b6a66', '#ffffff'
cmap = LinearSegmentedColormap.from_list(
    'seq_blue', ['#f7f9fc', '#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'])

plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'text.color': INK,
                     'axes.labelcolor': MUTED, 'xtick.color': MUTED, 'ytick.color': MUTED})
fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.75), dpi=300)
for k, (ax, (name, cm)) in enumerate(zip(axes, MATRICES)):
    norm = [[v / sum(row) for v in row] for row in cm]
    im = ax.imshow(norm, cmap=cmap, vmin=0, vmax=1)
    for i in range(4):
        for j in range(4):
            v = norm[i][j]
            color = SURFACE if v >= 0.55 else (INK if cm[i][j] else '#b5b3ad')
            weight = 'bold' if i == j else 'normal'
            ax.text(j, i - 0.12, str(cm[i][j]), ha='center', va='center', fontsize=11, color=color, fontweight=weight)
            if cm[i][j]:
                ax.text(j, i + 0.22, f'{v:.0%}', ha='center', va='center', fontsize=7, color=color)
    # 2px surface gap between cells
    for t in range(5):
        ax.axhline(t - 0.5, color=SURFACE, lw=2)
        ax.axvline(t - 0.5, color=SURFACE, lw=2)
    ax.set_xticks(range(4), LABELS, rotation=30, ha='right')
    ax.set_yticks(range(4), LABELS if k == 0 else [''] * 4)
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel('Predicted stage')
    if k == 0:
        ax.set_ylabel('True stage')
    errors = 160 - sum(cm[i][i] for i in range(4))
    ax.set_title(f'{name}\nacc = {accuracy(cm):.2%}  ·  {errors} errors', fontsize=9.5, color=INK, pad=8)

cbar = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
cbar.set_label('Share of true-class images (row %)', color=MUTED)
cbar.outline.set_visible(False)
cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
fig.savefig(FIG, bbox_inches='tight', facecolor=SURFACE)
plt.close(fig)

# --- Insert into the report, just before "Table 8" ----------------------------------------
doc = Document(SRC_DOCX)
paras = doc.paragraphs
anchor = next(p for p in paras if p.text.startswith('Table 8.'))
heading_tpl = anchor
caption_tpl = next(p for p in paras if p.text.startswith('Both checkpoints (unet_best.pt'))   # italic caption
body_tpl = next(p for p in paras if p.text.startswith('Key qualitative finding'))            # bold lead + body


def clone_before(tpl, texts):
    """Copy a template paragraph (keeps style/run formatting) and set run texts."""
    new = copy.deepcopy(tpl._p)
    anchor._p.addprevious(new)
    from docx.text.paragraph import Paragraph
    p = Paragraph(new, anchor._parent)
    runs = p.runs
    for r in runs[len(texts):]:
        r._r.getparent().remove(r._r)
    for r, t in zip(p.runs, texts):
        r.text = t
    return p


fig_head = clone_before(heading_tpl, ['Figure 1. Full-Pipeline Confusion Matrices (Test Set, n = 160)'])
fig_head.paragraph_format.keep_with_next = True
cap = clone_before(caption_tpl, [
    'Rows are true growth stages and columns are predicted stages for the three Table 6 pipelines. '
    'Each cell shows the image count and its share of the true class (row %); colour encodes the row %, '
    'so classes of different size (21 fruiting vs 60 flowering) are directly comparable. '
    'Diagonal cells are correct predictions.'])
cap.paragraph_format.keep_with_next = True

img_p = copy.deepcopy(anchor._p)
anchor._p.addprevious(img_p)
from docx.text.paragraph import Paragraph
img_par = Paragraph(img_p, anchor._parent)
for r in img_par.runs:
    r._r.getparent().remove(r._r)
img_par.style = doc.styles['Normal']
img_par.alignment = WD_ALIGN_PARAGRAPH.CENTER
img_par.add_run().add_picture(str(FIG), width=Inches(7.0))

clone_before(body_tpl, [
    'What the confusion matrices show: ',
    'Seeding is classified perfectly by all three pipelines, and most errors sit in the '
    'flowering/fruiting block (19 of 25 for DeepLabV3, 29 of 34 for native U-Net, 13 of 17 for crop-aligned U-Net). Native U-Net misclassifies 27 of 60 flowering images (45%) as fruiting, '
    'which accounts for most of its 34 errors. Crop-aligned preprocessing cuts that to 7, and the flowering '
    'recall rises from 52% to 85% without retraining the segmenter. This is consistent with the '
    'geometry-misalignment explanation above. DeepLabV3 sits between the two (13 flowering→fruiting, '
    '6 fruiting→flowering) and is the only pipeline with notable developing→flowering confusion (5 of 41). '
    'Fruiting is the hardest class for every pipeline: 6 of 21 (crop-aligned U-Net and DeepLabV3) '
    'are predicted as flowering, which matches its lowest F1 in Table 7.'])
prov = clone_before(caption_tpl, [
    'Provenance: U-Net (crop-aligned) counts are recomputed from the per-image predictions '
    '(full_pipeline_test_results.csv); U-Net (native) counts are from the saved comparison-notebook '
    'confusion matrix. The DeepLabV3 matrix is for the 0.8438 run used in Tables 6–7. It was reconstructed from '
    'that run\'s saved classification report (per-class support, recall and F1), which determines the counts '
    'uniquely. A separately saved DeepLabV3 rerun with 0.8750 accuracy has a slightly different matrix and is '
    'not shown, so that the figure matches Table 6.'])
prov.alignment = WD_ALIGN_PARAGRAPH.LEFT

# Extra Key Findings bullet
last_bullet = [p for p in doc.paragraphs if p.style.name == 'List Paragraph'][-1]
new_b = copy.deepcopy(last_bullet._p)
last_bullet._p.addnext(new_b)
b = Paragraph(new_b, last_bullet._parent)
for r in b.runs[1:]:
    r._r.getparent().remove(r._r)
b.runs[0].text = ('Confusion matrices (Figure 1): most pipeline errors are flowering↔fruiting confusions. '
                  'Crop alignment reduces U-Net\'s flowering→fruiting errors from 27 to 7, which accounts for most of '
                  'its accuracy gain, and seeding is perfectly classified by every variant.')

next(p for p in doc.paragraphs if p.text.strip() == 'Key Findings').paragraph_format.keep_with_next = True

doc.save(OUT_DOCX)
print('Saved', OUT_DOCX)
print('Saved', FIG)
