# R26-IT-143 — Honest Model Quality Assessment (for the team + viva prep)

## C1 — Swimmer detection (crowd)
Two models: v1 (single-source KSU-style dataset) and v2_combined.
v1 reportedly ~85.5% vs v2 ~80.6% → the dataset-heterogeneity finding.
VERDICT: usable; v1 is the right default. ISSUES: the exported models
(6 MB ≈ yolov8n-scale) don't match the training script (yolov8m) or the
24 MB run weights in results/ — verify which checkpoint was actually
exported, and re-run ml/scripts/evaluate_custom_model.py to produce
current P/R/mAP numbers for the thesis (no metrics file was included).

## C2 — Water quality (XGBoost)
Accuracy = 1.00, F1 = 1.00 on the synthetic augmented dataset.
VERDICT: works, but a perfect score means the synthetic classes are
separable by simple thresholds — a panel WILL probe this. DO: (a) compare
against a plain threshold baseline, (b) add noise/unseen ranges or real
samples to show genuine generalization, (c) k-fold cross-validation.
Honest framing: "the model matches expert thresholds on synthetic data;
its value is the pipeline + future retraining on real sensor logs."

## C3 — Drowning detection
P=0.877 · R=0.883 · mAP50=0.907 · mAP50-95=0.594 (100 epochs).
VERDICT: genuinely good for a student project — the strongest model.
DISCUSS: recall 0.883 → ~12% of drowning instances missed per frame; the
3-consecutive-frame rule trades alert stability vs. detection delay.
For a safety system argue recall-priority: consider lowering conf
threshold and measuring the resulting precision/recall shift (great
evaluation-chapter material). Weights (best.pt) must be added to the repo.

## C4 — Garbage detection
P=0.853 · R=0.878 · mAP50=0.903 · mAP50-95=0.393 (10 epochs, CPU, yolov8n).
VERDICT: promising but under-trained. mAP50 is decent; low mAP50-95 =
loose bounding boxes; 25 test images is far too few for credible claims.
DO: retrain 50-100 epochs on GPU (free Colab), evaluate on a bigger test
split, and CITE the Roboflow dataset (CC BY 4.0 — attribution is a
license requirement, not optional). The ball/leaf ≠ garbage class logic
is a genuine strength — present it as false-positive mitigation.

## Cross-cutting
All four models were trained/evaluated on different machines with no
common evaluation protocol. For the thesis, produce ONE results table
(same metric definitions, stated test-set sizes) and state test-set
sizes explicitly — small test sets are a standard panel question.
