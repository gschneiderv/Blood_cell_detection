# Blood cell detection with YOLO26

A YOLO26m model, fine-tuned on Kaggle's GPU, that finds white blood cells, red blood cells and platelets in blood smear images. The repo includes the full pipeline, a live training dashboard and three reports.

| Test split (126 images, 1,868 cells) | mAP50 | mAP50-95 | Precision | Recall |
| --- | ---: | ---: | ---: | ---: |
| **All classes** | **0.987** | **0.875** | 0.961 | 0.964 |
| WBC (131 cells) | 0.995 | 0.888 | 1.000 | 1.000 |
| RBC (1,688 cells) | 0.992 | 0.905 | 0.964 | 0.973 |
| Platelets (49 cells) | 0.974 | 0.831 | 0.918 | 0.918 |

The 95% interval for overall mAP50-95 is 0.855 to 0.893. Precision and recall use the confidence threshold chosen on validation (0.47), at IoU 0.50. Inference takes about 29 ms per image on a Tesla T4.

These results come from the same four image collections the model trained on. Its performance on images from another laboratory has not been measured, and it is not validated for clinical use. See [Limitations](#limitations).

## View the results

Open these files directly in a browser. They don't need a server.

- [Dashboard](dashboard/index.html): training progress, model performance and the dataset, updating live while a run is in progress
- [Dataset report](reports/dataset_report.html): what was used, what was checked and what was corrected
- [Training report](reports/training_report.html): settings, candidates and training curves
- [Performance report](reports/performance_report.html): test results, errors and limitations

## Use the model

```bash
python scripts/predict.py path/to/image_or_folder --out annotated/
```

```python
import cv2
from ultralytics import YOLO
from scripts.predict import detect

model = YOLO("models/yolo26_blood_cells.pt")
cells = detect(model, cv2.imread("smear.png"), conf=0.47)
# [{"class": "RBC", "confidence": 0.93, "box": [x1, y1, x2, y2]}, ...]
```

**Use `detect` rather than calling `model.predict()` directly.** Ultralytics pads images with a gray border during validation but not during prediction, and cells cut off by the image edge are detected much worse without it. On validation, macro F1 is 0.970 with the border and 0.953 without ([comparison](results/border_comparison.json)). `detect` adds the border for you.

## Data

Three datasets are provided in `data/`, which isn't committed because of its size. Only one can train a detector.

| Dataset | Contents | Used | Why |
| --- | --- | --- | --- |
| TXL-PBC | 1,260 images, 18,143 boxes, 3 classes | Yes | The only one with cell locations |
| BloodMNIST | 17,092 single-cell crops, 8 classes | No | No boxes; neighbouring cells appear unlabelled; it comes from a collection TXL-PBC already includes |
| Dataset_NeuNN | 5,605 cell crops, 7 undocumented folders | No | No boxes; appears to describe neutrophil abnormalities, which is a different question |

Two corrections were applied to TXL-PBC before training, leaving `data/` itself unchanged:

1. **Leakage.** Comparing all 793,170 image pairs found 30 pairs showing the same field of view across splits, including 10 of the 126 test images. 31 training and 2 validation images were removed. The test split is exactly as provided.
2. **Duplicate labels.** 47 exact duplicate rows were removed.

## How it was trained

Four candidates were fine-tuned from COCO-pretrained weights for 100 epochs each and compared on validation. The smallest model within 0.005 of the best validation mAP50-95 was selected, and only that model was evaluated on the test split.

| Candidate | Val mAP50-95 | Parameters | Training time |
| --- | ---: | ---: | ---: |
| **YOLO26m, full fine-tune (selected)** | **0.869** | 21.8M | 55 min |
| YOLO26s, full fine-tune | 0.862 | 10.0M | 27 min |
| YOLO26s, frozen backbone | 0.860 | 10.0M | 22 min |
| YOLO26n, full fine-tune | 0.858 | 2.5M | 17 min |

Settings are in [configs/experiment.yaml](configs/experiment.yaml).

## Limitations

- **Not yet tested on a new laboratory.** Test images come from the same collections as the training images. Images of one slide or patient could still appear in both, because file names don't record this.
- **Platelet detection depends on the stain.** Across validation and test, the model found 88 of 91 labelled platelets in PBC images and 61 of 64 in BCCD, but only 2 of 6 in Raabin-WBC. BCDD has no platelets in those splits.
- **Stain debris can be mistaken for platelets.**
- **The perfect WBC result covers only 131 cells**, usually large and central in the image. Fields with many or atypical white cells have not been tested.
- **It only distinguishes three classes.** It doesn't identify white cell subtypes or abnormal cells.
- **Labels began as YOLOv8n predictions**, so scores measure agreement with those labels rather than an independent expert reading.

## Reproduce

Kaggle only gives GPU and internet access to phone-verified accounts. Without verification, kernels silently run on a CPU with no network, and the training kernel stops rather than train on a CPU.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # then add your KAGGLE_API_TOKEN

python scripts/audit_dataset.py            # check the data against its documentation
python scripts/find_scene_twins.py         # compare all image pairs (about 15 minutes)
python scripts/prepare_dataset.py --owner YOUR_KAGGLE_USERNAME
python scripts/kaggle_jobs.py upload-dataset   # stays private

python scripts/kaggle_jobs.py push --smoke     # two short candidates, about 5 minutes
python scripts/kaggle_jobs.py push             # the full run, about 2 hours
caffeinate -i python scripts/monitor.py        # macOS: stops the computer sleeping while it runs
```

Change `OWNER` in [scripts/kaggle_jobs.py](scripts/kaggle_jobs.py) to your Kaggle username first. Open the dashboard to follow progress. When the kernel finishes, the monitor downloads its output, evaluates it and rebuilds the reports. `scripts/compare_border.py` regenerates the border comparison.

To repeat the final evaluation without retraining, upload a run's artifacts and push an evaluate-only kernel:

```bash
python scripts/kaggle_jobs.py upload-artifacts --artifacts results/raw/KERNEL/artifacts
python scripts/kaggle_jobs.py push --evaluate-only
python scripts/monitor.py --base-kernel YOUR_KAGGLE_USERNAME/yolo26-blood-cell-detection
```

## Repository layout

| Path | Contents |
| --- | --- |
| `scripts/` | Data audit, Kaggle job control, monitor, evaluation, reports and prediction |
| `kaggle/train_kernel.py` | The training job that runs on Kaggle |
| `configs/` | Experiment settings and the written conclusions in the reports |
| `models/yolo26_blood_cells.pt` | The final model |
| `dashboard/` | The dashboard and its data |
| `reports/` | The three reports and the data audit results |
| `results/` | Final metrics and per-candidate training logs |

## Credits and licences

- **Dataset:** Gan, Li and Wang, [TXL-PBC](https://arxiv.org/abs/2407.13214), which combines BCCD, BCDD, PBC and Raabin-WBC. Check the [dataset repository](https://github.com/lugan113/TXL-PBC_Dataset) for its licence terms before redistributing the data.
- **Model and training framework:** [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/), licensed under AGPL-3.0. That licence also affects models trained with it, so check [Ultralytics' licensing](https://www.ultralytics.com/license) before any commercial use.
