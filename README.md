# Driver Drowsiness Detection Using YOLOv7

Real-time object detection of driver behaviors associated with drowsiness and distraction using **YOLOv7**. The model is fine-tuned on **Driver Drowsiness v2** (Roboflow Universe) and supports inference on images, videos, and webcams, including an Android phone connected as a Windows webcam.

> **Note:** This project detects visual behaviors in individual frames. It is an experimental computer vision model, **not** a validated driver-safety or medical drowsiness monitoring system.

## Dataset

- **Dataset:** [Driver Drowsiness v2 — Roboflow Universe](https://universe.roboflow.com/dut-learning/driver-drowsiness-v2/dataset/2)
- **Workspace:** `dut-learning`
- **Version:** 2
- **License reported by the dataset:** CC BY 4.0
- **Number of classes:** 6

|  ID | Class        | Description                        |
| --: | ------------ | ---------------------------------- |
|   0 | `look_away`  | Driver looks away                  |
|   1 | `natural`    | Normal driver behavior             |
|   2 | `phone`      | Driver uses or holds a phone       |
|   3 | `rub_eye`    | Driver rubs their eyes             |
|   4 | `sleepy_eye` | Driver shows sleepy or closed eyes |
|   5 | `yawn`       | Driver yawns                       |

The class order must match the original annotation IDs exactly.

## Training configuration

| Setting               | Value                            |
| --------------------- | -------------------------------- |
| Architecture          | YOLOv7 (original implementation) |
| Pretrained checkpoint | `yolov7_training.pt`             |
| Epochs                | 50                               |
| Batch size            | 4                                |
| Image size            | 640 × 640                        |
| Classes               | 6                                |
| Device                | NVIDIA CUDA GPU (`--device 0`)   |
| Workers               | 0 (Windows-compatible setup)     |

## Validation results (epoch 50)

| Metric       |     Result |
| ------------ | ---------: |
| Precision    | **95.85%** |
| Recall       | **90.76%** |
| mAP@0.5      | **94.52%** |
| mAP@0.5:0.95 | **64.63%** |

These values are from the **validation log at the final epoch** (`results.txt`), **not** an independent test-set evaluation. Performance on new drivers, lighting conditions, or cameras may differ. The model's `best.pt` checkpoint may correspond to a different epoch.

## Repository structure

```text
Driver-Drowsiness-Detection-YOLOv7/
├── weights/
│   └── best.pt
├── config/
│   └── driver_drowsiness.yaml
├── results/
│   ├── results.png
│   └── results.txt
├── README.md
└── .gitignore
```

The repository contains the trained weights and supporting artifacts; inference uses the original YOLOv7 source code.

## Requirements

- Windows 10/11 (instructions below use PowerShell)
- Python 3.10 recommended for the original YOLOv7 dependencies
- Git
- PyTorch and torchvision compatible with your hardware
- Optional: NVIDIA GPU with CUDA support for faster inference
- Optional: Iriun Webcam (or another phone-to-webcam app) for Android camera input

## Quick start

### 1. Clone the original YOLOv7 code

```powershell
git clone https://github.com/WongKinYiu/yolov7.git
cd yolov7
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install compatible PyTorch and YOLOv7 dependencies in this environment. Consult the [YOLOv7 repository](https://github.com/WongKinYiu/yolov7) and [PyTorch installation guide](https://pytorch.org/get-started/locally/) for a build appropriate to your GPU/CPU. The original codebase may require legacy dependency versions.

### 2. Obtain the trained model

Download `best.pt` from this repository's `weights/` directory (or its GitHub Release, if distributed there), and place it at:

```text
yolov7/weights/best.pt
```

Create the `weights` directory if it does not exist.

### 3. Detect objects in an image or folder

```powershell
python detect.py `
  --weights weights/best.pt `
  --source inference/images `
  --img-size 640 `
  --conf-thres 0.25 `
  --device cpu `
  --name driver_drowsiness_images
```

Replace `inference/images` with the path to your own image, video, or image folder. Results are saved under `runs/detect/`.

### 4. Real-time webcam detection

```powershell
python detect.py `
  --weights weights/best.pt `
  --source 0 `
  --img-size 640 `
  --conf-thres 0.25 `
  --device cpu `
  --view-img `
  --nosave
```

- `--source 0`: default webcam. Try `1` or `2` if your Android phone appears as an additional camera.
- `--device cpu`: runs without CUDA. Change to `--device 0` if compatible NVIDIA CUDA support is correctly installed.
- `--view-img`: shows the live detection window.
- `--nosave`: avoids recording webcam output.

To use a phone camera, connect it through **Iriun Webcam** or a similar utility and identify its Windows camera index. Run tests while stationary, not while driving.

## Dataset configuration for retraining

In the original YOLOv7 repository, save a dataset YAML like this in `data/driver_drowsiness.yaml`:

```yaml
train: ./dataset/train/images
val: ./dataset/val/images
test: ./dataset/test/images
nc: 6
names:
  - look_away
  - natural
  - phone
  - rub_eye
  - sleepy_eye
  - yawn
```

These paths assume you organized the dataset into `dataset/train`, `dataset/val`, and `dataset/test` folders beneath the YOLOv7 root. Adjust the paths to match your actual data location. For a new experiment, keep the test set separate from training and model selection, and check for near-duplicate frames across splits.

Create `cfg/training/yolov7-custom.yaml` from `cfg/training/yolov7.yaml` and change the architecture's `nc` to `6`.

### Train

```powershell
python train.py `
  --workers 0 `
  --device 0 `
  --batch-size 4 `
  --epochs 50 `
  --img-size 640 640 `
  --data data/driver_drowsiness.yaml `
  --cfg cfg/training/yolov7-custom.yaml `
  --weights yolov7_training.pt `
  --hyp data/hyp.scratch.custom.yaml `
  --name driver_drowsiness_yolov7
```

This step requires separately downloading the pretrained `yolov7_training.pt` weights from the original YOLOv7 project.

### Evaluate on the test split

```powershell
python test.py `
  --weights weights/best.pt `
  --data data/driver_drowsiness.yaml `
  --batch-size 4 `
  --img-size 640 `
  --task test `
  --device 0 `
  --verbose
```

Replace `--device 0` with `--device cpu` if necessary. **Test-set scores are not included in the reported results above.**

## Limitations

- Single-frame predictions do not establish that a driver is clinically drowsy.
- Detection may vary with lighting, camera angle, occlusion, eyewear, and unfamiliar drivers.
- A robust alerting system would require temporal logic, separate real-world validation, and safety testing.

## Credits and references

- **YOLOv7:** [WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
- **Dataset:** [Driver Drowsiness v2, DUT Learning, Roboflow Universe](https://universe.roboflow.com/dut-learning/driver-drowsiness-v2/dataset/2) (CC BY 4.0 according to the exported dataset metadata)
