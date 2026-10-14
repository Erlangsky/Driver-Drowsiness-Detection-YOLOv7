# Driver Drowsiness Detection Using YOLOv7

Real-time object detection of driver behaviors associated with drowsiness and distraction using **YOLOv7**. The model is fine-tuned on **Driver Drowsiness v2** (Roboflow Universe) and supports inference on images, videos, and webcams, including an Android phone connected as a Windows webcam.

> **Note:** This project detects visual behaviors in individual frames. It is an experimental computer vision model, **not** a validated driver-safety or medical drowsiness monitoring system.

## Dataset

- **Dataset:** [Driver Drowsiness v2 — Roboflow Universe](https://universe.roboflow.com/dut-learning/driver-drowsiness-v2/dataset/2)
- **Workspace:** `dut-learning`
- **Version:** 2
- **License reported by the dataset:** CC BY 4.0
- **Number of classes:** 6

| ID | Class | Description |
|---:|---|---|
| 0 | `look_away` | Driver looks away |
| 1 | `natural` | Normal driver behavior |
| 2 | `phone` | Driver uses or holds a phone |
| 3 | `rub_eye` | Driver rubs their eyes |
| 4 | `sleepy_eye` | Driver shows sleepy or closed eyes |
| 5 | `yawn` | Driver yawns |

The class order must match the original annotation IDs exactly.

## Training configuration

| Setting | Value |
|---|---|
| Architecture | YOLOv7 (original implementation) |
| Pretrained checkpoint | `yolov7_training.pt` |
| Epochs | 50 |
| Batch size | 4 |
| Image size | 640 × 640 |
| Classes | 6 |
| Device | NVIDIA CUDA GPU (`--device 0`) |
| Workers | 0 (Windows-compatible setup) |

## Validation results (epoch 50)

| Metric | Result |
|---|---:|
| Precision | **95.85%** |
| Recall | **90.76%** |
| mAP@0.5 | **94.52%** |
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

### 2. Install PyTorch and dependencies (Windows / Python 3.10)

First, confirm that your virtual environment is active and upgrade pip:

```powershell
python --version
python -m pip install --upgrade pip
```

**Choose one PyTorch installation option**, based on your computer. The versions below are a reproducible *legacy setup* for the original YOLOv7 codebase, rather than the newest PyTorch release.

**Option A — NVIDIA GPU (CUDA 11.8 build):**

Check your GPU and driver with `nvidia-smi`. If your NVIDIA GPU/driver supports this CUDA build, run:

```powershell
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

**Option B — CPU only (no compatible NVIDIA GPU):**

```powershell
python -m pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

> **Compatibility note:** CUDA support depends on both the GPU architecture and NVIDIA driver, not just having a 4 GB GPU. Newer GPUs may need a newer PyTorch/CUDA combination. If the above CUDA build does not support your device, select a supported install command from the [official PyTorch selector](https://pytorch.org/get-started/locally/) and check compatibility with the legacy YOLOv7 code. You normally do **not** need to install the entire CUDA Toolkit to use the prebuilt PyTorch wheel.

Install additional dependencies commonly needed by the original YOLOv7 training/inference scripts:

```powershell
python -m pip install numpy==1.23.5 opencv-python==4.8.1.78 scipy==1.10.1 pandas==1.5.3 matplotlib==3.7.3 seaborn==0.12.2 protobuf==3.20.3
python -m pip install Pillow PyYAML requests tqdm tensorboard psutil thop ipython "setuptools<81"
```

The repository's unpinned or legacy `requirements.txt` may resolve to a different dependency combination. Do not blindly reinstall it over a working environment; if extra packages are reported missing, install them individually.

**Verify the installation:**

```powershell
python -c "import torch, torchvision, cv2; print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('OpenCV:', cv2.__version__); print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

Example output on a supported NVIDIA configuration:

```text
PyTorch: 2.0.1+cu118
Torchvision: 0.15.2+cu118
OpenCV: 4.8.1
CUDA available: True
Device: NVIDIA GeForce ...
```

If `CUDA available: False`, run inference with `--device cpu` and check the GPU driver/build if you expected CUDA. If PowerShell prevents activating `.venv`, you can allow activation for the current terminal only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### 3. Obtain the trained model


Download `best.pt` from this repository's `weights/` directory (or its GitHub Release, if distributed there), and place it at:

```text
yolov7/weights/best.pt
```

Create the `weights` directory if it does not exist.

### 4. Detect objects in an image or folder

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

### 5. Real-time webcam detection (built-in laptop camera)

Test the standard webcam first:

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

- `--source 0`: camera index 0 (typically the laptop camera, but this varies).
- `--device cpu`: works without NVIDIA CUDA. Use `--device 0` for a correctly configured NVIDIA GPU.
- `--view-img`: displays the detection window; `--nosave` avoids saving the live video.
- Focus the preview window and press **Q** to exit, or press **Ctrl+C** in the terminal.

## Use an Android phone camera for real-time detection

You can use an Android smartphone as the camera and run YOLOv7 **on the Windows laptop**. The phone only supplies the video stream; inference uses your laptop's CPU or GPU.

### Step 1 — Install and connect Iriun Webcam

1. Install **Iriun Webcam** on your Android phone and the **Iriun Webcam desktop app** on Windows: [iriun.com](https://iriun.com/).
2. Open both applications and allow camera access on the phone.
3. For the simplest setup, connect the phone and laptop to the **same Wi-Fi network**. Wait for the phone's camera preview to appear on the computer.
4. A USB connection may also be possible depending on the Iriun version and device setup; follow Iriun's own USB instructions if you prefer a cable.
5. Close Zoom, Teams, or other programs that might be using the Iriun camera exclusively.
6. If Windows blocks camera access, check **Settings → Privacy & security → Camera**, and enable camera access for desktop apps.

> Tip: When the connection is successful, Windows should expose the phone camera as a webcam device. This means YOLOv7 can use a numeric camera index such as `0`, `1`, or `2`.

### Step 2 — Find the phone camera index

In the YOLOv7 root directory, create a file named `check_camera.py`:

```python
import cv2

for index in range(5):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if cap.isOpened():
        ok, frame = cap.read()
        if ok:
            height, width = frame.shape[:2]
            print(f"Camera index {index}: available ({width}x{height})")
    cap.release()
```

Run it from the activated YOLOv7 Python environment:

```powershell
python check_camera.py
```

Example output (your indices can differ):

```text
Camera index 0: available (1280x720)
Camera index 1: available (1920x1080)
```

To verify which index belongs to the phone, test each available index with a preview. You can create `preview_camera.py`:

```python
import cv2
import sys

index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
if not cap.isOpened():
    raise SystemExit(f"Could not open camera index {index}")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("No frame received")
            break
        cv2.imshow(f"Camera index {index} - press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
```

Then try `python preview_camera.py 1` (or another index). Point the phone camera at an object and confirm the preview changes accordingly. Close the preview before starting detection.

### Step 3 — Run YOLOv7 with the phone camera

From within the **original YOLOv7 repository** with `weights/best.pt` in place:

```powershell
python detect.py `
  --weights weights/best.pt `
  --source 1 `
  --img-size 640 `
  --conf-thres 0.25 `
  --iou-thres 0.45 `
  --device 0 `
  --view-img `
  --nosave `
  --name realtime_phone
```

**Important:** Replace `--source 1` with the actual Iriun camera index. Replace `--device 0` with `--device cpu` if CUDA is unavailable. If you are running inference directly from your original training project, your checkpoint may instead be at `runs/train/driver_drowsiness_yolov7/weights/best.pt`.

The model detects these six classes: `look_away`, `natural`, `phone`, `rub_eye`, `sleepy_eye`, and `yawn`. When confidence is high enough, detections appear as labeled bounding boxes on the video.

### Step 4 — If detection is slow

Try a smaller inference size:

```powershell
python detect.py `
  --weights weights/best.pt `
  --source 1 `
  --img-size 416 `
  --conf-thres 0.25 `
  --device 0 `
  --view-img `
  --nosave
```

This can improve speed but may reduce detection quality for small details. Close other heavy applications and consider a lower phone-camera resolution if needed.

### Troubleshooting phone camera

| Issue | What to try |
|---|---|
| Phone preview does not appear in Iriun | Confirm both apps are running, allow camera permissions, check the same Wi-Fi network or supported USB setup. |
| YOLOv7 opens laptop camera instead | Change `--source` to the verified Iriun camera index. |
| Cannot open webcam / black screen | Close other camera apps, check Windows camera permissions, and test with `preview_camera.py`. |
| No bounding boxes | Ensure `best.pt` loads correctly, improve lighting, and optionally try `--conf-thres 0.10` for debugging. |
| Low FPS | Try `--img-size 416`, use CUDA if configured, and close resource-heavy applications. |

**Safety:** Test simulated driver behavior only while parked or seated safely, never while operating a moving vehicle. Do not use this experimental model as a safety-critical alert system.

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

