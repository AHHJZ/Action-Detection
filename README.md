# 🏃‍♂️ Action-Detection
End-to-end pipeline for **multi-person action detection** using **YOLO-Pose** for keypoint extraction and a **Temporal Convolutional Network (TCN)** for sequence classification.  
Supports real-time video/webcam inference, ONNX export, and safety-critical actions like **fall detection**.  

The model is trained to recognize the following classes:  
**Fall, Lie, LikeFall, Siting, Stand, Walking**

> ⚠️ **Note:** The *military salute* class is included **for fun purposes only**.

---

## 📌 Features
- **YOLO-Pose (Ultralytics)** to extract 17 COCO keypoints per frame.  
- Temporal features: positions, multi-scale velocities/accelerations, and optional joint-angle dynamics.  
- **TCN classifier** with dilated temporal convolutions.  
- Multi-person tracking via IoU assignment.  
- Real-time visualization (bounding boxes + action label + FPS).  
- Robust fall detection logic & heuristics (safety use-cases).  

---

## 🔧 Requirements & Installation
- Python **3.8+**

Install dependencies:
```bash
pip install -r requirements.txt
````

`requirements.txt`:

```txt
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
opencv-python>=4.8.0
ultralytics>=8.0.0
numpy>=1.23.0
scikit-learn>=1.2.0
```

---

## 📂 Dataset Structure

Organize your data as: **class folders → sequence folders → frames (images):**

```
pure_data/
 ├─ Fall/
 │   ├─ seq001/ frame_001.jpg, frame_002.jpg, ...
 │   ├─ seq002/ ...
 ├─ Lie/
 ├─ LikeFall/    
 ├─ Sitting/
 ├─ Stand/
 └─ Walking/
```

* **Class names** are used as labels.
* **Sequence folders** must start with `seq` (e.g., `seq1`, `seq_003`).

---

## 🔍 Feature Extraction

```bash
python extract.py \
  --data_root pure_data \
  --out_dir cache_kps_final \
  --openvino_model yolo11n-pose_openvino_model \
  --seq_len 32 --seq_stride 1 \
  --conf 0.6 --iou 0.5 \
  --kp_conf 0.25 --min_valid_kp 10 --min_valid_frames 12 \
  --add_angles --rotate_norm --ms_strides 1,2,4
```

Outputs in `cache_kps_final/`:

* `X.npy`, `y.npy`, `mask.npy`, `label_map.json`, `scaler.joblib`

---

## 🏋️ Training

```bash
python train.py \
  --cache_dir cache_kps_final \
  --out_dir runs_maxacc_final \
  --epochs 120 --batch 128 \
  --lr 3e-4 --dropout 0.2 --val_split 0.2 \
  --channels 64,128,256 --kernel_size 5 \
  --class_weight --label_smoothing 0.1
```

Checkpoint saved at:

```
runs_maxacc_final/tcn_best_final.pt
```

---

## 🎥 Inference
---


Pretrained weights are provided in the [`weights/`](weights/) folder:  
- `tcn_best_final.pt` → best trained TCN checkpoint  
- `scaler.joblib` → feature normalizer (must be present during inference)  

You can download/clone this repo and directly run inference using the provided weights.

# 🚀 Quick Start

Clone the repository and install dependencies:

```bash
git clone https://github.com/username/Action-Detection.git
cd Action-Detection
pip install -r requirements.txt
```
**Video:**

```bash
python test.py \
  --video path/to/video.mp4 \
  --ckpt runs_maxacc_final/tcn_best_final.pt \
  --openvino_model yolo11n-pose_openvino_model \
  --draw_pose
```

**Webcam:**

```bash
python test.py \
  --webcam \
  --ckpt runs_maxacc_final/tcn_best_final.pt \
  --openvino_model yolo11n-pose_openvino_model \
  --draw_pose
```

---

## 🔄 ONNX Export

```python
import torch
from train import TCN

ckpt = torch.load("runs_maxacc_final/tcn_best_final.pt", map_location="cpu")
model = TCN(in_feat=ckpt["in_feat"],
            channels=ckpt["channels"],
            num_classes=len(ckpt["label_map"]),
            kernel_size=ckpt["kernel_size"],
            dropout=ckpt["dropout"])
model.load_state_dict(ckpt["state_dict"])
model.eval()

dummy = torch.randn(1, 32, ckpt["in_feat"])
torch.onnx.export(
    model, dummy, "action_detection.onnx",
    input_names=["input_seq"], output_names=["logits"],
    dynamic_axes={"input_seq": {0: "batch", 1: "time"},
                  "logits": {0: "batch"}},
    opset_version=17
)
```

---

## ⚙️ Notes

* Accuracy strongly depends on **dataset quality and size**.
* Default classes: `Fall`, `Lie`, `LikeFall`, `Sitting`, `Stand`, `Walking`.
* `scaler.joblib` is automatically loaded at inference to match training.

---

## 🙌 Acknowledgments

* [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
* Temporal Convolutional Networks (TCN)
