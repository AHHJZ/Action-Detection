# Action-Detection
End-to-end pipeline for multi-person action detection using YOLO-Pose for keypoint extraction and a Temporal Convolutional Network (TCN) for sequence classification. Supports real-time video/webcam inference, ONNX export, and safety-critical actions like fall detection.

# ⚠️ Note  
The military salute in this model is included **for fun purposes only**.

---

# 🏃 Action Detection

Multi-person **action detection** using **YOLO-Pose** for keypoint extraction and a **Temporal Convolutional Network (TCN)** for sequence classification.  
**Dataset:** custom (manually collected). With more and higher-quality data, model performance will improve noticeably.  
**Export:** the trained model can be exported to **ONNX** for deployment.

---

# 📌 Features
- YOLO-Pose (Ultralytics) to extract 17 COCO keypoints per frame.
- Temporal features: positions, multi-scale velocities/accelerations, (optional) joint-angle dynamics.
- **TCN** classifier with dilated temporal convolutions.
- Multi-person tracking via IoU assignment.
- Real-time visualization (bboxes + action label + FPS).
- Robust fall logic & heuristics (safety use-cases).

---

# 🔧 Requirements & Installation
Python **3.8+**

Install dependencies:
```bash
pip install -r requirements.txt

requirements.txt:

torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
opencv-python>=4.8.0
ultralytics>=8.0.0
numpy>=1.23.0
scikit-learn>=1.2.0


---

📂 Dataset Structure

Organize your data as class folders → sequence folders → frames (images):

pure_data/
 ├─ Fall/
 │   ├─ seq1/ frame_0001.jpg, frame_0002.jpg, ...
 │   ├─ seq2/ ...
 ├─ Lie/
 ├─ Likefall/
 ├─ Siting/
 ├─ Stand/
 └─ Walking/

Class names are used as labels.

Sequence folders must start with seq (e.g., seq1, seq_0003).



---

🔍 Feature Extraction

python extract.py \
  --data_root pure_data \
  --out_dir cache_kps_final \
  --openvino_model yolo11n-pose_openvino_model \
  --seq_len 32 --seq_stride 1 \
  --conf 0.6 --iou 0.5 \
  --kp_conf 0.25 --min_valid_kp 10 --min_valid_frames 12 \
  --add_angles --rotate_norm --ms_strides 1,2,4

Outputs in cache_kps_final/:

X.npy, y.npy, mask.npy, label_map.json, scaler.joblib


---

🏋️ Training

python train.py \
  --cache_dir cache_kps_final \
  --out_dir runs_maxacc_final \
  --epochs 120 --batch 128 \
  --lr 3e-4 --dropout 0.2 --val_split 0.2 \
  --channels 64,128,256 --kernel_size 5 \
  --class_weight --label_smoothing 0.1

Checkpoint saved at:

runs_maxacc_final/tcn_best_final.pt


---

🎥 Inference

Video:

python test.py \
  --video path/to/video.mp4 \
  --ckpt runs_maxacc_final/tcn_best_final.pt \
  --openvino_model yolo11n-pose_openvino_model \
  --draw_pose

Webcam:

python test.py \
  --webcam \
  --ckpt runs_maxacc_final/tcn_best_final.pt \
  --openvino_model yolo11n-pose_openvino_model \
  --draw_pose


---

🔄 ONNX Export

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
torch.onnx.export(model, dummy, "action_detection.onnx",
    input_names=["input_seq"], output_names=["logits"],
    dynamic_axes={"input_seq": {0: "batch", 1: "time"},
                  "logits": {0: "batch"}},
    opset_version=17)


---

⚙️ Notes

Accuracy strongly depends on dataset quality/size.

Default classes: Fall, Lie, Likefall, Siting, Stand, Walking.

scaler.joblib is automatically loaded at inference to match training.





🙌 Acknowledgments

Ultralytics YOLO

Temporal Convolutional Networks (TCN)

