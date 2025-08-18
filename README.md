# Action-Detection
End-to-end pipeline for multi-person action detection using YOLO-Pose for keypoint extraction and a Temporal Convolutional Network (TCN) for sequence classification. Supports real-time video/webcam inference, ONNX export, and safety-critical actions like fall detection.


# 🏃‍♂️ Action-Detection
End-to-end pipeline for **multi-person action detection** using **YOLO-Pose** for keypoint extraction and a **Temporal Convolutional Network (TCN)** for sequence classification.  
Supports real-time video/webcam inference, ONNX export, and safety-critical actions like **fall detection**.

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
