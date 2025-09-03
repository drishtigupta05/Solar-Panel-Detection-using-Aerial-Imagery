# Solar-Panel-Detection-using-Aerial-Imagery

## Overview
This project introduces a system to **detect solar panels from aerial imagery** with a **YOLOv8**-based custom-trained deep learning model. The trained model has been optimized and deployed on an **NVIDIA Jetson Orin Nano** for real-time inference on edge devices.

The goal is to offer a cost-effective and scalable platform for monitoring and mapping solar panel installations that can be utilized for renewable energy research, infrastructure evaluation, and smart city planning.

***

## Features
- Solar panel object detection in aerial and satellite images with YOLOv8.
- Real-time inference on Jetson Orin Nano with TensorRT optimization.
- Training and preparation pipeline for bespoke aerial datasets.
- Low-weight and power-efficient deployment for UAVs, drones, or fixed monitoring posts.
- Scalable design that can be used to recognize other objects in aerial images.

***

## System Architecture

1. **Dataset Preparation**
   - Aerial images with ground truth bounding boxes for solar panels.
- Conversion of annotations to YOLO format.  

2. **Model Training (YOLOv8)**  
   - Model trained on the prepared dataset using Ultralytics YOLOv8.  
   - Data augmentation techniques applied (flipping, scaling, rotation).  
   - Evaluation based on standard object detection metrics.  

3. **Model Export and Deployment**
- Trained model exported to ONNX and converted to a TensorRT engine.  
   - Deployment on NVIDIA Jetson Orin Nano for optimized inference.  

4. **Inference Application**  
   - Supports real-time detection on aerial imagery (images or video streams).  
   - Outputs bounding boxes with associated confidence scores.  

***

## Requirements  

### Training Environment
- Python 3.8+  
- PyTorch with CUDA support  
- Ultralytics YOLOv8  
- OpenCV  
- CUDA and cuDNN  

### Deployment (Jetson Orin Nano)  
- NVIDIA JetPack SDK (with CUDA, cuDNN, TensorRT)  
- ONNX Runtime / TensorRT  
- OpenCV  
- Ultralytics YOLOv8  

***

## Installation  

### 1. Clone Repository  
```bash
git clone https://github.com/your-username/solar-panel-detection-yolov8.git
cd solar-panel-detection-yolov8
```

### 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
yolo detect train data=data.yaml model=yolov8n.pt epochs=50 imgsz=640
```

### 4. Export the Model for Deployment
```bash
yolo export model=best.pt format=onnx
```

### 5. Run Inference on Jetson Orin Nano
```bash
python jetson_infer.py --model best.onnx --input test_images/
```

***

## Results
- Performance tested using **Precision, Recall, and mAP** metrics.
- Inference tests illustrate readiness for near real-time use on edge hardware.
- Bounding box visualizations emphasize accurate detection of solar panels in aerial imagery.

***

## Project Structure

```
solar-panel-detection-yolov8/
 ┣ dataset/
 ┃ ┣ images/
 ┃ ┣ labels/
 ┣ models/
 ┃ ┣ best.pt
 ┣ inference_results/
 ┣ jetson_infer.py
 ┣ train.yaml
 ┣ requirements.txt
 ┗ README.md
```

***

## Applications
- Automated mapping of solar installations.
- Infrastructure monitoring and renewable energy analysis.
- Drone-based aerial surveys for solar farm inspection.
- Rural and urban energy planning.

***

## Future Improvements
- Extension to **instance segmentation** for better solar panel area estimation.  
- Integration with **geospatial mapping tools (QGIS, Google Earth Engine)**.  
- Deployment optimization for UAVs and real-time video feed processing.

***

## Acknowledgements
- Ultralytics YOLOv8 framework.  
- NVIDIA Jetson Orin Nano ecosystem.
- Community-developed open-source contributions to aerial imagery databases.

*** 

**Author**: [Your Name]  
**Contact**: your.email@example.com  
**Year**: 2025  

*** 

Would you prefer I structure this in an **academic-report style (with sections like Objectives, Methodology, Results, and Conclusion)** so it can be directly inserted into an academic submission?
