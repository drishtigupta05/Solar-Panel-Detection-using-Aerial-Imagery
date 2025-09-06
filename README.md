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
   - Evaluation based on standard object detection metrics.  

3. **Model Deployment**
   - Deployment on NVIDIA Jetson Orin Nano for optimized inference.  

4. **Inference Application**  
   - Supports real-time detection on aerial imagery (images or video streams).  
   - Outputs bounding boxes with associated confidence scores.  

***

## Requirements  

### Training Environment
- Python 3.8+  
- Ultralytics YOLOv8 

### Deployment (Jetson Orin Nano)  
- NVIDIA JetPack SDK (with CUDA, cuDNN, TensorRT)  
- OpenCV  
- Ultralytics YOLOv8  

***

## Installation  

### 1. Clone Repository  
```bash
git clone https://github.com/drishtigupta05/Solar-Panel-Detection-using-Aerial-Imagery.git
cd Solar-Panel-Detection-using-Aerial-Imagery
```

### 2. Install Dependencies  
```bash
pip install -r requirements.txt
```

### 3. Train the Model
```bash
!yolo task=detect mode=train model=yolov8m-obb.pt data=/content/Senior-Project-Ver-2-1/data.yaml epochs=100 imgsz=640
```

### 4. Export the Model for Deployment
```bash
yolo export model=best.pt format=onnx
```

## Results
- Performance tested using **Precision, Recall, and mAP** metrics.
- Inference tests illustrate readiness for near real-time use on edge hardware.
- Bounding box visualizations emphasize accurate detection of solar panels in aerial imagery.

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

**Author**: [Drishti Gupta]  
**Contact**: drigup73@example.com  
**Year**: 2025  

*** 
