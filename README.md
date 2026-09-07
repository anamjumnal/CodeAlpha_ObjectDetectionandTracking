# 👁️ VisionFlow — Object Detection & Multi-Object Tracking

> A real-time computer vision web application built with **YOLO + Streamlit** for detecting, classifying, and tracking multiple objects across images and video frames.

## 🚀 Live Demo

🌐 **Try VisionFlow Online:**  
https://visionflow-object-detection.streamlit.app/

---

## 📌 About the Project

**VisionFlow** is a real-time Object Detection and Multi-Object Tracking application developed as part of the **CodeAlpha Artificial Intelligence Internship — Task 4**.

The application uses a pretrained **YOLO model from Ultralytics** to detect objects and combines it with modern tracking algorithms to maintain persistent IDs for objects across consecutive video frames.

The system provides an interactive web interface where users can upload images/videos or use their webcam for real-time computer vision analysis.

---

## ✨ Key Features

### 🎯 Object Detection
- Detects multiple objects in images and videos
- Draws bounding boxes around detected objects
- Displays object class labels
- Shows detection confidence scores
- Supports pretrained YOLO models

### 🔄 Multi-Object Tracking
- Tracks multiple objects across video frames
- Assigns unique tracking IDs
- Maintains object identities while they move
- Supports **ByteTrack**
- Supports **BoT-SORT**

### 🎥 Real-Time Video Processing
- Upload and process video files
- Real-time detection and tracking
- Webcam-based detection
- Frame-by-frame object analysis

### 🖼️ Image Inspector
- Upload an image for object detection
- Adjustable confidence threshold
- Adjustable IoU threshold
- Visual bounding-box results

### 📊 Analytics
- Object counts
- Tracking information
- Detection statistics
- Performance information

### 🎨 Interactive UI
- Modern multicolour interface
- Dedicated pages for different workflows
- Configurable YOLO model and tracking settings
- User-friendly Streamlit dashboard

---

## 🧠 How It Works

VisionFlow follows a simple computer vision pipeline:

```text
          Input
            │
     ┌──────┴──────┐
     │             │
   Image         Video
     │             │
     └──────┬──────┘
            ↓
       YOLO Detection
            ↓
     Bounding Boxes
            ↓
     Object Tracking
      ┌─────┴─────┐
      │           │
  ByteTrack   BoT-SORT
      │           │
      └─────┬─────┘
            ↓
     Persistent IDs
            ↓
   Analytics + Output
