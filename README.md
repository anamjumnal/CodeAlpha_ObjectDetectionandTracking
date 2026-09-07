# 👁️ VisionFlow — Object Detection & Multi-Object Tracking

> A real-time computer vision web application built with YOLO, OpenCV and Streamlit for object detection and multi-object tracking.

## 🚀 Live Demo

🌐 **Try VisionFlow Online:**  
https://visionflow-object-detection.streamlit.app/

---

## 📌 About the Project

**VisionFlow** is an interactive Object Detection and Multi-Object Tracking web application developed as part of the **CodeAlpha Artificial Intelligence Internship — Task 4**.

The application uses pretrained **YOLO models from Ultralytics** to detect objects in images and videos and combines object detection with tracking algorithms to maintain persistent IDs for objects across consecutive video frames.

VisionFlow provides an easy-to-use interface for:

- Object detection in images
- Object detection in videos
- Multi-object tracking
- Real-time webcam detection
- Persistent object IDs
- Detection and tracking analytics

---

## ✨ Features

### 🎯 Object Detection

- Detects multiple objects in images and videos
- Draws bounding boxes around detected objects
- Displays object class labels
- Displays confidence scores
- Uses pretrained YOLO models
- Adjustable confidence threshold
- Adjustable IoU threshold

### 🔄 Multi-Object Tracking

- Tracks multiple objects across consecutive frames
- Assigns persistent IDs to detected objects
- Maintains object identity while objects move
- Supports ByteTrack
- Supports BoT-SORT

### 🎥 Real-Time Video Processing

- Upload video files
- Process video frames using YOLO
- Detect objects in moving scenes
- Track objects throughout the video
- Webcam-based detection and tracking

### 🖼️ Image Inspector

Upload an image and analyze it using the selected YOLO model.

The Image Inspector provides:

- Object detection
- Bounding boxes
- Class labels
- Confidence scores
- Adjustable detection settings

### 📊 Analytics

The Analytics section provides information about detection and tracking results, including:

- Detected objects
- Object counts
- Tracking IDs
- Detection statistics
- Processing information

### 🎨 Interactive Dashboard

VisionFlow includes a multi-page Streamlit interface with dedicated sections for different computer vision workflows.

---

## 🧠 How VisionFlow Works

```text
                         INPUT
                           │
              ┌────────────┴────────────┐
              │                         │
            IMAGE                     VIDEO
              │                         │
              └────────────┬────────────┘
                           │
                           ▼
                  YOLO OBJECT DETECTION
                           │
                           ▼
                BOUNDING BOXES + LABELS
                           │
                           ▼
                  MULTI-OBJECT TRACKING
                           │
                    ┌──────┴──────┐
                    │             │
                ByteTrack     BoT-SORT
                    │             │
                    └──────┬──────┘
                           │
                           ▼
                 PERSISTENT OBJECT IDs
                           │
                           ▼
                 VISUALIZATION + ANALYTICS
🔍 Object Detection

For every input image or video frame, the YOLO model identifies objects and produces:

Bounding box coordinates
Object class
Confidence score

Example:

Car       → Confidence: 0.91
Person    → Confidence: 0.87
Bus       → Confidence: 0.84
Bicycle   → Confidence: 0.79

The detected objects are displayed with bounding boxes and labels.

🔄 Multi-Object Tracking

Object tracking allows VisionFlow to maintain the identity of objects across multiple frames.

Example:

Frame 1 → Car → ID 1
Frame 2 → Car → ID 1
Frame 3 → Car → ID 1
Frame 4 → Car → ID 1

This allows the system to follow moving objects instead of treating every detection as a completely new object.

🚦 Tracking Algorithms
ByteTrack

ByteTrack is an efficient multi-object tracking algorithm suitable for real-time applications.

BoT-SORT

BoT-SORT is another multi-object tracking algorithm designed for robust object association across video frames, including situations involving movement and partial occlusion.

Both trackers can be selected directly from the VisionFlow interface.

🧠 YOLO Models

VisionFlow supports pretrained YOLO models including:

yolo11n.pt
yolo11s.pt
YOLO11n

A lightweight model suitable for faster inference and systems with limited computational resources.

YOLO11s

A larger model that provides improved detection capability while requiring more computational resources.

The model can be selected directly from the application sidebar.

🖥️ Application Pages
🏠 Command Center

The main VisionFlow dashboard providing an overview of the application and available computer vision tools.

🎥 Live Detection

Use a webcam for real-time object detection and tracking.

📹 Video Lab

Upload a video and process it using YOLO object detection and multi-object tracking.

🖼️ Image Inspector

Upload an image and detect objects using the selected YOLO model.

📊 Analytics

View detection and tracking statistics generated during processing.

ℹ️ About

Provides information about VisionFlow, its technologies and its development purpose.

🛠️ Tech Stack
Technology	Purpose
Python	Core programming language
Streamlit	Web application framework
Ultralytics YOLO	Object detection and tracking
OpenCV	Image and video processing
ByteTrack	Multi-object tracking
BoT-SORT	Multi-object tracking
Pillow	Image processing
NumPy	Numerical operations
Streamlit-WebRTC	Webcam and real-time video
Streamlit Community Cloud	Deployment
📂 Project Structure
CodeAlpha_Object_Detection_Tracking/
│
├── app.py
├── requirements.txt
│
├── .streamlit/
│   └── config.toml
│
├── models/
│   └── .gitkeep
│
└── README.md
⚙️ Installation
1. Clone the Repository
git clone https://github.com/Simran685-art/CodeAlpha_Object_Detection_Tracking.git
2. Enter the Project Directory
cd CodeAlpha_Object_Detection_Tracking
3. Install Dependencies
pip install -r requirements.txt
4. Run the Application
streamlit run app.py

The application will open in your browser.

📦 Requirements

The main dependencies used by VisionFlow are:

streamlit
opencv-python-headless
ultralytics
streamlit-webrtc
av
pillow
numpy
lap
🌐 Deployment

VisionFlow is deployed using Streamlit Community Cloud.

Live Application

https://visionflow-object-detection.streamlit.app/

The application can be accessed directly through a web browser without requiring local installation.

🎯 CodeAlpha Internship Task

This project was developed for the:

CodeAlpha Artificial Intelligence Internship

Task 4 — Object Detection and Tracking

The project demonstrates:

Real-time video input
Pretrained object detection
Bounding boxes
Object labels
Multi-object tracking
Persistent tracking IDs
Real-time visualization
💡 Real-World Applications

Object detection and multi-object tracking can be used in:

🚗 Traffic monitoring
🚦 Intelligent transportation systems
👥 People counting
🏙️ Smart city applications
📹 Video surveillance
🏭 Industrial monitoring
🏟️ Sports analytics
🤖 Robotics
🚘 Autonomous systems
📊 Video analytics
🔮 Future Improvements

Possible future enhancements include:

Object movement trails
Vehicle counting
People counting
Object speed estimation
Region and zone-based analytics
Detection heatmaps
Advanced analytics dashboards
Custom-trained YOLO models
Improved video processing
Cloud-based video processing
Improved tracking during heavy occlusion
📸 Demo Workflow
Upload Image / Video
        ↓
Select YOLO Model
        ↓
Set Confidence Threshold
        ↓
Set IoU Threshold
        ↓
Select Tracking Algorithm
        ↓
Start Detection
        ↓
Objects Detected
        ↓
Objects Assigned Tracking IDs
        ↓
Results Displayed
        ↓
Analytics Generated
🔗 Project Links
🌐 Live Demo

https://visionflow-object-detection.streamlit.app/

💻 GitHub Repository

https://github.com/Simran685-art/CodeAlpha_Object_Detection_Tracking

👩‍💻 Project Information

Project: VisionFlow — Object Detection & Multi-Object Tracking

Internship: CodeAlpha Artificial Intelligence Internship

Task: Task 4 — Object Detection and Tracking

Framework: Streamlit

Detection Model: Ultralytics YOLO

Tracking Algorithms: ByteTrack / BoT-SORT

🙏 Acknowledgements

This project uses the following open-source technologies:

Ultralytics — YOLO object detection and tracking
Streamlit — Interactive web application framework
OpenCV — Computer vision and video processing
NumPy — Numerical computing
Pillow — Image processing
Streamlit-WebRTC — Real-time webcam functionality
📜 License

This project was developed for educational and internship purposes as part of the CodeAlpha Artificial Intelligence Internship Program.

⭐ VisionFlow

See it. Detect it. Track it.

🌐 https://visionflow-object-detection.streamlit.app/
