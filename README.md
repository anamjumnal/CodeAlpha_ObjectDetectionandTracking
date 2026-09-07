# 👁️ VisionFlow — Object Detection & Multi-Object Tracking

> A real-time computer vision web application built with YOLO, OpenCV, and Streamlit for object detection and multi-object tracking.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://visionflow-object-detection.streamlit.app/)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 🚀 Quick Start

**Try VisionFlow Online:**  
[🌐 Live Demo](https://visionflow-object-detection.streamlit.app/)

No installation required—access VisionFlow directly through your web browser!

---

## 📌 About the Project

**VisionFlow** is an interactive object detection and multi-object tracking web application developed as part of the **CodeAlpha Artificial Intelligence Internship — Task 4**.

The application leverages pretrained **YOLO models from Ultralytics** to detect objects in images and videos, combined with advanced tracking algorithms to maintain persistent IDs for objects across consecutive video frames.

### Key Capabilities

- Object detection in images and videos
- Real-time multi-object tracking
- Live webcam detection
- Persistent object tracking IDs
- Detection and tracking analytics
- Interactive, user-friendly interface

---

## ✨ Features

### 🎯 Object Detection
- Detect multiple objects in images and videos
- Automatic bounding box visualization
- Class labels and confidence scores
- Pretrained YOLO models (YOLOv11n, YOLOv11s)
- Adjustable confidence threshold
- Adjustable IoU threshold for fine-tuning detection

### 🔄 Multi-Object Tracking
- Track multiple objects across video frames
- Persistent object ID assignment
- Maintain object identity during movement
- **ByteTrack** algorithm support
- **BoT-SORT** algorithm support

### 🎥 Real-Time Video Processing
- Upload and process video files
- Webcam-based detection and tracking
- Frame-by-frame analysis
- Real-time detection overlay

### 🖼️ Image Inspector
- Upload and analyze images
- Object detection with bounding boxes
- View class labels and confidence scores
- Adjustable detection settings

### 📊 Analytics Dashboard
- Detection statistics and summaries
- Object count tracking
- Tracking ID management
- Processing performance metrics

### 🎨 Interactive Interface
Multi-page Streamlit dashboard with dedicated sections for different computer vision workflows.

---

## 🧠 How It Works

```
┌─────────────────────────────────────────────────────────┐
│                      INPUT                              │
│                   Image / Video                         │
└─────────────────┬───────────────────────────────────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │ YOLO OBJECT         │
        │ DETECTION           │
        └─────────┬───────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │ Bounding Boxes +    │
        │ Labels + Scores     │
        └─────────┬───────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │ MULTI-OBJECT        │
        │ TRACKING            │
        └─────┬───────────┬───┘
              │           │
        ┌─────▼─┐   ┌────▼─────┐
        │ Byte  │   │ BoT-SORT  │
        │ Track │   │           │
        └─────┬─┘   └────┬─────┘
              │           │
              └─────┬─────┘
                    │
                    ▼
        ┌─────────────────────┐
        │ Persistent Object   │
        │ IDs & Tracking      │
        └─────────┬───────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │ Visualization +     │
        │ Analytics           │
        └─────────────────────┘
```

### 🔍 Object Detection Process

The YOLO model analyzes each frame and produces:
- **Bounding box coordinates** — precise object location
- **Object class** — what the object is (e.g., car, person, bus)
- **Confidence score** — detection certainty (0–1)

**Example Detection Output:**
```
Car       → Confidence: 0.91
Person    → Confidence: 0.87
Bus       → Confidence: 0.84
Bicycle   → Confidence: 0.79
```

### 🔄 Multi-Object Tracking Algorithm

Object tracking maintains object identity across frames:

```
Frame 1 → Car → ID 1
Frame 2 → Car → ID 1 (same car)
Frame 3 → Car → ID 1 (same car)
Frame 4 → Car → ID 1 (same car)
```

This allows VisionFlow to follow moving objects rather than treating every detection as new.

#### Supported Tracking Algorithms

| Algorithm | Strengths | Use Case |
|-----------|-----------|----------|
| **ByteTrack** | Efficient, real-time capable | Fast-paced scenes |
| **BoT-SORT** | Robust association, handles occlusion | Complex tracking scenarios |

---

## 🖥️ Application Pages

| Page | Description |
|------|-------------|
| **🏠 Command Center** | Main dashboard with application overview |
| **🎥 Live Detection** | Real-time webcam object detection & tracking |
| **📹 Video Lab** | Upload and process video files |
| **🖼️ Image Inspector** | Analyze images for object detection |
| **📊 Analytics** | View detection and tracking statistics |
| **ℹ️ About** | Project information and technologies |

---

## 🧠 YOLO Models

VisionFlow supports Ultralytics pretrained YOLO models:

| Model | Size | Speed | Accuracy | Resource Use |
|-------|------|-------|----------|--------------|
| **YOLOv11n** | 2.6M | ⚡⚡⚡ | ⭐⭐⭐ | Low |
| **YOLOv11s** | 12.2M | ⚡⚡ | ⭐⭐⭐⭐ | Medium |

Select your preferred model directly from the application sidebar.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **Python** | Core programming language |
| **Streamlit** | Web application framework |
| **Ultralytics YOLO** | Object detection & tracking |
| **OpenCV** | Image & video processing |
| **ByteTrack** | Multi-object tracking |
| **BoT-SORT** | Multi-object tracking |
| **Pillow** | Image processing |
| **NumPy** | Numerical computing |
| **Streamlit-WebRTC** | Real-time webcam functionality |

---

## ⚙️ Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Setup Steps

**1. Clone the Repository**
```bash
git clone https://github.com/Simran685-art/CodeAlpha_Object_Detection_Tracking.git
cd CodeAlpha_Object_Detection_Tracking
```

**2. Install Dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Application**
```bash
streamlit run app.py
```

The application will open automatically in your default web browser at `http://localhost:8501`.

### Required Dependencies

```
streamlit
opencv-python-headless
ultralytics
streamlit-webrtc
av
pillow
numpy
lap
```

---

## 📂 Project Structure

```
CodeAlpha_Object_Detection_Tracking/
├── app.py                      # Main application entry point
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
│
├── .streamlit/
│   └── config.toml            # Streamlit configuration
│
└── models/
    └── .gitkeep               # YOLO models directory
```

---

## 🌐 Deployment

VisionFlow is deployed on **Streamlit Community Cloud** for easy access without local installation.

**Live Application:** [🌐 https://visionflow-object-detection.streamlit.app/](https://visionflow-object-detection.streamlit.app/)

---

## 💡 Real-World Applications

Object detection and tracking are used in:

- 🚗 **Traffic monitoring** — Vehicle detection and counting
- 🚦 **Intelligent transportation systems** — Smart traffic flow
- 👥 **People counting** — Crowd analytics
- 🏙️ **Smart cities** — Urban monitoring
- 📹 **Video surveillance** — Security systems
- 🏭 **Industrial monitoring** — Safety and quality control
- 🏟️ **Sports analytics** — Player tracking
- 🤖 **Robotics** — Robot vision systems
- 🚘 **Autonomous systems** — Vehicle perception
- 📊 **Video analytics** — Content analysis

---

## 🔮 Future Enhancements

Planned improvements include:

- [ ] Object movement trails and trajectory visualization
- [ ] Vehicle and people counting across regions
- [ ] Object speed estimation
- [ ] Zone-based analytics and alerts
- [ ] Detection heatmaps
- [ ] Advanced analytics dashboards
- [ ] Custom-trained YOLO models
- [ ] Optimized video processing pipeline
- [ ] Cloud-based processing
- [ ] Improved occlusion handling

---

## 📸 Demo Workflow

```
1. Upload Image/Video
         ↓
2. Select YOLO Model (YOLOv11n or YOLOv11s)
         ↓
3. Set Confidence Threshold
         ↓
4. Set IoU Threshold
         ↓
5. Choose Tracking Algorithm (ByteTrack or BoT-SORT)
         ↓
6. Start Detection
         ↓
7. View Detected Objects with Bounding Boxes
         ↓
8. Objects Assigned Persistent Tracking IDs
         ↓
9. Results & Analytics Displayed
```

---

## 🔗 Links

- **Live Demo:** [🌐 VisionFlow Application](https://visionflow-object-detection.streamlit.app/)
- **GitHub Repository:** [💻 Simran685-art/CodeAlpha_Object_Detection_Tracking](https://github.com/Simran685-art/CodeAlpha_Object_Detection_Tracking)

---

## 🙏 Acknowledgments

This project uses the following exceptional open-source technologies:

- **[Ultralytics YOLO](https://github.com/ultralytics/ultralytics)** — Object detection and tracking
- **[Streamlit](https://streamlit.io/)** — Interactive web framework
- **[OpenCV](https://opencv.org/)** — Computer vision library
- **[NumPy](https://numpy.org/)** — Numerical computing
- **[Pillow](https://python-pillow.org/)** — Image processing
- **[ByteTrack](https://github.com/ifzhang/ByteTrack)** — Multi-object tracking
- **[BoT-SORT](https://github.com/NirAharon/BoT-SORT)** — Robust tracking

---

## 📜 License

This project was developed for educational purposes as part of the **CodeAlpha Artificial Intelligence Internship Program**.

---

## 👨‍💼 Project Information

| Aspect | Details |
|--------|---------|
| **Project Name** | VisionFlow — Object Detection & Multi-Object Tracking |
| **Internship** | CodeAlpha AI Internship |
| **Task** | Task 4 — Object Detection and Tracking |
| **Framework** | Streamlit |
| **Detection Model** | Ultralytics YOLO |
| **Tracking Algorithms** | ByteTrack, BoT-SORT |
| **Status** | ✅ Completed & Deployed |

---

## 🌟 See It. Detect It. Track It.

**VisionFlow** — Making object detection accessible to everyone.

[**Try it now →**](https://visionflow-object-detection.streamlit.app/)
