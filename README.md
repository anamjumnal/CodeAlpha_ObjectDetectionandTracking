# VisionFlow — CodeAlpha AI Internship Task 4

A polished object detection and tracking web application built with Streamlit, OpenCV, Ultralytics YOLO and ByteTrack/BoT-SORT.

## Features

- Real-time webcam detection
- Uploaded video detection
- Pretrained YOLO model
- Bounding boxes and labels
- Multi-object tracking
- Persistent track IDs
- Confidence / IoU controls
- ByteTrack and BoT-SORT
- Image inspection mode
- Analytics dashboard
- Multi-page UI

## 1. Install

Use Python 3.10 or 3.11.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

## 2. Run

```bash
streamlit run app.py
```

On first run, Ultralytics downloads the selected YOLO weights automatically.

## 3. Best demo

1. Open Video Lab.
2. Upload a street / traffic / people video.
3. Select `yolo11n.pt`.
4. Confidence: 0.35.
5. Tracker: `bytetrack.yaml`.
6. Click Start detection & tracking.
7. Point out the bounding boxes, labels and IDs.
8. Open Analytics.

## 4. Webcam

Open Live Detection and allow browser camera access.

If webcam mode is unavailable, make sure:

```bash
pip install -r requirements.txt
```

was completed and restart Streamlit.

## Notes for deployment

For Streamlit Community Cloud, add all packages in `requirements.txt`. Browser webcam support may require HTTPS, which Streamlit Community Cloud provides.

For a free CPU deployment, `yolo11n.pt` is recommended because it is lightweight.

## Suggested project title

VisionFlow — Real-Time Object Detection & Multi-Object Tracking System
