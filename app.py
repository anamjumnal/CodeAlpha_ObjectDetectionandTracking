import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Optional webcam support
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except Exception:
    WEBRTC_AVAILABLE = False


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="VisionFlow — Object Detection & Tracking",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# THEME / UI
# =========================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 5% 5%, rgba(255, 20, 147, .22), transparent 25%),
        radial-gradient(circle at 95% 8%, rgba(0, 210, 255, .20), transparent 23%),
        radial-gradient(circle at 80% 92%, rgba(255, 190, 30, .22), transparent 25%),
        radial-gradient(circle at 10% 90%, rgba(124, 58, 237, .16), transparent 24%),
        #fff7fc;
    color: #32152f;
}

section[data-testid="stSidebar"] {
    background:
        linear-gradient(180deg, #ffd1ea 0%, #ffeaf6 42%, #e7ddff 100%);
    border-right: 2px solid #ff5cac;
}

.block-container {
    padding-top: 1.2rem;
    max-width: 1450px;
}

.hero {
    padding: 30px 34px;
    border-radius: 30px;
    background:
        linear-gradient(
            115deg,
            #ff4fa3 0%,
            #ff78bd 25%,
            #9b5cff 52%,
            #20c9e8 76%,
            #ffd43b 100%
        );
    border: 3px solid rgba(255,255,255,.8);
    box-shadow: 0 18px 50px rgba(236, 72, 153, .25);
    color: white;
}

.hero h1 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(2.2rem, 4vw, 4rem);
    margin: 4px 0 0;
    letter-spacing: -2px;
    color: white;
}

.hero p {
    color: rgba(255,255,255,.94);
    font-size: 1.05rem;
    margin-top: 10px;
}

.badge {
    display: inline-block;
    padding: 7px 13px;
    border-radius: 999px;
    background: #fff3a6;
    color: #7a285d;
    border: 2px solid white;
    font-size: .82rem;
    font-weight: 800;
}

.card {
    padding: 21px;
    border-radius: 23px;
    background: rgba(255,255,255,.92);
    border: 2px solid #ffb4d9;
    box-shadow: 0 10px 30px rgba(236,72,153,.13);
    color: #3b1838;
}

.metric {
    padding: 18px;
    border-radius: 19px;
    background:
        linear-gradient(
            135deg,
            #fff 0%,
            #ffe0f1 48%,
            #e0faff 100%
        );
    border: 2px solid #ff9dce;
    box-shadow: 0 8px 22px rgba(236,72,153,.10);
}

.metric .label {
    color: #8b4776;
    font-size: .82rem;
}

.metric .value {
    font-size: 1.7rem;
    font-weight: 700;
    margin-top: 4px;
    color: #d92d86;
}

.section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.45rem;
    font-weight: 700;
    margin: 25px 0 12px;
    color: #d42b87;
}

.stButton > button {
    border-radius: 15px !important;
    border: 0 !important;
    font-weight: 800 !important;
    background:
        linear-gradient(
            90deg,
            #ff3f9f,
            #9b5cff,
            #00cfe8,
            #ffc928
        ) !important;
    color: white !important;
    box-shadow: 0 7px 20px rgba(236,72,153,.25);
}

div[data-testid="stFileUploader"] {
    background:
        linear-gradient(
            135deg,
            #fff0f8,
            #e9faff,
            #fff7d1
        );
    border: 2px dashed #ff63ad;
    border-radius: 20px;
    padding: 12px;
}

div[data-testid="stFileUploader"] section {
    background: transparent !important;
}

.stRadio label,
.stSelectbox label,
.stSlider label {
    color: #6d2858 !important;
    font-weight: 700 !important;
}

.stAlert {
    border-radius: 16px !important;
}

[data-testid="stMetric"] {
    background:
        linear-gradient(
            135deg,
            #fff,
            #ffe4f3
        );
    border: 2px solid #ffacd5;
    padding: 12px;
    border-radius: 18px;
}

.small {
    color: #7e5271;
    font-size: .86rem;
}

footer {
    visibility: hidden;
}


/* Sidebar / top chrome */

header[data-testid="stHeader"] {
    background: transparent !important;
}

header[data-testid="stHeader"] * {
    color: #7a285d !important;
}

[data-testid="stAppViewContainer"] {
    background: transparent !important;
}

[data-testid="stSidebar"] * {
    color: #5b1745 !important;
}

section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #5b1745 !important;
}

section[data-testid="stSidebar"] .stCaption,
section[data-testid="stSidebar"] small {
    color: #7a3b67 !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] label {
    background: rgba(255,255,255,.58) !important;
    border: 2px solid rgba(255,92,172,.28) !important;
    border-radius: 12px !important;
    padding: 7px 10px !important;
    margin: 4px 0 !important;
}

section[data-testid="stSidebar"] [role="radiogroup"] label:hover {
    background: #ffd2e9 !important;
}

section[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,.82) !important;
    color: #5b1745 !important;
    border: 2px solid #ff9dce !important;
}

section[data-testid="stSidebar"] [data-baseweb="slider"] {
    color: #ff3f9f !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,63,159,.30) !important;
}

[data-testid="stDecoration"] {
    background:
        linear-gradient(
            90deg,
            #ff4fa3,
            #9b5cff,
            #20c9e8,
            #ffd43b
        ) !important;
    height: 4px !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODEL
# =========================================================

MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL = "yolo11n.pt"


@st.cache_resource(show_spinner="Loading the computer-vision model…")
def load_model(model_name: str):
    return YOLO(model_name)


# =========================================================
# DETECTION HELPERS
# =========================================================

def draw_results(frame, result):
    annotated = result.plot(
        conf=True,
        labels=True,
        boxes=True
    )
    return annotated


def result_stats(result):

    count = 0
    classes = {}
    ids = set()

    if result.boxes is not None:

        count = len(result.boxes)

        if result.boxes.cls is not None:

            for cls in result.boxes.cls.tolist():

                name = result.names.get(
                    int(cls),
                    str(int(cls))
                )

                classes[name] = classes.get(name, 0) + 1

        if result.boxes.id is not None:

            ids = {
                int(x)
                for x in result.boxes.id.tolist()
            }

    return count, classes, ids


# =========================================================
# SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown(
        '<div style="font-size:1.65rem;font-weight:800;'
        'color:#d42b87 !important;margin-bottom:2px;">'
        '👁️ VisionFlow'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div style="color:#7a3b67 !important;'
        'font-weight:600;font-size:.86rem;">'
        'Object Detection + Multi-Object Tracking'
        '</div>',
        unsafe_allow_html=True
    )

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Command Center",
            "🎥 Live Detection",
            "📹 Video Lab",
            "🖼️ Image Inspector",
            "📊 Analytics",
            "ℹ️ About"
        ],
        label_visibility="collapsed",
    )

    st.divider()

    st.markdown("### Model")

    model_choice = st.selectbox(
        "YOLO model",
        [
            "yolo11n.pt",
            "yolo11s.pt"
        ],
        index=0
    )

    conf = st.slider(
        "Confidence threshold",
        0.10,
        0.95,
        0.35,
        0.05
    )

    iou = st.slider(
        "IoU threshold",
        0.10,
        0.95,
        0.50,
        0.05
    )

    tracker = st.selectbox(
        "Tracker",
        [
            "bytetrack.yaml",
            "botsort.yaml"
        ]
    )

    st.divider()


# =========================================================
# COMMAND CENTER
# =========================================================

if page == "🏠 Command Center":

    st.markdown("""
    <div class="hero">
        <span class="badge">REAL-TIME COMPUTER VISION</span>
        <h1>VisionFlow</h1>
        <p>
            A polished real-time computer-vision workspace
            for detecting, tracking and counting objects
            with persistent IDs.
        </p>
    </div>
    """, unsafe_allow_html=True)


    st.markdown(
        '<div class="section-title">'
        'What this application does'
        '</div>',
        unsafe_allow_html=True
    )


    c1, c2, c3, c4 = st.columns(4)


    cards = [

        (
            "🎯",
            "Detect",
            "Find objects using a pretrained YOLO model.",
            "#ff4fa3",
            "#ffd1ea"
        ),

        (
            "🧭",
            "Track",
            "Assign persistent IDs with ByteTrack / BoT-SORT.",
            "#7c4dff",
            "#ddd0ff"
        ),

        (
            "🔢",
            "Count",
            "Monitor how many objects are visible in each frame.",
            "#00a9d6",
            "#c9f7ff"
        ),

        (
            "📈",
            "Analyze",
            "Inspect classes, confidence and tracking activity.",
            "#f29b00",
            "#fff0bd"
        )

    ]


    # FIXED CARD HTML
    # No nested indentation that Streamlit can interpret as code.

    for col, (
        icon,
        title,
        text,
        accent,
        bg
    ) in zip(
        [c1, c2, c3, c4],
        cards
    ):

        with col:

            card_html = (
                f'<div style="'
                f'padding:20px;'
                f'border-radius:23px;'
                f'background:linear-gradient(135deg,{bg},#ffffff);'
                f'border:3px solid {accent};'
                f'box-shadow:0 10px 25px rgba(0,0,0,.08);'
                f'min-height:145px;">'

                f'<div style="font-size:2rem;">'
                f'{icon}'
                f'</div>'

                f'<h3 style="'
                f'margin:7px 0;'
                f'color:{accent};'
                f'">'
                f'{title}'
                f'</h3>'

                f'<div style="color:#633b58;">'
                f'{text}'
                f'</div>'

                f'</div>'
            )

            st.markdown(
                card_html,
                unsafe_allow_html=True
            )


    st.markdown(
        '<div class="section-title">'
        'Recommended demo flow'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown("""
    <div class="card">

    <b>1.</b>
    Open <b>Video Lab</b> and upload a traffic /
    people / street video.

    <br><br>

    <b>2.</b>
    Keep confidence around <b>0.35</b> and select
    <b>ByteTrack</b> for a clean demo.

    <br><br>

    <b>3.</b>
    Start processing and show the annotated
    detection + tracking output.

    <br><br>

    <b>4.</b>
    Open <b>Analytics</b> to explain detection
    counts and class distribution.

    </div>
    """, unsafe_allow_html=True)


    st.info(
        "Tip: For the strongest demo, use a video "
        "containing multiple moving objects so "
        "the persistent IDs are clearly visible."
    )


# =========================================================
# LIVE DETECTION
# =========================================================

elif page == "🎥 Live Detection":

    st.markdown(
        '<div class="hero">'
        '<span class="badge">LIVE CAMERA</span>'
        '<h1>Live Detection</h1>'
        '<p>'
        'Real-time webcam detection and multi-object tracking.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )


    if not WEBRTC_AVAILABLE:

        st.error(
            "Webcam mode needs streamlit-webrtc. "
            "Install the requirements from requirements.txt "
            "and restart the app."
        )

        st.code(
            "pip install -r requirements.txt"
        )


    else:

        st.markdown("### Camera stream")

        st.caption(
            "Allow browser camera permission when prompted. "
            "Detection and tracking run on incoming frames."
        )

        model = load_model(
            model_choice
        )


        class VideoProcessor:

            def __init__(self):

                self.model = model
                self.frame_count = 0
                self.last_count = 0
                self.last_ids = set()


            def recv(self, frame):

                img = frame.to_ndarray(
                    format="bgr24"
                )


                results = self.model.track(
                    img,
                    persist=True,
                    conf=conf,
                    iou=iou,
                    tracker=tracker,
                    verbose=False,
                )


                result = results[0]


                annotated = draw_results(
                    img,
                    result
                )


                (
                    self.last_count,
                    _,
                    self.last_ids
                ) = result_stats(result)


                self.frame_count += 1


                return av.VideoFrame.from_ndarray(
                    annotated,
                    format="bgr24"
                )


        rtc_config = RTCConfiguration(
            {
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.l.google.com:19302"
                        ]
                    }
                ]
            }
        )


        webrtc_streamer(

            key="visionflow-live",

            mode=WebRtcMode.SENDRECV,

            rtc_configuration=rtc_config,

            video_processor_factory=VideoProcessor,

            media_stream_constraints={
                "video": True,
                "audio": False
            },

            async_processing=True,
        )


# =========================================================
# VIDEO LAB
# =========================================================

elif page == "📹 Video Lab":

    st.markdown(
        '<div class="hero">'
        '<span class="badge">VIDEO ANALYSIS</span>'
        '<h1>Video Lab</h1>'
        '<p>'
        'Upload a video and process every frame '
        'through detection + tracking.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )


    uploaded = st.file_uploader(
        "Upload a video",
        type=[
            "mp4",
            "avi",
            "mov",
            "mkv",
            "webm"
        ]
    )


    if uploaded:

        # -------------------------------------------------
        # Save uploaded video
        # -------------------------------------------------

        input_path = Path(
            "temp_input.mp4"
        )

        output_path = Path(
            "temp_detected.mp4"
        )


        input_path.write_bytes(
            uploaded.getbuffer()
        )


        # -------------------------------------------------
        # Read video
        # -------------------------------------------------

        cap = cv2.VideoCapture(
            str(input_path)
        )


        total = int(
            cap.get(
                cv2.CAP_PROP_FRAME_COUNT
            )
        ) or 0


        fps = cap.get(
            cv2.CAP_PROP_FPS
        )


        if not fps or fps <= 0:
            fps = 25.0


        width = int(
            cap.get(
                cv2.CAP_PROP_FRAME_WIDTH
            )
        ) or 640


        height = int(
            cap.get(
                cv2.CAP_PROP_FRAME_HEIGHT
            )
        ) or 480


        duration = (
            total / fps
            if total > 0
            else 0
        )


        # -------------------------------------------------
        # Source information
        # -------------------------------------------------

        st.markdown("### Source")


        m1, m2, m3, m4 = st.columns(4)


        with m1:

            st.markdown(
                f'<div class="metric">'
                f'<div class="label">Resolution</div>'
                f'<div class="value">'
                f'{width} × {height}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )


        with m2:

            st.markdown(
                f'<div class="metric">'
                f'<div class="label">FPS</div>'
                f'<div class="value">'
                f'{fps:.1f}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )


        with m3:

            st.markdown(
                f'<div class="metric">'
                f'<div class="label">Frames</div>'
                f'<div class="value">'
                f'{total:,}'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )


        with m4:

            st.markdown(
                f'<div class="metric">'
                f'<div class="label">Duration</div>'
                f'<div class="value">'
                f'{duration:.1f}s'
                f'</div>'
                f'</div>',
                unsafe_allow_html=True
            )


        # -------------------------------------------------
        # Processing
        # -------------------------------------------------

        st.markdown("### Processing")


        start = st.button(
            "▶ Start detection & tracking",
            type="primary",
            width="stretch"
        )


        if start:

            model = load_model(
                model_choice
            )


            # -------------------------------------------------
            # Create output video
            # -------------------------------------------------

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )


            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )


            if not writer.isOpened():

                cap.release()

                st.error(
                    "Could not create the output video."
                )

                st.stop()


            # -------------------------------------------------
            # Display containers
            # -------------------------------------------------

            frame_box = st.empty()

            progress = st.progress(0)

            status = st.empty()

            stats_box = st.empty()


            # -------------------------------------------------
            # Analytics
            # -------------------------------------------------

            class_totals = {}

            unique_ids = set()

            processed = 0

            t0 = time.time()


            # -------------------------------------------------
            # Process frames
            # -------------------------------------------------

            while True:

                ok, frame = cap.read()


                if not ok:
                    break


                results = model.track(
                    frame,
                    persist=True,
                    conf=conf,
                    iou=iou,
                    tracker=tracker,
                    verbose=False,
                )


                result = results[0]


                annotated = draw_results(
                    frame,
                    result
                )


                count, classes, ids = result_stats(
                    result
                )


                # Class totals

                for k, v in classes.items():

                    class_totals[k] = (
                        class_totals.get(k, 0)
                        + v
                    )


                # Unique tracking IDs

                unique_ids.update(
                    ids
                )


                # Write processed frame

                writer.write(
                    annotated
                )


                # Show current detected frame

                frame_box.image(
                    cv2.cvtColor(
                        annotated,
                        cv2.COLOR_BGR2RGB
                    ),
                    channels="RGB",
                    use_container_width=True
                )


                processed += 1


                # Progress

                elapsed = max(
                    time.time() - t0,
                    0.001
                )


                speed = (
                    processed / elapsed
                )


                pct = (
                    min(
                        processed / total,
                        1.0
                    )
                    if total
                    else 0
                )


                progress.progress(
                    pct
                )


                status.markdown(
                    f"**Frame {processed:,} / "
                    f"{total:,}** · "
                    f"{speed:.1f} FPS · "
                    f"**Visible:** {count} · "
                    f"**Unique IDs:** "
                    f"{len(unique_ids)}"
                )


                stats_box.json(
                    {
                        "visible_objects": count,
                        "active_track_ids":
                            sorted(ids),
                        "class_counts":
                            classes
                    }
                )


            # -------------------------------------------------
            # Finish
            # -------------------------------------------------

            cap.release()

            writer.release()


            elapsed_total = (
                time.time() - t0
            )


            # -------------------------------------------------
            # Save analytics
            # -------------------------------------------------

            st.session_state[
                "last_analytics"
            ] = {

                "class_totals":
                    class_totals,

                "unique_ids":
                    len(unique_ids),

                "frames":
                    processed,

                "elapsed":
                    elapsed_total,

            }


            progress.progress(
                1.0
            )


            st.success(
                f"Completed. Processed "
                f"{processed:,} frames with "
                f"{len(unique_ids)} unique "
                f"track IDs."
            )


            # -------------------------------------------------
            # FINAL DETECTED VIDEO PLAYER
            # -------------------------------------------------

            if (
                output_path.exists()
                and output_path.stat().st_size > 0
            ):

                st.markdown(
                    "### ▶ Detected & Tracked Video"
                )


                video_bytes = (
                    output_path.read_bytes()
                )


                st.video(
                    video_bytes,
                    format="video/mp4"
                )


            else:

                st.error(
                    "The processed video "
                    "could not be generated."
                )


    else:

        st.markdown(
            '<div class="card">'
            '<b>Supported:</b> MP4, AVI, MOV, MKV and WEBM.'
            '<br>'
            '<span class="small">'
            'For the demo, choose a clip with several '
            'moving objects.'
            '</span>'
            '</div>',
            unsafe_allow_html=True
        )


# =========================================================
# IMAGE INSPECTOR
# =========================================================

elif page == "🖼️ Image Inspector":

    st.markdown(
        '<div class="hero">'
        '<span class="badge">SINGLE FRAME</span>'
        '<h1>Image Inspector</h1>'
        '<p>'
        'Quickly validate the detector before '
        'running a full video.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )


    image_file = st.file_uploader(
        "Upload an image",
        type=[
            "jpg",
            "jpeg",
            "png",
            "webp"
        ]
    )


    if image_file:

        image = Image.open(
            image_file
        ).convert("RGB")


        model = load_model(
            model_choice
        )


        arr = np.array(
            image
        )


        result = model(
            arr,
            conf=conf,
            iou=iou,
            verbose=False
        )[0]


        annotated = result.plot(
            conf=True,
            labels=True,
            boxes=True
        )


        count, classes, ids = result_stats(
            result
        )


        a, b = st.columns(2)


        with a:

            st.image(
                image,
                caption="Original",
                use_container_width=True
            )


        with b:

            st.image(
                cv2.cvtColor(
                    annotated,
                    cv2.COLOR_BGR2RGB
                ),
                caption="Detected objects",
                use_container_width=True
            )


        x, y, z = st.columns(3)


        x.metric(
            "Objects",
            count
        )


        y.metric(
            "Classes",
            len(classes)
        )


        z.metric(
            "Confidence threshold",
            f"{conf:.2f}"
        )


        st.json(
            classes
        )


# =========================================================
# ANALYTICS
# =========================================================

elif page == "📊 Analytics":

    st.markdown(
        '<div class="hero">'
        '<span class="badge">INSIGHTS</span>'
        '<h1>Analytics</h1>'
        '<p>'
        'Turn the detection stream into a clean story '
        'for your internship demo.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )


    data = st.session_state.get(
        "last_analytics"
    )


    if not data:

        st.warning(
            "Run a video from Video Lab first. "
            "Analytics from the latest run "
            "will appear here."
        )


    else:

        a, b, c = st.columns(3)


        a.metric(
            "Frames processed",
            f"{data['frames']:,}"
        )


        b.metric(
            "Unique track IDs",
            data["unique_ids"]
        )


        c.metric(
            "Processing time",
            f"{data['elapsed']:.1f}s"
        )


        st.markdown(
            "### Class activity"
        )


        items = sorted(
            data["class_totals"].items(),
            key=lambda x: x[1],
            reverse=True
        )


        if items:

            names = [
                x[0]
                for x in items
            ]


            vals = [
                x[1]
                for x in items
            ]


            chart_data = {
                "Object class": names,
                "Detections across frames": vals
            }


            st.bar_chart(
                chart_data,
                x="Object class",
                y="Detections across frames"
            )


        else:

            st.info(
                "No object classes were detected "
                "in the last run."
            )


# =========================================================
# ABOUT
# =========================================================

else:

    st.markdown(
        '<div class="hero">'
        '<span class="badge">PROJECT DETAILS</span>'
        '<h1>About VisionFlow</h1>'
        '<p>'
        'A real-time computer-vision workspace '
        'for detection, tracking and analytics.'
        '</p>'
        '</div>',
        unsafe_allow_html=True
    )


    st.markdown("""
    <div class="card">

    <h3>Technology stack</h3>

    <ul>

        <li>
            <b>Python + Streamlit</b>
            — web application UI
        </li>

        <li>
            <b>OpenCV</b>
            — video capture and frame processing
        </li>

        <li>
            <b>Ultralytics YOLO</b>
            — pretrained real-time object detector
        </li>

        <li>
            <b>ByteTrack / BoT-SORT</b>
            — multi-object tracking with persistent IDs
        </li>

        <li>
            <b>streamlit-webrtc</b>
            — browser webcam streaming
        </li>

    </ul>

    </div>
    """, unsafe_allow_html=True)
