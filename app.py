import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import av
import threading
import queue
import time
from processor import YOLOProcessor
from llm_describer import LLMDescriber

st.set_page_config(page_title="VideoMind AI", page_icon="👁️", layout="wide")

# -- Custom CSS for "Premium" Look --
st.markdown("""
<style>
    /* Global Styles */
    .stApp {
        background-color: #0E1117;
    }
    h1 {
        color: #FAFAFA;
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 700;
        margin-bottom: 0px;
    }
    .caption {
        font-size: 1.2rem;
        color: #A0A0A0;
        margin-bottom: 2rem;
    }
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF914D 100%);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.3);
    }
    /* Stats Card Styling */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #FF4B4B;
    }
</style>
""", unsafe_allow_html=True)

# -- Header --
st.title("👁️ VideoMind")
st.markdown('<p class="caption">Real-Time Computer Vision & Generative AI Dashboard</p>', unsafe_allow_html=True)

# -- Controls (Top Bar) --
with st.container():
    c1, c2, c3 = st.columns([1, 1, 2])
    with c1:
        conf_threshold = st.slider("🎯 Detection Sensitivity", 0.0, 1.0, 0.5, 0.05)
    with c2:
        model_name = st.text_input("🤖 AI Model", "llava", help="Make sure this model is pulled in Ollama")
    with c3:
        st.info("💡 **Tip**: Run `ollama serve` in a terminal to enable AI descriptions.")

st.divider()

# -- Processor Class --
class VideoMindProcessor(VideoProcessorBase):
    def __init__(self):
        self.yolo = YOLOProcessor(conf_threshold=0.5)
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.stats = {}

    def update_settings(self, conf):
        self.yolo.conf_threshold = conf

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        
        # Process
        annotated_img, stats = self.yolo.process_frame(img)
        
        # Update thread-safe storage
        with self.frame_lock:
            self.latest_frame = img 
            self.stats = stats
            
        return av.VideoFrame.from_ndarray(annotated_img, format="bgr24")

    def get_stats(self):
        with self.frame_lock:
            return self.stats

    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame

# -- Main Layout --
col_video, col_stats = st.columns([1.8, 1])

# 1. Video Section
with col_video:
    st.subheader("📹 Live Feed")
    ctx = webrtc_streamer(
        key="videomind-feed",
        video_processor_factory=VideoMindProcessor,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if ctx.video_processor:
        ctx.video_processor.update_settings(conf_threshold)

# 2. Analytics Section
with col_stats:
    st.subheader("📊 Analytics")
    
    # Real-time stats container
    stats_container = st.empty()
    
    st.markdown("---")
    st.subheader("🧠 AI Scene Analysis")
    
    desc_btn = st.button("✨ Describe Scene", use_container_width=True)
    desc_container = st.container()

    # Handle Button Click
    if desc_btn:
        if ctx.state.playing and ctx.video_processor:
            frame = ctx.video_processor.get_latest_frame()
            if frame is not None:
                with desc_container:
                    with st.spinner("🤖 Analyzing visual context..."):
                        llm = LLMDescriber(model_name=model_name)
                        desc = llm.describe_frame(frame)
                        
                        if desc and "Error" not in desc:
                            st.success(desc)
                        elif desc:
                            st.error(desc)
            else:
                st.warning("Wait for video stream to initialize.")
        else:
            st.warning("Please start the video stream first.")

    # Live Stats Loop
    if ctx.state.playing:
        while True:
            if ctx.video_processor:
                stats = ctx.video_processor.get_stats()
                # Create a nice metric view
                with stats_container.container():
                    if stats:
                        cols = st.columns(2)
                        for idx, (label, count) in enumerate(stats.items()):
                            cols[idx % 2].metric(label.capitalize(), count)
                    else:
                        st.info("No objects detected currently.")
            time.sleep(0.5)

