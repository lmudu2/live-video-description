# 👁️ VideoMind

**VideoMind** is a real-time computer vision and generative AI dashboard built with Streamlit and Ollama. It combines object detection (YOLOv8) with multimodal LLMs (LLaVA) to "see" and "understand" live video streams.

## ✨ Features

*   **Real-Time Object Detection**: Uses YOLOv8 to detect and count objects in a live webcam feed.
*   **Generative AI Scene Analysis**: Instantly describe what's happening in the frame using the **LLaVA** model running locally via Ollama.
*   **Live Analytics**: Real-time counters for detected objects (e.g., "Person: 1", "Cell Phone: 2").
*   **Thread-Safe Processing**: Optimized `streamlit-webrtc` implementation for smooth video streaming.

## 🛠️ Tech Stack

*   **Frontend**: Streamlit
*   **Computer Vision**: Ultralytics YOLOv8, OpenCV
*   **Generative AI**: Ollama (LLaVA / Llama 3 Vision)
*   **Streaming**: Streamlit WebRTC

## 🚀 Quick Start

### 1. Prerequisites
You need **Ollama** installed to run the AI features.
1.  Download [Ollama](https://ollama.com/).
2.  Pull the LLaVA model:
    ```bash
    ollama pull llava
    ```
    *(Note: You can use other multimodal models compatible with Ollama)*

### 2. Installation
Clone the repository and install dependencies.

```bash
git clone https://github.com/your-username/video-mind.git
cd video-mind
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Run the App
Start the Streamlit server.
```bash
streamlit run app.py
```

### 4. Usage
*   **Detection Sensitivity**: Adjust the slider to filter weak detections.
*   **AI Model**: Enter the name of your local Ollama model (default: `llava`).
*   **Describe Scene**: Click the button to send the current frame to the AI for a description.

## 🤝 Contributing
Feel free to open issues or submit pull requests to improve VideoMind!

## 📄 License
MIT License
