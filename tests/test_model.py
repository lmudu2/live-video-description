import sys
import os

# Add parent directory (video_mind) to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from processor import YOLOProcessor
import cv2
import numpy as np

def test_yolo_processor():
    print("Initializing YOLO Processor...")
    try:
        processor = YOLOProcessor(model_path="yolov8n.pt")
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create dummy black image [480, 640, 3]
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    print("Processing dummy frame...")
    try:
        annotated, stats = processor.process_frame(dummy_frame)
        print(f"Processing successful. Stats: {stats}")
        print(f"Output shape: {annotated.shape}")
    except Exception as e:
        print(f"Processing failed: {e}")

if __name__ == "__main__":
    test_yolo_processor()
