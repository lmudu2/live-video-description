from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter

class YOLOProcessor:
    def __init__(self, model_path="yolov8n.pt", conf_threshold=0.5):
        """
        Initialize YOLOv8 model.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.last_stats = {}

    def process_frame(self, frame):
        """
        Process a single video frame:
        1. Detect objects
        2. Draw bounding boxes
        3. Return annotated frame and count statistics
        """
        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        result = results[0] # Single frame result or result

        # Plot detections (returns BGR numpy array)
        annotated_frame = result.plot()

        # Count objects
        classes = result.boxes.cls.cpu().numpy()
        names = result.names
        counts = Counter([names[int(cls)] for cls in classes])
        
        self.last_stats = dict(counts)

        return annotated_frame, self.last_stats

    def get_stats(self):
        return self.last_stats
