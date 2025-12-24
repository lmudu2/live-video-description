import ollama
import base64
import cv2
import time
from typing import Optional

class LLMDescriber:
    """
    Handles interaction with the local Ollama instance to describe video frames.
    Uses the 'llava' model by default.
    """
    def __init__(self, model_name: str = "llava"):
        self.model_name = model_name
        self.last_call_time = 0
        self.min_interval = 2.0 # Minimum seconds between calls to prevent lag

    def describe_frame(self, frame, prompt: str = "Describe this image in one concise sentence.") -> Optional[str]:
        """
        Sends a frame to Ollama for description.
        Returns the description string or None if skipped/failed.
        """
        current_time = time.time()
        if current_time - self.last_call_time < self.min_interval:
            return None

        try:
            # Convert frame (numpy array) to JPEG bytes
            _, buffer = cv2.imencode('.jpg', frame)
            # Ollama python client expects bytes or path
            
            response = ollama.chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': [buffer.tobytes()]
                    }
                ]
            )
            
            self.last_call_time = current_time
            return response['message']['content']

        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return f"Error: {e}"
