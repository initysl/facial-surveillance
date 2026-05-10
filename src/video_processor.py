import cv2
import numpy as np
from typing import Generator, Tuple, Optional

class VideoProcessor:
    """Handle video input from file or RTSP stream"""
    def __init__(self, source: str, skip_frames: int = 0):
        """
        Args:
            source: Video file path or RTSP URL
            skip_frames: Process every Nth frame (0 = all frames)
        """
        self.source = source
        self.skip_frames = skip_frames
        self.cap = cv2.VideoCapture(source)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source: {source}")
        
        # Video properties
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video opened: {source}")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"FPS: {self.fps}")
        print(f"Total frames: {self.total_frames}")
        
        self.frame_count = 0
    
    def get_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Yield frames from video"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            
            if not ret:
                break
            
            self.frame_count += 1
            
            # Skip frames if configured
            if self.skip_frames > 0 and self.frame_count % (self.skip_frames + 1) != 0:
                continue
            
            yield self.frame_count, frame
    
    def get_timestamp(self, frame_number: int) -> float:
        """Convert frame number to video timestamp in seconds"""
        return frame_number / self.fps if self.fps > 0 else 0
    
    def release(self):
        """Release video capture"""
        self.cap.release()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()