import cv2
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional

class ClipExtractor:
    """Extract video clips around match events."""
    
    def __init__(self, buffer_seconds: int = 5, fps: int = 30, output_dir: str = 'outputs/clips'):
        """
        Args:
            buffer_seconds: Seconds of video to save before match
            fps: Frames per second
            output_dir: Directory to save clips
        """
        self.buffer_seconds = buffer_seconds
        self.fps = fps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Rolling buffer for pre-match frames
        max_frames = buffer_seconds * fps
        self.frame_buffer = deque(maxlen=max_frames)
        
        # Post-match collection
        self.collecting_post = False
        self.post_frames = []
        self.post_frames_needed = buffer_seconds * fps
        self.current_clip_path = None
    
    def add_frame(self, frame: np.ndarray):
        """Add frame to buffer."""
        # Always store in pre-buffer
        self.frame_buffer.append(frame.copy())
        
        # If collecting post-match frames
        if self.collecting_post:
            self.post_frames.append(frame.copy())
            
            # Check if done collecting
            if len(self.post_frames) >= self.post_frames_needed:
                self._save_clip()
                self.collecting_post = False
    
    def start_clip(self, event_id: int) -> str:
        """Start collecting frames for a clip."""
        self.collecting_post = True
        self.post_frames = []
        
        # Generate output path
        self.current_clip_path = str(
            self.output_dir / f"match_{event_id}_{int(cv2.getTickCount())}.mp4"
        )
        
        return self.current_clip_path
    
    def _save_clip(self):
        """Save buffered frames as video clip."""
        if not self.current_clip_path:
            return
        
        # Combine pre and post frames
        all_frames = list(self.frame_buffer) + self.post_frames
        
        if not all_frames:
            return
        
        # Get frame dimensions
        height, width = all_frames[0].shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        writer = cv2.VideoWriter(
            self.current_clip_path,
            fourcc,
            self.fps,
            (width, height)
        )
        
        # Write frames
        for frame in all_frames:
            writer.write(frame)
        
        writer.release()
        
        print(f"Clip saved: {self.current_clip_path} ({len(all_frames)} frames)")
        
        self.current_clip_path = None
    
    def save_immediate(self, event_id: int) -> Optional[str]:
        """Save current buffer immediately (no post-frames)."""
        if not self.frame_buffer:
            return None
        
        output_path = str(
            self.output_dir / f"match_{event_id}_{int(cv2.getTickCount())}.mp4"
        )
        
        frames = list(self.frame_buffer)
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
        
        for frame in frames:
            writer.write(frame)
        
        writer.release()
        
        return output_path