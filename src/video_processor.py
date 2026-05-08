import cv2
from typing import Generator

class VideoProcessor:
    def __int__(self, source):
        """source: video file path or RSTP stram URL"""
        self.cap = cv2.VideoCapture(source)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = 0

    def get_frames(self, skip_frames=0) -> Generator:
        """Yield frames from video"""
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            self.frame_count += 1

            # Skip frames for speed (process every Nth frame)
            if skip_frames > 0 and self.frame_count % (skip_frames + 1 ) !=0:
                continue

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield self.frame_count, frame_rgb, frame

    def release(self):
        self.cap.release()