import cv2
import numpy as np


class FaceQualityFilter:
    def __init__(self, min_size: int, min_blur_score: int):
        self.min_size = 80
        self.min_blur_score = 100
        self.max_pose_angle = 45
    
    def is_acceptable(self, face, frame):
        """Check if face meets quality standards"""
        # Size check
        x1, y1, x2, y2 = face['bbox']
        if (x2 - x1) < self.min_size or (y2 - y1) < self.min_size:
            return False, "too_small"
        
        # Blur check
        face_crop = frame[y1:y2, x1:x2]
        blur_score = cv2.Laplacian(
            cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY), 
            cv2.CV_64F
        ).var()
        
        if blur_score < self.min_blur_score:
            return False, "too_blurry"
        
        # Pose check (simplified)
        landmarks = face['landmarks']
        if self._is_profile(landmarks):
            return False, "profile_view"
        
        return True, "acceptable"
    
    def _is_profile(self, landmarks):
        """Rough profile detection"""
        left_eye, right_eye, nose = landmarks[0], landmarks[1], landmarks[2]
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        offset = abs(nose[0] - eye_center_x)
        eye_distance = np.linalg.norm(left_eye - right_eye)
        
        return offset > eye_distance * 0.3