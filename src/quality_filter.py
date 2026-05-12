import cv2
import numpy as np
from typing import Tuple

class FaceQualityFilter:
    """Filter faces based on quality metrics."""
    
    def __init__(self, min_size: int = 50, min_blur_score: float = 80, check_pose: bool = True):

        self.min_size = min_size
        self.min_blur_score = min_blur_score
        self.check_pose = check_pose
    
    def is_acceptable(self, face: dict, frame: np.ndarray) -> Tuple[bool, str]:
        """Check if face meets quality standards."""
        # Size check
        x1, y1, x2, y2 = face['bbox']
        width = x2 - x1
        height = y2 - y1
        
        if width < self.min_size or height < self.min_size:
            return False, "too_small"
        
        # Blur check
        try:
            face_crop = frame[y1:y2, x1:x2]
            blur_score = self._compute_blur_score(face_crop)
            face['blur_score'] = blur_score  # Add to face dict
            
            if blur_score < self.min_blur_score:
                return False, "too_blurry"
        except Exception:
            return False, "crop_failed"
        
        # Pose check
        if self.check_pose and 'landmarks' in face:
            if self._is_profile(face['landmarks']):
                return False, "profile_view"
        
        return True, "acceptable"
    
    def _compute_blur_score(self, image: np.ndarray) -> float:
        """Compute image sharpness using Laplacian variance."""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    
    def _is_profile(self, landmarks: list) -> bool:
        """Detect profile faces from landmarks."""
        if len(landmarks) < 3:
            return False
        
        # landmarks: [[x1,y1], [x2,y2], ...] for 5 points
        left_eye = np.array(landmarks[0])
        right_eye = np.array(landmarks[1])
        nose = np.array(landmarks[2])
        
        # Eye center
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        
        # Nose offset from eye center
        offset = abs(nose[0] - eye_center_x)
        
        # Eye distance
        eye_distance = np.linalg.norm(left_eye - right_eye)
        
        # If nose is far from center, face is turned
        return offset > eye_distance * 0.3