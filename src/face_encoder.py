import cv2
from insightface.app import FaceAnalysis
import numpy as np
from typing import List, Dict, Optional
from src.quality_filter import FaceQualityFilter

class FaceEncoder:
    """
    Unified face detection, alignment, and encoding using InsightFace
    """
    def __init__(self, det_size=(640, 640), det_thresh=0.5, device='cuda'):
        """
        Args:
            det_size: Detection input size (width, height)
            det_thresh: Detection confidence threshold
            device: 'cuda' or 'cpu'
        """
        self.quality_filter = FaceQualityFilter()
        # Initialize InsightFace
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device == 'cuda' else ['CPUExecutionProvider']

        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=providers
        )

        self.app.prepare(
            ctx_id=0 if device == 'cuda' else -1,
            det_size=det_size,
            det_thresh=det_thresh
        )

        print(f" InsightFace initialized on {device}")
        print(f"  Detection size: {det_size}")
        print(f"  Detection threshold: {det_thresh}")

    def process_image(self, image: np.ndarray) -> List[Dict]:
        """
        Detect all faces and extract embeddings.        
        Args:
            image: numpy array (BGR format from cv2)
        """
        # Upscale small images
        h, w = image.shape[:2]

        if w < 300 or h < 300:
            # Scale up to minimum 300px on shortest side
            scale = 300 / min(w, h)
            new_w = int(w * scale)
            new_h = int(h * scale)

            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            print(f"Upscaled image from {w}x{h} to {new_w}x{new_h}")


        faces = self.app.get(image)

        results = []
        for face in faces:
            # Quality check
            # is_ok, reason = self.quality_filter.is_acceptable(face, image)

            # if not is_ok:
            #     print(f"Rejected face: {reason}")
            #     continue

            results.append({
                'bbox': face.bbox.astype(int).tolist(),
                'embedding': face.normed_embedding, 
                'det_score': float(face.det_score),
                'landmarks': face.kps.astype(int).tolist(),  # 5 points: eyes, nose, mouth corners
                'age': getattr(face, 'age', None),
                'gender': getattr(face, 'gender', None)
            })
        
        return results
    
    def get_best_face(self, image: np.ndarray) -> Optional[Dict]:
        """
        Return face with highest detection score. 
        For single person target images
        """
        faces = self.process_image(image)

        if not faces:
            return None
        
        best_face = max(faces, key=lambda x: x['det_score'])
        return best_face
    
    def get_embedding(self, image: np.ndarray, skip_quality_check=False) -> Optional[np.ndarray]:
        """Extract embedding from best face in image"""
        face = self.get_best_face(image)
        return face['embedding'] if face else None