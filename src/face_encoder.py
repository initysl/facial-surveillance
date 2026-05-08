from insightface.app import FaceAnalysis
import numpy as np
from typing import List, Dict, Optional

import torch

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

        faces = self.app.get(image)

        results = []
        for face in faces:
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
        Return face with highest detection score
        For single person target images
        """
        faces = self.process_image(image)

        if not faces:
            return None
        
        best_face = max(faces, key=lambda x: x['det_score'])
        return best_face
    
    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Extract embedding from best face in image"""
        face = self.get_best_face(image)
        return face['embedding'] if face else None