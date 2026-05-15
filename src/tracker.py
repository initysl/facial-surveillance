from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np
from typing import List, Dict, Tuple, Optional

class FaceTracker:
    """DeepSORT: Track faces across frames"""
    def __init__(self, max_age=30, n_init=3):
        """
        Args:
            max_age: Frames to keep track alive without detection
            n_init: Frames needed to confirm track
        """
        self.tracker = DeepSort(
            max_age=max_age,
            n_init=n_init,
            max_iou_distance=0.7,
            max_cosine_distance=0.3,
            nn_budget=100
        )
        
        # Store embeddings per track
        self.track_embeddings = {}  # {track_id: [embedding1, embedding2, ...]}
    
    def update(self, faces: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            faces: List of face dicts from FaceEncoder.process_image()
            frame: Current frame (for DeepSORT appearance features)
        
        Returns:
            List of faces with added 'track_id' field
        """
        if not faces:
            self.tracker.update_tracks([], frame=frame)
            return []
        
        # Convert to DeepSORT format
        detections = []
        for face in faces:
            x1, y1, x2, y2 = face['bbox']
            conf = face['det_score']
            
            # DeepSORT expects: ([left, top, width, height], confidence, feature)
            detections.append((
                [x1, y1, x2 - x1, y2 - y1],
                conf,
                face['embedding']  # Use face embedding as appearance feature
            ))
        
        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Map tracks back to faces
        tracked_faces = []
        for i, track in enumerate(tracks):
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            
            # Find corresponding face
            if i < len(faces):
                face = faces[i].copy()
                face['track_id'] = track_id
                
                # Store embedding for this track
                if track_id not in self.track_embeddings:
                    self.track_embeddings[track_id] = []
                
                self.track_embeddings[track_id].append(face['embedding'])
                
                # Keep only recent embeddings (memory optimization)
                if len(self.track_embeddings[track_id]) > 10:
                    self.track_embeddings[track_id].pop(0)
                
                tracked_faces.append(face)
        
        return tracked_faces
    
    def get_track_embedding(self, track_id: int) -> Optional[np.ndarray]:
        """Get average embedding for a track (more robust)"""
        if track_id not in self.track_embeddings:
            return None
        
        embeddings = self.track_embeddings[track_id]
        return np.mean(embeddings, axis=0)