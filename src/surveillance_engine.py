from src.face_encoder import FaceEncoder
from src.matcher import FaceMatcher
from src.tracker import FaceTracker
from src.temporal_validator import TemporalValidator
from src.video_processor import VideoProcessor
import cv2
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm

class SurveillanceEngine:
    """Full surveillance pipeline with tracking and validation"""
    def __init__(self, target_embedding: np.ndarray, threshold: float = 0.4, skip_frames: int = 2, min_consecutive: int = 3, device: str = 'cpu'):
        self.target_embedding = target_embedding
        self.encoder = FaceEncoder(device=device)
        self.matcher = FaceMatcher(threshold=threshold)
        self.tracker = FaceTracker(max_age=30, n_init=3)
        self.validator = TemporalValidator(
            min_consecutive=min_consecutive,
            window_size=10
        )
        self.skip_frames = skip_frames
        
        # Event storage
        self.matches = []
        self.confirmed_detections = []  # Only validated matches
    
    def process_video(self,  video_source: str, output_path: Optional[str] = None, show_preview: bool = False) -> Dict:
        """Scan video with tracking and temporal validation"""
        with VideoProcessor(video_source, self.skip_frames) as video:
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
                writer = cv2.VideoWriter(
                    output_path, fourcc, video.fps,
                    (video.width, video.height)
                )
            
            total = video.total_frames // (self.skip_frames + 1)
            pbar = tqdm(total=total, desc="Processing", unit="frame")
            
            for frame_num, frame in video.get_frames():
                # Detect faces
                faces = self.encoder.process_image(frame)
                
                # Update tracker
                tracked_faces = self.tracker.update(faces, frame)
                
                # Match each tracked face
                for face in tracked_faces:
                    track_id = face['track_id']
                    
                    # Compare with target
                    is_match, similarity = self.matcher.is_match(
                        self.target_embedding,
                        face['embedding']
                    )
                    
                    # Temporal validation
                    is_confirmed = self.validator.update(track_id, is_match)
                    
                    if is_match:
                        # Log all matches
                        self.matches.append({
                            'frame': frame_num,
                            'timestamp': video.get_timestamp(frame_num),
                            'track_id': track_id,
                            'similarity': similarity,
                            'bbox': face['bbox'],
                            'confirmed': self.validator.is_confirmed(track_id)
                        })
                    
                    # First confirmation alert
                    if is_confirmed:
                        print(f"\nCONFIRMED TARGET - Track ID: {track_id} @ {video.get_timestamp(frame_num):.2f}s")
                        
                        self.confirmed_detections.append({
                            'frame': frame_num,
                            'timestamp': video.get_timestamp(frame_num),
                            'track_id': track_id,
                            'similarity': similarity,
                            'bbox': face['bbox']
                        })
                    
                    # Draw on frame
                    if self.validator.is_confirmed(track_id):
                        self._draw_confirmed_match(frame, face, track_id, similarity)
                    elif is_match:
                        self._draw_tentative_match(frame, face, track_id, similarity)
                    else:
                        self._draw_tracked_face(frame, face, track_id)
                
                if writer:
                    writer.write(frame)
                
                if show_preview:
                    cv2.imshow('Surveillance', cv2.resize(frame, (1280, 720)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                pbar.update(1)
            
            pbar.close()
            
            if writer:
                writer.release()
            
            if show_preview:
                cv2.destroyAllWindows()
        
        return {
            'all_matches': self.matches,
            'confirmed_detections': self.confirmed_detections,
            'summary': self.get_match_summary()
        }
    
    def _draw_confirmed_match(self, frame, face, track_id, similarity):
        """Draw GREEN box for confirmed target."""
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        
        label = f"TARGET #{track_id} [{similarity:.3f}]"
        self._draw_label(frame, label, (x1, y1), (0, 255, 0))
    
    def _draw_tentative_match(self, frame, face, track_id, similarity):
        """Draw YELLOW box for unconfirmed match."""
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        
        match_rate = self.validator.get_match_rate(track_id)
        label = f"Checking #{track_id} [{similarity:.3f}] {match_rate:.0%}"
        self._draw_label(frame, label, (x1, y1), (0, 255, 255))
    
    def _draw_tracked_face(self, frame, face, track_id):
        """Draw GRAY box for tracked non-match."""
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
        cv2.putText(frame, f"#{track_id}", (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    def _draw_label(self, frame, text, pos, color):
        """Draw text with background."""
        x, y = pos
        size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - size[1] - 10), (x + size[0], y), color, -1)
        cv2.putText(frame, text, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    def get_match_summary(self):
        """Generate summary statistics."""
        return {
            'total_raw_matches': len(self.matches),
            'confirmed_targets': len(self.confirmed_detections),
            'unique_tracks_matched': len(set(m['track_id'] for m in self.matches)),
            'confirmed_tracks': len(set(d['track_id'] for d in self.confirmed_detections)),
            'false_positive_rate': 1 - (len(self.confirmed_detections) / max(len(self.matches), 1))
        }