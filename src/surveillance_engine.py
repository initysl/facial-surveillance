from src.face_encoder import FaceEncoder
from src.matcher import FaceMatcher
from src.video_processor import VideoProcessor
import cv2
import numpy as np
from typing import List, Dict
from tqdm import tqdm

class SurveillanceEngine:
    """Main surveillance pipeline: video -> detect -> match -> alert"""
    def __init__(self,  target_embedding: np.ndarray, threshold: float = 0.4, skip_frames: int = 2, device: str = 'cuda'):
        self.target_embedding = target_embedding
        self.encoder = FaceEncoder(device=device)
        self.matcher = FaceMatcher(threshold=threshold)
        self.skip_frames = skip_frames
        
        # Match storage
        self.matches = []
    
    def process_video(self, video_source: str, output_path: str, show_preview: bool = False) -> List[Dict]:
        """Scan video for target face"""
        with VideoProcessor(video_source, self.skip_frames) as video:
            # Setup output writer
            writer = None
            if output_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
                writer = cv2.VideoWriter(
                    output_path, fourcc, video.fps,
                    (video.width, video.height)
                )
            
            # Progress bar
            total = video.total_frames // (self.skip_frames + 1)
            pbar = tqdm(total=total, desc="Processing", unit="frame")
            
            # Process frames
            for frame_num, frame in video.get_frames():
                # Detect faces in frame
                faces = self.encoder.process_image(frame)
                
                # Check each face against target
                frame_matches = []
                for face in faces:
                    is_match, similarity = self.matcher.is_match(
                        self.target_embedding,
                        face['embedding']
                    )
                    
                    if is_match:
                        # Store match event
                        match_event = {
                            'frame': frame_num,
                            'timestamp': video.get_timestamp(frame_num),
                            'similarity': similarity,
                            'bbox': face['bbox'],
                            'det_score': face['det_score']
                        }
                        
                        self.matches.append(match_event)
                        frame_matches.append(match_event)
                        
                        # Draw on frame
                        self._draw_match(frame, face, similarity)
                
                # Draw non-match faces (optional)
                for face in faces:
                    if face not in [m['bbox'] for m in frame_matches]:
                        self._draw_detection(frame, face)
                
                # Write frame
                if writer:
                    writer.write(frame)
                
                # Show preview
                if show_preview:
                    cv2.imshow('Surveillance', cv2.resize(frame, (960, 540)))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                pbar.update(1)
            
            pbar.close()
            
            if writer:
                writer.release()
            
            if show_preview:
                cv2.destroyAllWindows()
        
        return self.matches
    
    def _draw_match(self, frame: np.ndarray, face: Dict, similarity: float):
        """Draw green box and label for matched face."""
        x1, y1, x2, y2 = face['bbox']
        
        # Green box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Label
        label = f"TARGET: {similarity:.3f}"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        
        # Background for text
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 10),
            (x1 + label_size[0], y1),
            (0, 255, 0),
            -1
        )
        
        # Text
        cv2.putText(
            frame, label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )
    
    def _draw_detection(self, frame: np.ndarray, face: Dict):
        """Draw gray box for non-match faces."""
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 128, 128), 2)
    
    def get_match_summary(self) -> Dict:
        """Get statistics about matches found."""
        if not self.matches:
            return {
                'total_matches': 0,
                'avg_similarity': 0.0,
                'max_similarity': 0.0,
                'first_detection': None,
                'last_detection': None
            }
        
        similarities = [m['similarity'] for m in self.matches]
        
        return {
            'total_matches': len(self.matches),
            'avg_similarity': np.mean(similarities),
            'max_similarity': np.max(similarities),
            'min_similarity': np.min(similarities),
            'first_detection': self.matches[0]['timestamp'],
            'last_detection': self.matches[-1]['timestamp']
        }