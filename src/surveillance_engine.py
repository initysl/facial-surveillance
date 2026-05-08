from src.face_encoder import FaceEncoder
from src.matcher import FaceMatcher
from src.video_processor import VideoProcessor
import cv2

class SurveillanceEngine:
    def __init__(self, target_embedding, threshold=0.6, skip_frames=2):
        self.target_embedding = target_embedding
        self.encoder = FaceEncoder()
        self.matcher = FaceMatcher(threshold=threshold)
        self.skip_frames = skip_frames
        
    def process_video(self, video_source, output_path=None):
        """Scan video for target face"""
        processor = VideoProcessor(video_source) 
        matches = []
        
        # Optional: video writer for output
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
            out = cv2.VideoWriter(
                output_path, fourcc, processor.fps,
                (int(processor.cap.get(3)), int(processor.cap.get(4)))
            )
        
        for frame_num, frame_rgb, frame_bgr in processor.get_frames(self.skip_frames):
            # Detect face in frame
            stream_embedding = self.encoder.process_image(frame_rgb)
            
            if stream_embedding is not None:
                # Compare with target
                is_match, similarity = self.matcher.is_match(
                    self.target_embedding, 
                    stream_embedding
                )
                
                if is_match:
                    print(f"MATCH at frame {frame_num} (similarity: {similarity:.3f})")
                    matches.append({
                        'frame': frame_num,
                        'similarity': similarity,
                        'timestamp': frame_num / processor.fps
                    })
                    
                    # Draw bounding box on frame
                    cv2.putText(
                        frame_bgr, 
                        f"MATCH: {similarity:.2f}", 
                        (50, 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, (0, 255, 0), 2
                    )
            
            if output_path:
                out.write(frame_bgr)
        
        processor.release()
        if output_path:
            out.release()
        
        return matches