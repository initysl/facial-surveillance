import os
from dotenv import load_dotenv
from src.face_encoder import FaceEncoder
from src.matcher import FaceMatcher
from src.tracker import FaceTracker
from src.temporal_validator import TemporalValidator
from src.video_processor import VideoProcessor
from src.database import Database
from src.alerter import TelegramAlerter
from src.clip_extractor import ClipExtractor
from src.quality_filter import FaceQualityFilter
import cv2
import numpy as np
from typing import Optional, Dict
from tqdm import tqdm
from pathlib import Path

class SurveillanceEngine:
    """
    Surveillance pipeline: video -> detection -> storage -> alerts.
    """
    
    def __init__(self, target_embedding: np.ndarray, config: Dict):
        """
        Args:
            target_embedding: Reference face embedding
            config: Configuration dictionary
        """

        load_dotenv()

        self.target_embedding = target_embedding
        self.config = config
        
        if 'alerts' in self.config:
            # Use .env value as priority, fall back to what's in the config
            self.config['alerts']['telegram_token'] = os.getenv(
                "TELEGRAM_BOT_TOKEN", self.config['alerts'].get('telegram_token')
            )
            self.config['alerts']['telegram_chat_id'] = os.getenv(
                "TELEGRAM_CHAT_ID", self.config['alerts'].get('telegram_chat_id')
            )
        # Core components
        self.encoder = FaceEncoder(
            device=config['model']['device'],
            det_size=tuple(config['model']['det_size'])
        )
        
        self.matcher = FaceMatcher(
            threshold=config['matching']['threshold']
        )
        
        self.tracker = FaceTracker(
            max_age=config['tracking']['max_age'],
            n_init=config['tracking']['n_init']
        )
        
        self.validator = TemporalValidator(
            min_consecutive=config['matching']['min_consecutive'],
            window_size=config['matching']['window_size']
        )
        
        # Commented qualiy filter (needs improvement)
        # self.quality_filter = FaceQualityFilter(
        #     min_size=config['quality']['min_face_size'],
        #     min_blur_score=config['quality']['min_blur_score']
        # )
        
        # Storage components
        self.database = Database(config['storage']['database_path'])
        
        # Alert system - NOW uses the patched config
        telegram_config = config.get('alerts', {})
        self.alerter = TelegramAlerter(
            bot_token=telegram_config.get('telegram_token', ''),
            chat_id=telegram_config.get('telegram_chat_id', ''),
            enabled=telegram_config.get('enabled', False)
        )
        
        # Clip extraction
        clip_config = config.get('clips', {})
        self.clip_extractor = ClipExtractor(
            buffer_seconds=clip_config.get('buffer_seconds', 5),
            fps=clip_config.get('fps', 30),
            output_dir=config['storage']['output_dir'] + '/clips'
        ) if clip_config.get('enabled', False) else None
        
        # Output directories
        self.output_dir = Path(config['storage']['output_dir'])
        self.matches_dir = self.output_dir / 'matches'
        self.matches_dir.mkdir(parents=True, exist_ok=True)
        
        # Session tracking
        self.session_id = None
        self.frame_count = 0
        self.match_count = 0
        self.confirmed_count = 0
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'tracks_created': 0,
            'raw_matches': 0,
            'confirmed_matches': 0
        }
    
    def process_video(self, video_source: str, output_video_path: Optional[str] = None,
                     target_name: Optional[str] = None,
                     show_preview: bool = False) -> Dict:
        """Process video with full pipeline."""
        
        # Start database session
        self.session_id = self.database.start_session(
            video_source=video_source,
            target_name=target_name,
            config=self.config
        )
        
        # Send start alert
        self.alerter.alert_session_start(video_source, target_name)
        
        print(f"\n{'-'*60}")
        print(f"SURVEILLANCE SESSION STARTED")
        print(f"{'-'*60}")
        print(f"Session ID: {self.session_id}")
        print(f"Video Source: {video_source}")
        print(f"Target: {target_name or 'Unknown'}")
        print(f"{'-'*60}\n")
        
        # Process video
        try:
            with VideoProcessor(
                video_source,
                skip_frames=self.config['processing']['skip_frames']
            ) as video:
                
                # Setup output writer
                writer = None
                if output_video_path:
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # type: ignore
                    # Account for stats panel height
                    output_height = video.height + 100
                    writer = cv2.VideoWriter(
                        output_video_path,
                        fourcc,
                        video.fps,
                        (video.width, output_height)
                    )
                
                # Progress bar
                total_frames = video.total_frames // (self.config['processing']['skip_frames'] + 1)
                pbar = tqdm(total=total_frames, desc="Processing", unit="frame")
                
                # Process frames
                for frame_num, frame in video.get_frames():
                    self._process_frame(
                        frame=frame,
                        frame_num=frame_num,
                        video_timestamp=video.get_timestamp(frame_num),
                        video_source=video_source,
                        writer=writer,
                        show_preview=show_preview
                    )
                    
                    pbar.update(1)
                
                pbar.close()
                
                if writer:
                    writer.release()
                
                if show_preview:
                    cv2.destroyAllWindows()
        
        finally:
            # End session
            self.database.end_session(
                self.session_id,
                self.stats['frames_processed'],
                self.stats['confirmed_matches']
            )
            
            # Send end alert
            self.alerter.alert_session_end(
                self.stats['raw_matches'],
                self.stats['confirmed_matches']
            )
        
        # Print results
        self._print_results()
        
        return {
            'session_id': self.session_id,
            'stats': self.stats,
            'database_path': self.config['storage']['database_path']
        }
    
    def _process_frame(self, frame: np.ndarray, frame_num: int,
                      video_timestamp: float,
                      video_source: str,
                      writer,
                      show_preview: bool):
        """Process single frame."""
        
        self.stats['frames_processed'] += 1
        
        # Add to clip buffer
        if self.clip_extractor:
            self.clip_extractor.add_frame(frame)
        
        # Detect faces
        faces = self.encoder.process_image(frame)
        self.stats['faces_detected'] += len(faces)
        
        # Quality filtering
        # quality_filtered_faces = []
        # for face in faces:
        #     is_acceptable, reason = self.quality_filter.is_acceptable(face, frame)
        #     if is_acceptable:
        #         quality_filtered_faces.append(face)
        
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
            is_newly_confirmed = self.validator.update(track_id, is_match)
            is_confirmed = self.validator.is_confirmed(track_id)
            
            # Log raw matches
            if is_match:
                self.stats['raw_matches'] += 1
                
                # Extract face crop
                x1, y1, x2, y2 = face['bbox']
                face_crop = frame[y1:y2, x1:x2]
                
                # Save face crop
                crop_path = self.matches_dir / f"match_{frame_num}_{track_id}.jpg"
                cv2.imwrite(str(crop_path), face_crop)
                
                # Start clip collection on first confirmation
                clip_path = None
                if is_newly_confirmed and self.clip_extractor:
                    clip_path = self.clip_extractor.start_clip(
                        event_id=f"{self.session_id}_{track_id}" # type: ignore
                    )
                
                # Log to database
                event_id = self.database.log_match(
                    video_source=video_source,
                    frame_number=frame_num,
                    video_timestamp=video_timestamp,
                    similarity=similarity,
                    bbox=face['bbox'],
                    track_id=track_id,
                    det_score=face.get('det_score'),
                    blur_score=face.get('blur_score'),
                    face_crop=face_crop,
                    clip_path=clip_path,
                    confirmed=is_confirmed
                )
            
            # First-time confirmation alert
            if is_newly_confirmed:
                self.stats['confirmed_matches'] += 1
                
                print(f"\nTARGET CONFIRMED - Track #{track_id} @ {video_timestamp:.2f}s (Similarity: {similarity:.3f})")
                
                # Send Telegram alert
                self.alerter.alert_match_confirmed(
                    track_id=track_id,
                    similarity=similarity,
                    timestamp=video_timestamp,
                    video_source=video_source,
                    face_crop=face_crop
                )
            
            # Draw on frame
            if is_confirmed:
                self._draw_confirmed_match(frame, face, track_id, similarity)
            elif is_match:
                self._draw_tentative_match(frame, face, track_id, similarity)
            else:
                self._draw_tracked_face(frame, face, track_id)
        
        # Add stats panel
        frame = self._add_stats_panel(frame)
        
        # Write output
        if writer:
            writer.write(frame)
        
        # Show preview
        if show_preview:
            preview = cv2.resize(frame, (1280, 720))
            cv2.imshow('Surveillance', preview)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                raise KeyboardInterrupt
    
    def _draw_confirmed_match(self, frame, face, track_id, similarity):
        """Draw green box for confirmed target."""
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
        
        label = f"TARGET #{track_id} [{similarity:.3f}]"
        self._draw_label(frame, label, (x1, y1), (0, 255, 0))
    
    def _draw_tentative_match(self, frame, face, track_id, similarity):
        """Draw yellow box for unconfirmed match."""
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
        
        match_rate = self.validator.get_match_rate(track_id)
        label = f"Checking #{track_id} [{similarity:.3f}] {match_rate:.0%}"
        self._draw_label(frame, label, (x1, y1), (0, 255, 255))
    
    def _draw_tracked_face(self, frame, face, track_id):
        """Draw gray box for tracked non-match."""
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
    
    def _add_stats_panel(self, frame):
        """Add statistics panel to frame."""
        panel_height = 100
        panel = np.zeros((panel_height, frame.shape[1], 3), dtype=np.uint8)
        
        # Stats text
        y_offset = 25
        line_height = 25
        
        texts = [
            f"Frames: {self.stats['frames_processed']} | Faces: {self.stats['faces_detected']} | Tracks: {len(self.tracker.track_embeddings)}",
            f"Raw Matches: {self.stats['raw_matches']} | Confirmed: {self.stats['confirmed_matches']}",
            f"Session ID: {self.session_id}"
        ]
        
        for i, text in enumerate(texts):
            cv2.putText(
                panel, text, (10, y_offset + i * line_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
            )
        
        return np.vstack([panel, frame])
    
    def _print_results(self):
        """Print final results."""
        print(f"\n{'-'*60}")
        print("SURVEILLANCE SESSION COMPLETED")
        print(f"{'-'*60}")
        print(f"Session ID: {self.session_id}")
        print(f"Frames Processed: {self.stats['frames_processed']}")
        print(f"Faces Detected: {self.stats['faces_detected']}")
        print(f"Raw Matches: {self.stats['raw_matches']}")
        print(f"Confirmed Targets: {self.stats['confirmed_matches']}")
        print(f"False Positive Rate: {(1 - self.stats['confirmed_matches'] / max(self.stats['raw_matches'], 1)) * 100:.1f}%")
        print(f"{'-'*60}\n")    