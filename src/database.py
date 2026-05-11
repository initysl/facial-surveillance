from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, LargeBinary, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import numpy as np
import cv2
from typing import Optional, List

Base = declarative_base()

class MatchEvent(Base):
    """Database model for face match events."""
    __tablename__ = 'match_events'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    
    # Video context
    video_source = Column(String, nullable=False)
    frame_number = Column(Integer)
    video_timestamp = Column(Float)  # Seconds into video
    
    # Match details
    similarity = Column(Float)
    track_id = Column(Integer, nullable=True, index=True)
    bbox_x1 = Column(Integer)
    bbox_y1 = Column(Integer)
    bbox_x2 = Column(Integer)
    bbox_y2 = Column(Integer)
    
    # Quality metrics
    det_score = Column(Float)  # Detection confidence
    blur_score = Column(Float, nullable=True)
    
    # Evidence
    face_crop = Column(LargeBinary, nullable=True)  # JPEG bytes
    video_clip_path = Column(String, nullable=True)
    
    # Status
    confirmed = Column(Boolean, default=False)
    reviewed = Column(Boolean, default=False)
    notes = Column(String, nullable=True)

class SurveillanceSession(Base):
    """Track surveillance sessions."""
    __tablename__ = 'sessions'
    
    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime, default=datetime.utcnow)
    end_time = Column(DateTime, nullable=True)
    video_source = Column(String)
    target_name = Column(String, nullable=True)
    total_frames_processed = Column(Integer, default=0)
    total_matches = Column(Integer, default=0)
    config_used = Column(String, nullable=True)  # JSON config snapshot

class Database:
    """Database manager for surveillance events."""
    
    def __init__(self, db_path: str = 'surveillance.db'):
        self.engine = create_engine(f'sqlite:///{db_path}', echo=False)
        Base.metadata.create_all(self.engine)
        
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
        
        print(f"✓ Database connected: {db_path}")
    
    def start_session(self, video_source: str, target_name: Optional[str] = None, config: Optional[dict] = None) -> int:
        """Start a new surveillance session."""
        import json
        
        session = SurveillanceSession(
            video_source=video_source,
            target_name=target_name,
            config_used=json.dumps(config) if config else None
        )
        
        self.session.add(session)
        self.session.commit()
        
        return session.id # type: ignore
    
    def end_session(self, session_id: int, total_frames: int, total_matches: int):
        """Close a surveillance session."""
        session = self.session.query(SurveillanceSession).filter_by(id=session_id).first()
        
        if session:
            session.end_time = datetime.now()  # type: ignore
            session.total_frames_processed = total_frames  # type: ignore
            session.total_matches = total_matches  # type: ignore
            self.session.commit()
    
    def log_match(self, video_source: str, frame_number: int, video_timestamp: float, similarity: float,
                  bbox: list,
                  track_id: Optional[int] = None,
                  det_score: Optional[float] = None,
                  blur_score: Optional[float] = None,
                  face_crop: Optional[np.ndarray] = None,
                  clip_path: Optional[str] = None,
                  confirmed: bool = False) -> int:
        """Log a match event to database."""
        
        # Encode face crop as JPEG bytes
        face_bytes = None
        if face_crop is not None:
            _, buffer = cv2.imencode('.jpg', face_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            face_bytes = buffer.tobytes()
        
        event = MatchEvent(video_source=video_source, frame_number=frame_number, video_timestamp=video_timestamp, similarity=similarity,
            track_id=track_id,
            bbox_x1=bbox[0],
            bbox_y1=bbox[1],
            bbox_x2=bbox[2],
            bbox_y2=bbox[3],
            det_score=det_score,
            blur_score=blur_score,
            face_crop=face_bytes,
            video_clip_path=clip_path,
            confirmed=confirmed
        )
        
        self.session.add(event)
        self.session.commit()
        
        return event.id # type: ignore
    
    def get_recent_matches(self, limit: int = 10) -> List[MatchEvent]:
        """Get most recent match events."""
        return self.session.query(MatchEvent)\
            .order_by(MatchEvent.timestamp.desc())\
            .limit(limit)\
            .all()
    
    def get_matches_by_track(self, track_id: int) -> List[MatchEvent]:
        """Get all matches for a specific track ID."""
        return self.session.query(MatchEvent)\
            .filter_by(track_id=track_id)\
            .order_by(MatchEvent.timestamp)\
            .all()
    
    def get_confirmed_matches(self) -> List[MatchEvent]:
        """Get all confirmed (validated) matches."""
        return self.session.query(MatchEvent)\
            .filter_by(confirmed=True)\
            .order_by(MatchEvent.timestamp.desc())\
            .all()
    
    def export_to_csv(self, output_path: str):
        """Export all matches to CSV."""
        import csv
        
        matches = self.session.query(MatchEvent).all()
        
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'ID', 'Timestamp', 'Video Source', 'Frame', 'Video Time (s)',
                'Track ID', 'Similarity', 'Confirmed', 'Bbox', 'Clip Path'
            ])
            
            # Data
            for match in matches:
                writer.writerow([
                    match.id,
                    match.timestamp,
                    match.video_source,
                    match.frame_number,
                    match.video_timestamp,
                    match.track_id,
                    match.similarity,
                    match.confirmed,
                    f"({match.bbox_x1},{match.bbox_y1},{match.bbox_x2},{match.bbox_y2})",
                    match.video_clip_path
                ])
        
        print(f"Exported {len(matches)} matches to {output_path}")
    
    def get_face_crop(self, event_id: int) -> Optional[np.ndarray]:
        """Retrieve face crop image from database."""
        event = self.session.query(MatchEvent).filter_by(id=event_id).first()
        
        if event and event.face_crop: # type: ignore
            # Decode JPEG bytes to numpy array
            nparr = np.frombuffer(event.face_crop, np.uint8) # type: ignore
            return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return None
    
    def close(self):
        """Close database connection."""
        self.session.close()