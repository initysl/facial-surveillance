from collections import defaultdict, deque

class TemporalValidator:
    """Validate matches across frames to reduce false positives"""
    def __init__(self, 
                 min_consecutive: int = 3,
                 window_size: int = 10):
        """
        Args:
            min_consecutive: Matches needed in window to confirm
            window_size: Frame window to check
        """
        self.min_consecutive = min_consecutive
        self.window_size = window_size
        
        # Track match history: {track_id: deque([True, False, True, ...])}
        self.match_history = defaultdict(lambda: deque(maxlen=window_size))
        
        # Confirmed tracks
        self.confirmed_tracks = set()
    
    def update(self, track_id: int, is_match: bool) -> bool:
        """Update match history for track and check if confirmed"""
        # Add to history
        self.match_history[track_id].append(is_match)
        
        # Check for consecutive matches
        recent = list(self.match_history[track_id])
        
        if len(recent) < self.min_consecutive:
            return False
        
        # Count matches in window
        match_count = sum(recent[-self.window_size:])
        
        # Confirm if min_consecutive matches found
        is_confirmed = match_count >= self.min_consecutive
        
        # Track confirmed targets
        if is_confirmed:
            if track_id not in self.confirmed_tracks:
                self.confirmed_tracks.add(track_id)
                return True  # First confirmation - emit alert
        else:
            # Unconfirm if match rate drops
            if track_id in self.confirmed_tracks:
                self.confirmed_tracks.discard(track_id)
        
        return False
    
    def is_confirmed(self, track_id: int) -> bool:
        """Check if track is currently confirmed"""
        return track_id in self.confirmed_tracks
    
    def get_match_rate(self, track_id: int) -> float:
        """Get match rate for track in current window"""
        if track_id not in self.match_history:
            return 0.0
        
        history = list(self.match_history[track_id])
        if not history:
            return 0.0
        
        return sum(history) / len(history)