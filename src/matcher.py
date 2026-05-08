import numpy as np
from typing import Tuple

class FaceMatcher:
    """Compare face embeddings using cosine similarity"""
    def __init__(self, threshold=0.4):
        """
        Args:
            threshold: Similarity threshold for positive match
                      0.4 = conservative (fewer false positives)
                      0.3 = balanced
                      0.2 = liberal (more false positives)
        """
        self.threshold = threshold
    
    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Compute cosine similarity (dot product of normalized vectors)"""
        # Embeddings from InsightFace are already normalized
        similarity = np.dot(embedding1, embedding2)
        return float(similarity)
    
    def is_match(self, embedding1: np.ndarray, embedding2: np.ndarray) -> Tuple[bool, float]:
        """Check if two embeddings represent the same person"""
        similarity = self.compute_similarity(embedding1, embedding2)
        return similarity >= self.threshold, similarity
    
    def find_matches(self, query_embedding: np.ndarray, gallery_embeddings: list) -> list:
        """Find all matches in a gallery of embeddings"""
        matches = []
        
        for face_id, emb in gallery_embeddings:
            is_match, sim = self.is_match(query_embedding, emb)
            if is_match:
                matches.append((face_id, sim))
        
        # Sort by similarity (descending)
        matches.sort(key=lambda x: x[1], reverse=True)
        
        return matches