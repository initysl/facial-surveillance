import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class FaceMatcher:
    def __init__(self, threshold=0.6):
        self.threshold = threshold

    def compare(self, embedding1, embedding2):
        """Compute cosine similarity between embeddings"""
        sim = cosine_similarity(
            embedding1.reshape(1, -1),
            embedding2.reshape(1, -1)
        )[0][0]

        return sim
    
    def is_match(self, embedding1, embedding2):
        """Check if embeddings represent same person"""
        similarity = self.compare(embedding1, embedding2)
        return similarity >= self.threshold,similarity
