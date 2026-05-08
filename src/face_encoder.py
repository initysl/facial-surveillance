import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import numpy as np
from PIL import Image

class FaceEncoder:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # Face detector
        self.mtcnn = MTCNN(
            image_size = 160,
            margin=20,
            keep_all=False, # Only return best face
            device=device
        )

        # Face recognition model (VGGFace2)
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    def detect_and_align(self, image):
        """Detect face and return aligned crop"""
        # image: PIL image or numpy array
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Detect and align
        face = self.mtcnn(image)
        return face
    
    def encode(self, face_tensor):
        """Generate embedding[512] from aligned face"""
        if face_tensor is None:
            return None
        
        with torch.no_grad():
            face_tensor = face_tensor.unsqueeze(0).to(self.device)
            embedding = self.model(face_tensor)

        return embedding.cpu().numpy()[0]
    
    def process_image(self, image):
        """Full pipeline: image -> embedding"""
        face = self.detect_and_align(image)
        if face is None:
            return None
        return self.encode(face)