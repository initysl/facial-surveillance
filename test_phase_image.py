from src.face_encoder import FaceEncoder
from src.matcher import FaceMatcher
import cv2
import numpy as np

def test_encoding():
    """Test face detection and embedding extraction."""
    encoder = FaceEncoder(device='cpu')  # or 'cuda'
    
    target_img = cv2.imread('data/target/person01.png')
    print(f"Target image shape: {target_img.shape}") # type: ignore
    
    target_face = encoder.get_best_face(target_img) # type: ignore
    
    if target_face is None:
        print("No face detected in target image")
        return
    
    print(f"Face detected")
    print(f"Bbox: {target_face['bbox']}")
    print(f"Confidence: {target_face['det_score']:.3f}")
    print(f"Embedding shape: {target_face['embedding'].shape}")
    print(f"Embedding norm: {np.linalg.norm(target_face['embedding']):.3f} (should be ~1.0)")
    
    test_img = cv2.imread('data/frame/frame01.png')
    test_face = encoder.get_best_face(test_img) # type: ignore
    
    if test_face is None:
        print("No face in test image")
        return
    
    # Compare embeddings
    matcher = FaceMatcher(threshold=0.4)
    is_match, similarity = matcher.is_match(
        target_face['embedding'], 
        test_face['embedding']
    )
    
    print(f"\n{'-'*50}")
    print(f"Comparison Results:")
    print(f"Similarity: {similarity:.4f}")
    print(f"Match: {'YES' if is_match else 'NO'}")
    print(f"{'-'*50}")

def test_multi_face():
    """Test detection of multiple faces in one image."""
    encoder = FaceEncoder()
    
    img = cv2.imread('data/test/crowd.jpg')
    faces = encoder.process_image(img) # type: ignore
    
    print(f"\nDetected {len(faces)} faces")
    
    for i, face in enumerate(faces):
        print(f"  Face {i+1}: bbox={face['bbox']}, conf={face['det_score']:.3f}")
    
    # Visualize
    for face in faces:
        x1, y1, x2, y2 = face['bbox']
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2) # type: ignore
    
    cv2.imwrite('outputs/multi_face_detection.jpg', img) # type: ignore
    print("Saved visualization to outputs/multi_face_detection.jpg")

if __name__ == '__main__':
    print("Testing InsightFace Integration\n")
    test_encoding()
    # test_multi_face()