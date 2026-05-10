from src.face_encoder import FaceEncoder
from src.surveillance_engine import SurveillanceEngine
import cv2

def main():
    encoder = FaceEncoder(device='cpu')
    
    target_img = cv2.imread('data/target/person03.png')
    if target_img is None:
        print("Error: Could not load target image")
        return
    
    target_embedding = encoder.get_embedding(target_img)
    
    if target_embedding is None:
        print("No face in target")
        return
    
    print("Target loaded\n")
    
    engine = SurveillanceEngine(
        target_embedding=target_embedding,
        threshold=0.35,  # Slightly lower threshold
        skip_frames=0,   # Process every other frame
        min_consecutive=3,
        device='cpu'
    )
    
    results = engine.process_video(
        video_source='data/videos/sample.mp4',
        output_path='outputs/tracked_phase3.mp4',
        show_preview=False
    )
    
    print(f"\n{'-'*60}")
    print("PHASE 3 RESULTS (WITH TRACKING)")
    print(f"{'-'*60}")
    print(f"Raw Matches: {results['summary']['total_raw_matches']}")
    print(f"Confirmed Targets: {results['summary']['confirmed_targets']}")
    print(f"False Positive Reduction: {results['summary']['false_positive_rate']:.1%}")
    print(f"{'-'*60}\n")

if __name__ == '__main__':
    main()