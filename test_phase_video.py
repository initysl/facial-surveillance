from src.face_encoder import FaceEncoder
from src.surveillance_engine import SurveillanceEngine
import cv2

def main():
    # Load target image
    encoder = FaceEncoder(device='cpu') # or cuda
    
    target_img = cv2.imread('data/target/person02.png')
    if target_img is None:
        print("Error: Could not load target image")
        return
    
    target_embedding = encoder.get_embedding(target_img)
    
    if target_embedding is None:
        print("No face detected in target image")
        return
    
    print("Target embedding extracted\n")
    
    # Run surveillance
    engine = SurveillanceEngine(
        target_embedding=target_embedding,
        threshold=0.4,
        skip_frames=0,  # Process every 3rd frame
        device='cuda'
    )
    
    matches = engine.process_video(
        video_source='data/videos/surveillance.mp4',
        output_path='outputs/detected_01.mp4',
        show_preview=False
    )
    
    
    print(f"\n{'-'*60}")
    print("SURVEILLANCE RESULTS")
    print(f"{'-'*60}")
    
    summary = engine.get_match_summary()
    
    print(f"Total Matches: {summary['total_matches']}")
    print(f"Average Similarity: {summary['avg_similarity']:.4f}")
    print(f"Max Similarity: {summary['max_similarity']:.4f}")
    
    if matches:
        print(f"\nFirst Detection: {summary['first_detection']:.2f}s")
        print(f"Last Detection: {summary['last_detection']:.2f}s")
        
        print(f"\nTop 5 Matches:")
        sorted_matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)[:5]
        for i, match in enumerate(sorted_matches, 1):
            print(f"  {i}. Frame {match['frame']} @ {match['timestamp']:.2f}s - Similarity: {match['similarity']:.4f}")
    
    print(f"{'-'*60}\n")

if __name__ == '__main__':
    main()