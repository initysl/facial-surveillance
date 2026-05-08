from src.face_encoder import FaceEncoder
from src.surveillance_engine import SurveillanceEngine
from PIL import Image

encoder = FaceEncoder()
target_img = Image.open('data/target/person02.png')
target_embedding = encoder.process_image(target_img)

engine = SurveillanceEngine(
    target_embedding,
    threshold=0.6, 
    skip_frames=0  # Process every 3rd frame
)

matches = engine.process_video(
    'data/videos/surveillance.mp4',
    output_path='outputs/detected.mp4'
)

print(f"\nTotal matches: {len(matches)}")
for match in matches:
    print(f"Frame {match['frame']} @ {match['timestamp']:.2f}s - Similarity: {match['similarity']:.3f}")