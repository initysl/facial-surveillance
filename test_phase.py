from src.face_encoder import FaceEncoder
from src.matcher import FaceMatcher
from PIL import Image

encoder = FaceEncoder()
matcher = FaceMatcher(threshold=0.6)

# Load target image
test_img = Image.open('data/target/person02.png')
target_embedding = encoder.process_image(test_img)

# Test against another image
test_img = Image.open('data/frame/frame002.png')
test_embedding = encoder.process_image(test_img)

# Compare
is_match, similarity = matcher.is_match(target_embedding, test_embedding)
print(f"Match: {is_match}, Similarity: {similarity:.3f}")


