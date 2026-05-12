import argparse
from pathlib import Path
import sys
from src.database import Database
import cv2

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))



def main():
    parser = argparse.ArgumentParser(description='Database Viewer Utility')
    parser.add_argument('--db', default='surveillance.db', help='Database path')
    parser.add_argument('--recent', type=int, help='Show N recent matches')
    parser.add_argument('--track', type=int, help='Show matches for track ID')
    parser.add_argument('--confirmed', action='store_true', help='Show only confirmed')
    parser.add_argument('--export-csv', help='Export to CSV file')
    parser.add_argument('--show-crops', action='store_true', help='Display face crops')
    
    args = parser.parse_args()
    
    # Connect to database
    db = Database(args.db)
    
    # Query matches
    if args.track is not None:
        matches = db.get_matches_by_track(args.track)
        print(f"\n{'-'*60}")
        print(f"MATCHES FOR TRACK ID: {args.track}")
        print(f"{'-'*60}")
    elif args.confirmed:
        matches = db.get_confirmed_matches()
        print(f"\n{'-'*60}")
        print(f"CONFIRMED MATCHES")
        print(f"{'-'*60}")
    elif args.recent:
        matches = db.get_recent_matches(args.recent)
        print(f"\n{'-'*60}")
        print(f"RECENT {args.recent} MATCHES")
        print(f"{'-'*60}")
    else:
        matches = db.get_recent_matches(10)
        print(f"\n{'-'*60}")
        print(f"RECENT 10 MATCHES")
        print(f"{'-'*60}")
    
    # Display matches
    if not matches:
        print("No matches found.")
        return
    
    print(f"\nTotal: {len(matches)} matches\n")
    
    for match in matches:
        print(f"ID: {match.id}")
        print(f"  Time: {match.timestamp}")
        print(f"  Video: {match.video_source}")
        print(f"  Frame: {match.frame_number} @ {match.video_timestamp:.2f}s")
        print(f"  Track ID: {match.track_id}")
        print(f"  Similarity: {match.similarity:.4f}")
        print(f"  Confirmed: {'YES' if bool(match.confirmed) else 'NO'}")
        print(f"  Bbox: ({match.bbox_x1}, {match.bbox_y1}, {match.bbox_x2}, {match.bbox_y2})")
        if match.video_clip_path is not None:
            print(f"  Clip: {match.video_clip_path}")
        print()
        
        # Show face crop
        if args.show_crops and match.face_crop is not None:
            face_img = db.get_face_crop(match.id) # type: ignore
            if face_img is not None:
                cv2.imshow(f"Match {match.id} - Track {match.track_id}", face_img)
    
    if args.show_crops:
        print("Press any key to close images...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    # Export CSV
    if args.export_csv:
        db.export_to_csv(args.export_csv)
    
    db.close()

if __name__ == '__main__':
    main()