import argparse
import cv2
import sys
from pathlib import Path

from src.face_encoder import FaceEncoder
from src.surveillance_engine import SurveillanceEngine
from src.config import Config

def main():
    parser = argparse.ArgumentParser(
        description='Facial Verification Surveillance System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python main.py --target suspect.jpg --video footage.mp4
  
  # With configuration
  python main.py --target suspect.jpg --video rtsp://camera1 --config production.yaml
  
  # Live preview
  python main.py --target suspect.jpg --video footage.mp4 --preview
  
  # Save annotated output
  python main.py --target suspect.jpg --video footage.mp4 --output result.mp4
        """
    )
    
    # Required arguments
    parser.add_argument(
        '--target',
        required=True,
        help='Path to target person image'
    )
    
    parser.add_argument(
        '--video',
        required=True,
        help='Path to video file or RTSP stream URL'
    )
    
    # Optional arguments
    parser.add_argument(
        '--config',
        default='config/default.yaml',
        help='Configuration file (default: config/default.yaml)'
    )
    
    parser.add_argument(
        '--output',
        help='Path to save annotated output video'
    )
    
    parser.add_argument(
        '--name',
        help='Name of target person (for alerts/logs)'
    )
    
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Show live preview window'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        help='Override similarity threshold from config'
    )
    
    parser.add_argument(
        '--device',
        choices=['cuda', 'cpu'],
        help='Override device from config'
    )
    
    parser.add_argument(
        '--export-csv',
        help='Export results to CSV file'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = Config.load(args.config)
        print(f"Loaded config: {args.config}")
    except Exception as e:
        print(f"Config error: {e}")
        sys.exit(1)
    
    # Apply CLI overrides
    if args.threshold:
        config['matching']['threshold'] = args.threshold
        print(f"  Override threshold: {args.threshold}")
    
    if args.device:
        config['model']['device'] = args.device
        print(f"  Override device: {args.device}")
    
    # Load target image
    print(f"\nLoading target: {args.target}")
    target_img = cv2.imread(args.target)
    
    if target_img is None:
        print(f"Could not load target image: {args.target}")
        sys.exit(1)
    
    # Extract target embedding
    encoder = FaceEncoder(
        device=config['model']['device'],
        det_size=tuple(config['model']['det_size'])
    )
    
    target_embedding = encoder.get_embedding(target_img, skip_quality_check=True)
    
    if target_embedding is None:
        print("No face detected in target image")
        sys.exit(1)
    
    print(f"Target embedding extracted (shape: {target_embedding.shape})")
    
    # Initialize surveillance engine
    print("\nInitializing surveillance engine...")
    engine = SurveillanceEngine(
        target_embedding=target_embedding,
        config=config
    )
    
    # Run surveillance
    try:
        results = engine.process_video(
            video_source=args.video,
            output_video_path=args.output,
            target_name=args.name,
            show_preview=args.preview
        )
        
        # Export CSV if requested
        if args.export_csv:
            engine.database.export_to_csv(args.export_csv)
            print(f"✓ Results exported to {args.export_csv}")
        
        print("\nSurveillance completed successfully")
        print(f"Session ID: {results['session_id']}")
        print(f"Database: {results['database_path']}")
        
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()