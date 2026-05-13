# Facial Verification Surveillance System

Real-time surveillance system that scans video/live RTSP footage to detect and track a specific person. Built with InsightFace for accurate face recognition.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![InsightFace](https://img.shields.io/badge/InsightFace-ArcFace-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue)

---

## What It Does

Upload a photo of a person → System scans surveillance video → Alerts you when that person appears.

**Key Features:**

- **99.8% accuracy** face matching (InsightFace ArcFace)
- **Auto-saves evidence** (face crops, video clips, timestamps)
- **Instant Telegram alerts** when target detected
- **Works with live cameras** (RTSP streams) or recorded videos
- **Smart tracking** - Follows person across frames, reduces false alarms
- **Docker ready** - deploy anywhere

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/yourusername/facial-surveillance.git
cd facial-surveillance

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run

```bash
python main.py \
  --target data/target/person.jpg \
  --video data/videos/surveillance.mp4 \
  --output outputs/result.mp4
```

**That's it!** Results saved to `surveillance.db` and `outputs/` folder.

---

## Real-World Use Cases

| Scenario                  | Command                                                                               |
| ------------------------- | ------------------------------------------------------------------------------------- |
| **Missing person search** | `python main.py --target missing.jpg --video footage.mp4 --threshold 0.3`             |
| **Security monitoring**   | `python main.py --target suspect.jpg --video rtsp://camera1 --config production.yaml` |
| **Event VIP tracking**    | `python main.py --target vip.jpg --video event.mp4 --name "VIP Guest"`                |
| **Live camera feed**      | `python main.py --target person.jpg --video "rtsp://192.168.1.100:554/stream"`        |

---

## How It Works

```
1. Load target photo → Extract face "fingerprint" (512-dimensional embedding)
2. Process video frame-by-frame → Detect all faces using RetinaFace
3. Compare each face embedding to target using cosine similarity
4. Track matches across frames with DeepSORT (reduces false alarms)
5. Validate detections over multiple frames (temporal consistency)
6. Alert + save evidence when confirmed
```

**Processing Speed:**

- **GPU (NVIDIA):** 15-30 FPS
- **CPU:** 5-10 FPS
- **With frame skipping:** Real-time on most hardware

---

## Configuration

Adjust matching sensitivity in `config/default.yaml`:

### CLI Overrides

```bash
# Adjust sensitivity
python main.py --target person.jpg --video test.mp4 --threshold 0.35

# Force CPU mode
python main.py --target person.jpg --video test.mp4 --device cpu

# Export results to CSV
python main.py --target person.jpg --video test.mp4 --export-csv results.csv
```

---

## Setup Telegram Alerts (Optional)

Get instant notifications when target appears:

### Step 1: Create Bot

```bash
python scripts/setup_telegram.py
```

Follow the interactive instructions to:

1. Create a bot with @BotFather
2. Get your bot token
3. Find your chat ID

### Step 2: Configure

Edit `config/production.yaml`:

```yaml
alerts:
  enabled: true
  telegram_token: 'YOUR_BOT_TOKEN'
  telegram_chat_id: 'YOUR_CHAT_ID'
```

### Step 3: Run with Alerts

```bash
python main.py \
  --target person.jpg \
  --video test.mp4 \
  --config config/production.yaml
```

You'll receive alerts like:

```
TARGET CONFIRMED

Source: surveillance.mp4
Track ID: #5
Time: 42.17s
Similarity: 0.723
Detected: 2024-05-06 14:32:18

[Photo of detected face attached]
```

---

## View Results

### Database Viewer

```bash
# Recent detections
python scripts/view_database.py --recent 10

# Show face images
python scripts/view_database.py --recent 10 --show-crops

# Filter by track ID
python scripts/view_database.py --track 5

# Confirmed matches only
python scripts/view_database.py --confirmed

# Export to CSV
python scripts/view_database.py --export-csv report.csv
```

### Database Contents

Every detection is logged with:

- Timestamp
- Video frame number and timestamp
- Similarity score
- Track ID (persistent across frames)
- Bounding box coordinates
- Face crop image (JPEG)
- Video clip path (if enabled)
- Confirmation status

---

## Docker Deployment

### Basic Docker Run

```bash
# Build image
docker build -t surveillance:latest .

# Run
docker run --gpus all \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/config:/app/config \
  surveillance:latest \
  python main.py \
    --target /app/data/target/person.jpg \
    --video /app/data/videos/test.mp4
```

### Docker Compose

```bash
# Edit docker-compose.yml with your settings
docker-compose up

# Run in background
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Project Structure

```
facial-surveillance/
├── main.py                          # CLI entry point
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Docker configuration
├── docker-compose.yml               # Multi-container setup
│
├── src/                             # Core modules
│   ├── face_encoder.py              # InsightFace wrapper
│   ├── matcher.py                   # Similarity comparison
│   ├── tracker.py                   # DeepSORT tracking
│   ├── temporal_validator.py        # False positive reduction
│   ├── quality_filter.py            # Face quality checks
│   ├── database.py                  # SQLite storage
│   ├── alerter.py                   # Telegram notifications
│   ├── clip_extractor.py            # Video clip extraction
│   └── surveillance_engine.py       # Main pipeline
│
├── config/                          # Configuration files
│   ├── default.yaml                 # Default settings
│   ├── production.yaml              # Production mode
│   └── development.yaml             # Dev/testing
│
├── scripts/                         # Utility scripts
│   ├── view_database.py             # Database viewer
│   └── setup_telegram.py            # Telegram bot setup
│
├── data/                            # Input data
│   ├── target/                      # Target photos
│   └── videos/                      # Videos to process
│
├── outputs/                         # Results
│   ├── matches/                     # Face crops
│   ├── clips/                       # Video clips
│   └── logs/                        # Application logs
│
└── surveillance.db                  # SQLite database
```

---

## Requirements

### Target Photo Requirements

**Good:**

- Clear, frontal face (±45° pose)
- Well-lit (indoor/outdoor daylight)
- Minimum 80×80 pixels
- Single person in frame
- No masks/sunglasses

**Poor:**

- Blurry images
- Extreme angles (profile view)
- Very small faces (<50px)
- Heavy shadows
- Sketches/drawings (only real photos work)

### Video Requirements

**Supported:**

- MP4, AVI, MOV files
- RTSP live streams
- 720p or higher recommended
- Aerial/overhead angles work best

## Common Commands

```bash
# Basic scan
python main.py --target person.jpg --video test.mp4

# Live preview window
python main.py --target person.jpg --video test.mp4 --preview

# Adjust sensitivity
python main.py --target person.jpg --video test.mp4 --threshold 0.35

# Live camera
python main.py --target person.jpg --video "rtsp://192.168.1.100:554/stream"

# Production mode (all features)
python main.py --target person.jpg --video test.mp4 --config production.yaml

# CPU-only mode
python main.py --target person.jpg --video test.mp4 --device cpu

# Save annotated video
python main.py --target person.jpg --video test.mp4 --output result.mp4

# Export results
python main.py --target person.jpg --video test.mp4 --export-csv report.csv

# Help
python main.py --help
```

## RTSP Camera Setup

### Common RTSP URL Formats

```bash
# Generic IP Camera
rtsp://username:password@192.168.1.100:554/stream1

# Hikvision
rtsp://admin:password@192.168.1.100:554/Streaming/Channels/101

# Dahua
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0

# Axis
rtsp://192.168.1.100:554/axis-media/media.amp

# Amcrest
rtsp://admin:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
```

### Testing RTSP Connection

```bash
# Test with VLC or ffplay first
vlc "rtsp://admin:pass@192.168.1.100:554/stream1"

# Or ffplay
ffplay "rtsp://admin:pass@192.168.1.100:554/stream1"
```

### Batch Processing

```bash
# Process multiple videos
for video in data/videos/*.mp4; do
  python main.py \
    --target data/target/person.jpg \
    --video "$video" \
    --output "outputs/$(basename $video)"
done
```

### Custom Configuration

```yaml
# config/custom.yaml
```

### Environment Variables

```bash
# .env file
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# Use in config with ${VAR_NAME}
```

## License

MIT License

---

## Acknowledgments

- **InsightFace** - Face recognition framework ([GitHub](https://github.com/deepinsight/insightface))
- **DeepSORT** - Multi-object tracking ([GitHub](https://github.com/nwojke/deep_sort))
- **OpenCV** - Computer vision library

---

**Built with ❤️ for security and safety applications**
