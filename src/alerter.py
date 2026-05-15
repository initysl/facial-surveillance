import asyncio
from telegram import Bot
import cv2
import numpy as np
from typing import Optional
from datetime import datetime
import io


class TelegramAlerter:
    """Send real-time alerts via Telegram bot."""
    
    def __init__(self, bot_token: str, chat_id: str, enabled: bool = True):
        self.enabled = enabled
        
        if not enabled:
            print("Telegram alerts disabled")
            return
        
        self.bot = Bot(token=bot_token)
        self.chat_id = chat_id
        
        # Verify connection
        try:
            asyncio.run(self._test_connection())
            print("Telegram bot connected")
        except Exception as e:
            print(f"Telegram connection failed: {e}")
            self.enabled = False
    
    async def _test_connection(self):
        """Test bot connection."""
        bot_info = await self.bot.get_me()
        print(f"  Bot: @{bot_info.username}")
    
    def send_alert(self, message: str, face_crop: Optional[np.ndarray] = None, clip_path: Optional[str] = None):
        """Send alert with optional image/video."""
        if not self.enabled:
            return
        
        try:
            asyncio.run(self._send_async(message, face_crop, clip_path))
        except Exception as e:
            print(f"⚠️  Alert send failed: {e}")
    
    async def _send_async(self, message: str, face_crop, clip_path):
        """Async send implementation."""
        
        # Send with photo
        if face_crop is not None:
            # Convert numpy to bytes
            _, buffer = cv2.imencode('.jpg', face_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
            photo_bytes = io.BytesIO(buffer.tobytes())
            
            await self.bot.send_photo(
                chat_id=self.chat_id,
                photo=photo_bytes,
                caption=message,
                parse_mode='HTML'
            )
        
        # Send with video clip
        elif clip_path is not None:
            with open(clip_path, 'rb') as video_file:
                await self.bot.send_video(
                    chat_id=self.chat_id,
                    video=video_file,
                    caption=message,
                    parse_mode='HTML'
                )
        
        # Text only
        else:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode='HTML'
            )
    
    def alert_match_confirmed(self,
                            track_id: int,
                            similarity: float,
                            timestamp: float,
                            video_source: str,
                            face_crop: Optional[np.ndarray] = None):
        """Alert when target is confirmed."""
        
        message = (
            f"<b>TARGET CONFIRMED</b>\n\n"
            f"Source: {video_source}\n"
            f"Track ID: #{track_id}\n"
            f"Time: {timestamp:.2f}s\n"
            f" Similarity: {similarity:.3f}\n"
            f" Detected: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        self.send_alert(message, face_crop=face_crop)
    
    def alert_session_start(self, video_source: str, target_name: Optional[str] = None):
        """Alert when surveillance session starts."""
        
        message = (
            f"<b>Surveillance Started</b>\n\n"
            f"Source: {video_source}\n"
        )
        
        if target_name:
            message += f"🎯 Target: {target_name}\n"
        
        message += f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        self.send_alert(message)
    
    def alert_session_end(self, total_matches: int, confirmed_matches: int):
        """Alert when surveillance session ends."""
        
        message = (
            f"<b>Surveillance Ended</b>\n\n"
            f"Total Matches: {total_matches}\n"
            f"Confirmed: {confirmed_matches}\n"
            f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
        
        self.send_alert(message)