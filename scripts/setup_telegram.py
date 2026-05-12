import asyncio
from telegram import Bot
from telegram.error import TelegramError

async def get_chat_id(bot_token: str):
    """Get chat ID by sending test message."""
    bot = Bot(token=bot_token)
    
    print("\n🤖 Bot Information:")
    me = await bot.get_me()
    print(f"  Username: @{me.username}")
    print(f"  Name: {me.first_name}")
    print(f"  ID: {me.id}")
    
    print("\n📱 To get your chat ID:")
    print(f"  1. Open Telegram and search for @{me.username}")
    print("  2. Start a conversation with the bot")
    print("  3. Send any message to the bot")
    print("  4. Run this script again\n")
    
    # Get updates
    updates = await bot.get_updates()
    
    if updates:
        print("💬 Recent Messages:")
        for update in updates[-5:]:  # Last 5 messages
            if update.message:
                chat = update.message.chat
                print(f"\n  Chat ID: {chat.id}")
                print(f"  Type: {chat.type}")
                if chat.username:
                    print(f"  Username: @{chat.username}")
                print(f"  Message: {update.message.text}")
    else:
        print("⚠️  No messages found. Send a message to the bot first.")

def main():
    print("="*60)
    print("TELEGRAM BOT SETUP")
    print("="*60)
    
    print("\n📋 Instructions:")
    print("  1. Go to https://t.me/BotFather")
    print("  2. Send /newbot and follow instructions")
    print("  3. Copy the bot token\n")
    
    bot_token = input("Enter your bot token: ").strip()
    
    if not bot_token:
        print("❌ Token required")
        return
    
    try:
        asyncio.run(get_chat_id(bot_token))
        
        print("\n" + "-"*60)
        print("✅ Setup Complete!")
        print("-"*60)
        print("\nAdd to your config.yaml:")
        print(f"""
alerts:
  enabled: true
  telegram_token: "{bot_token}"
  telegram_chat_id: "YOUR_CHAT_ID_FROM_ABOVE"
        """)
        
    except TelegramError as e:
        print(f"\n❌ Error: {e}")
        print("Check your bot token and try again.")

if __name__ == '__main__':
    main()