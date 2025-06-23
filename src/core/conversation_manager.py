import os 
import json
import re
import glob

HISTORY_DIR = "chat_history"
class ConversationManager:
    
    def save_chat_history(messages, filename):
        """Saves the chat history to a JSON file."""
        filepath = os.path.join(HISTORY_DIR, ConversationManager.sanitize_filename(filename))
        with open(filepath, 'w') as f:
            json.dump(messages, f, indent=4)

    def load_chat_history(filename):
        """Loads a chat history from a JSON file."""
        filepath = os.path.join(HISTORY_DIR, filename)
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None

    def get_chat_history_files():
        """Returns a sorted list of chat history files."""
        files = glob.glob(os.path.join(HISTORY_DIR, "*.json"))
        # Sort files by modification time, newest first
        files.sort(key=os.path.getmtime, reverse=True)
        return [os.path.basename(f) for f in files]
    
    def sanitize_filename(filename):
        # Remove invalid characters for Windows filenames
        return re.sub(r'[<>:"/\\|?*]', '', filename)
