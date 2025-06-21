import json
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: float
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convert message to dictionary format."""
        return asdict(self)

@dataclass
class Conversation:
    """Represents a conversation session."""
    user_id: str
    messages: List[Message]
    created_at: float
    last_updated: float
    max_messages: int = 20  # Maximum messages to keep in memory
    
    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a new message to the conversation."""
        message = Message(
            role=role,
            content=content,
            timestamp=time.time(),
            metadata=metadata
        )
        self.messages.append(message)
        self.last_updated = time.time()
        
        # Trim old messages if exceeding limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def get_recent_messages(self, count: int = 5) -> List[Message]:
        """Get the most recent messages."""
        return self.messages[-count:] if self.messages else []
    
    def get_context_window(self, max_tokens: int = 2000) -> List[Message]:
        """Get messages that fit within a token limit (approximate)."""
        # Simple token estimation: ~4 characters per token
        char_limit = max_tokens * 4
        current_chars = 0
        context_messages = []
        
        for message in reversed(self.messages):
            message_chars = len(message.content)
            if current_chars + message_chars <= char_limit:
                context_messages.insert(0, message)
                current_chars += message_chars
            else:
                break
        
        return context_messages
    
    def to_dict(self) -> Dict:
        """Convert conversation to dictionary format."""
        return {
            "user_id": self.user_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "max_messages": self.max_messages
        }

class ConversationManager:
    """
    Manages conversation history for multiple users with memory optimization.
    
    Features:
    - Per-user conversation tracking
    - Automatic cleanup of old conversations
    - Memory-efficient storage
    - Context window management
    """
    
    def __init__(
        self,
        max_conversations: int = 1000,
        conversation_ttl: int = 3600,  # 1 hour in seconds
        enable_persistence: bool = True,
        persistence_file: str = "./cache/conversations.json"
    ):
        self.max_conversations = max_conversations
        self.conversation_ttl = conversation_ttl
        self.enable_persistence = enable_persistence
        self.persistence_file = persistence_file
        
        # In-memory storage
        self.conversations: Dict[str, Conversation] = {}
        
        # Load existing conversations if persistence is enabled
        if self.enable_persistence:
            self._load_conversations()
    
    def get_conversation(self, user_id: str) -> Conversation:
        """Get or create a conversation for a user."""
        if user_id not in self.conversations:
            self.conversations[user_id] = Conversation(
                user_id=user_id,
                messages=[],
                created_at=time.time(),
                last_updated=time.time()
            )
        
        return self.conversations[user_id]
    
    def add_user_message(self, user_id: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add a user message to the conversation."""
        conversation = self.get_conversation(user_id)
        conversation.add_message("user", content, metadata)
        self._save_conversations()
    
    def add_assistant_message(self, user_id: str, content: str, metadata: Optional[Dict] = None) -> None:
        """Add an assistant message to the conversation."""
        conversation = self.get_conversation(user_id)
        conversation.add_message("assistant", content, metadata)
        self._save_conversations()
    
    def get_conversation_history(self, user_id: str, max_messages: int = 5) -> List[Message]:
        """Get recent conversation history for a user."""
        conversation = self.get_conversation(user_id)
        return conversation.get_recent_messages(max_messages)
    
    def get_context_window(self, user_id: str, max_tokens: int = 2000) -> List[Message]:
        """Get conversation context that fits within token limit."""
        conversation = self.get_conversation(user_id)
        return conversation.get_context_window(max_tokens)
    
    def clear_conversation(self, user_id: str) -> None:
        """Clear conversation history for a user."""
        if user_id in self.conversations:
            del self.conversations[user_id]
            self._save_conversations()
    
    def cleanup_old_conversations(self) -> int:
        """Remove conversations that haven't been updated recently."""
        current_time = time.time()
        expired_conversations = []
        
        for user_id, conversation in self.conversations.items():
            if current_time - conversation.last_updated > self.conversation_ttl:
                expired_conversations.append(user_id)
        
        for user_id in expired_conversations:
            del self.conversations[user_id]
        
        # Also remove oldest conversations if we exceed max_conversations
        if len(self.conversations) > self.max_conversations:
            sorted_conversations = sorted(
                self.conversations.items(),
                key=lambda x: x[1].last_updated
            )
            conversations_to_remove = len(self.conversations) - self.max_conversations
            for i in range(conversations_to_remove):
                del self.conversations[sorted_conversations[i][0]]
        
        if expired_conversations or len(self.conversations) > self.max_conversations:
            self._save_conversations()
        
        return len(expired_conversations)
    
    def _load_conversations(self) -> None:
        """Load conversations from disk."""
        try:
            if os.path.exists(self.persistence_file):
                with open(self.persistence_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for user_id, conv_data in data.items():
                        messages = [Message(**msg) for msg in conv_data.get('messages', [])]
                        self.conversations[user_id] = Conversation(
                            user_id=user_id,
                            messages=messages,
                            created_at=conv_data.get('created_at', time.time()),
                            last_updated=conv_data.get('last_updated', time.time()),
                            max_messages=conv_data.get('max_messages', 20)
                        )
                logger.info(f"Loaded {len(self.conversations)} conversations from disk")
        except Exception as e:
            logger.error(f"Error loading conversations: {e}")
    
    def _save_conversations(self) -> None:
        """Save conversations to disk."""
        if not self.enable_persistence:
            return
        
        try:
            os.makedirs(os.path.dirname(self.persistence_file), exist_ok=True)
            data = {
                user_id: conversation.to_dict()
                for user_id, conversation in self.conversations.items()
            }
            with open(self.persistence_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Error saving conversations: {e}")
    
    def get_stats(self) -> Dict:
        """Get conversation manager statistics."""
        total_messages = sum(len(conv.messages) for conv in self.conversations.values())
        return {
            "total_conversations": len(self.conversations),
            "total_messages": total_messages,
            "avg_messages_per_conversation": total_messages / len(self.conversations) if self.conversations else 0
        }

# Import os for file operations
import os 