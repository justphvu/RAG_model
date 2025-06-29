from dotenv import load_dotenv
import os

load_dotenv()

class Constants:
    """
    Конфигурация проекта
    """
    DEVICE=os.getenv("DEVICE")
    RAW_HTML_PATH = "./data/"
    DB_PATH = os.getenv("CHROMA_PERSISTENCE_DIR")
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
        
    """Chunking config"""
    parent_chunk_size = 2000
    parent_overlap = 0
    child_chunk_size = 300
    child_overlap = 50
    MARKDOWN_SEPARATORS = [
        "\n#{1,6} ",
        "```\n",
        "\n\\*\\*\\*+\n",
        "\n---+\n",
        "\n___+\n",
        "\n\n",
        "\n",
        ".",
        "?",
        "!",
        " ",
        "",
    ]
    
    """Model's name"""
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
    LLM_MODEL = os.getenv("LLM_MODEL")
    
    """Model's source"""
    LLM_MODEL_PATH = "./models/llm"
    TOKENIZER_PATH = "./models/tokenizer"
    EMBEDDING_MODEL_PATH = "./models/embedding"
    
    """Database's names"""
    PARENT_COLLECTION = "parent"
    CHILD_COLLECTION = "child"
    
    """Conversation's config"""
    MAX_CONVERSATIONS = 1000 # Maximum number of conversations to keep in memory
    CONVERSATION_TTL = 3600 # 1 hour
    ENABLE_PERSISTENCE = True # Enable persistence of conversations to a file
    PERSISTENCE_FILE = "./cache/conversations.json"
    
    """Retriever's config"""
    MAX_CONTEXT_TOKENS = 2000 # Maximum number of tokens to use for context window
    
    """Embedding's config"""
    BATCH_SIZE = 32 # Batch size for embedding
    ENABLE_DISK_CACHE = True # Enable disk cache for embeddings
    USE_HALF_PRECISION = False # Enable half precision for embeddings
    EMBEDDING_CACHE_DIR = "./cache/embeddings" # Directory for embedding cache
    
    """LLM's config"""
    MAX_MEMORY = None # Maximum memory for LLM
    LOAD_IN_8BIT = True # Load in 8bit    
    
    """Answer generation's config"""
    DEFAULT_SYSTEM_PROMPT = """Ты лаконичный и точный ассистент. Отвечай на вопросы пользователя прямо, только на основе предоставленного контекста. Не объясняй, не обосновывай и не уточняй, если об этом явно не просят. Если ответа нет в контексте, ответь: "Ответа нет." """
    max_new_tokens = 768
    temperature = 0.7
    top_k = 50
    top_p = 0.95

