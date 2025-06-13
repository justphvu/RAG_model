from dotenv import load_dotenv
import os

load_dotenv()

class Constants:
    """
    Конфигурация проекта
    """
    DEVICE="cuda:2"
    RAW_HTML_PATH = "./data/"
    HF_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
        
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
    EMBEDDING_MODEL = "ai-forever/FRIDA"
    LLM_MODEL = "yandex/YandexGPT-5-Lite-8B-instruct"
    
    """Model's source"""
    LLM_MODEL_PATH = "./models/llm"
    TOKENIZER_PATH = "./models/tokenizer"
    EMBEDDING_MODEL_PATH = "./models/embedding"
    
    """Database's names"""
    PARENT_COLLECTION = "parent"
    CHILD_COLLECTION = "child"
    
    """Database's path"""
    DB_PATH = "./outputs/chroma_db"
    
    """Answer generation's config"""
    DEFAULT_SYSTEM_PROMPT = """Ты лаконичный и точный ассистент. Отвечай на вопросы пользователя прямо, только на основе предоставленного контекста. Не объясняй, не обосновывай и не уточняй, если об этом явно не просят. Если ответа нет в контексте, ответь: "Ответа нет." """
    max_new_tokens = 768
    temperature = 0.7
    top_k = 50
    top_p = 0.95

