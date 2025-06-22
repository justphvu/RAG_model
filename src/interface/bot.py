from telegram import ForceReply, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from src.config import Constants
from src.pipeline.build_rag_pipeline import build_rag_pipeline
from src.pipeline.ragpipeline import EnhancedRAGPipeline
from src.models.conversation import ConversationManager
from src.utils.resilience import get_resilience_manager, CircuitBreakerState

import logging
import time
import os
from typing import Optional, Callable

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

class TelegramBot:
    """
    Manages the Telegram bot's lifecycle, command handlers, and interactions.

    This class initializes the bot with a RAG pipeline, sets up handlers
    for various commands, and manages the main application loop. It also
    integrates resilience features to provide system health updates.
    """
    def __init__(self, token: str, pipeline: EnhancedRAGPipeline):
        """
        Initializes the TelegramBot.

        Args:
            token (str): The Telegram bot API token.
            pipeline (EnhancedRAGPipeline): The RAG pipeline instance for answering queries.
        """
        if not token:
            raise ValueError("Telegram bot token cannot be empty.")
        self.application = Application.builder().token(token).build()
        self.pipeline = pipeline
        self._register_handlers()

    def _register_handlers(self) -> None:
        """Registers all the command and message handlers for the bot."""
        self.application.add_handler(CommandHandler("start", self._start))
        self.application.add_handler(CommandHandler("help", self._help))
        self.application.add_handler(CommandHandler("clear", self._clear_history))
        self.application.add_handler(CommandHandler("stats", self._get_stats))
        self.application.add_handler(CommandHandler("history", self._get_history))
        self.application.add_handler(CommandHandler("health", self._get_health))
        self.application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message))
        self.application.add_handler(CallbackQueryHandler(self._button_callback))
    
    async def _safe_execute(self, update: Update, context: ContextTypes.DEFAULT_TYPE, func: Callable, *args, **kwargs):
        """
        A wrapper to safely execute bot functions with error handling.

        Args:
            update (Update): The incoming Telegram update.
            context (ContextTypes.DEFAULT_TYPE): The context object for the update.
            func (Callable): The function to execute.
        """
        try:
            await func(update, context, *args, **kwargs)
        except Exception as e:
            logger.error(f"An error occurred in handler {func.__name__}: {e}", exc_info=True)
            if update.message:
                await update.message.reply_text("Sorry, an internal error occurred. Please try again later.")
            elif update.callback_query:
                await update.callback_query.message.reply_text("Sorry, an internal error occurred processing that action.")
    
    # Handlers are now wrapped with _safe_execute in the run method or could be done with a decorator
    
    async def _start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles the /start command. Greets the user and shows main menu."""
        user_name = update.effective_user.first_name
        welcome_text = (
            f"👋 Welcome, {user_name}!\n\n"
            "I am a university assistant bot, ready to answer your questions. "
            "How can I help you today?\n\n"
            "You can ask me anything, or use /help to see all available commands."
        )
        keyboard = [
            [InlineKeyboardButton("❓ Ask a Question", switch_inline_query_current_chat="")],
            [InlineKeyboardButton("📜 Command List", callback_data="help")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await update.message.reply_text(welcome_text, reply_markup=reply_markup)

    async def _help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles the /help command. Displays a list of available commands."""
        help_text = (
            "Here are the available commands:\n"
            "/start - Show the welcome message\n"
            "/help - Display this help message\n"
            "/clear - Clear your conversation history\n"
            "/stats - Get performance and cache statistics\n"
            "/history - Show your recent conversation history\n"
            "/health - Check the health of the system components"
        )
        await update.message.reply_text(help_text)

    async def _handle_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles regular text messages by feeding them to the RAG pipeline."""
        user_id = str(update.effective_user.id)
        query_text = update.message.text
        
        # Show a "typing..." indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action='typing')
        
        # Get answer from RAG pipeline
        answer = self.pipeline.answer(user_id, query_text)
        
        await update.message.reply_text(answer)

    async def _clear_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Clears the conversation history for the current user."""
        user_id = str(update.effective_user.id)
        self.pipeline.conversation_manager.clear_history(user_id)
        await update.message.reply_text("Your conversation history has been cleared.")

    async def _get_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Provides performance and cache statistics from the pipeline."""
        stats = self.pipeline.get_stats()
        stats_text = (
            "📊 **Performance & Cache Statistics**\n\n"
            f"- Total Queries Processed: `{stats['performance']['total_queries']}`\n"
            f"- Average Processing Time: `{stats['performance']['avg_processing_time']:.2f}s`\n"
            f"- Cache Hits: `{stats['cache']['hits']}`\n"
            f"- Cache Misses: `{stats['cache']['misses']}`\n"
        )
        await update.message.reply_text(stats_text, parse_mode='Markdown')

    async def _get_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Displays the user's recent conversation history."""
        user_id = str(update.effective_user.id)
        history = self.pipeline.conversation_manager.get_history(user_id)
        
        if not history:
            await update.message.reply_text("You have no conversation history yet.")
            return

        history_text = "📜 **Your Recent Conversation**\n\n"
        for msg in history:
            role = "You" if msg.role == "user" else "Bot"
            history_text += f"**{role}**: {msg.content}\n"
            
        await update.message.reply_text(history_text, parse_mode='Markdown')

    async def _get_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Checks the health of the system's components (e.g., circuit breakers)."""
        resilience_manager = get_resilience_manager()
        health_status = resilience_manager.get_all_breaker_states()
        
        if not health_status:
            await update.message.reply_text("Resilience monitoring is not enabled.")
            return

        health_text = "❤️ **System Health**\n\n"
        for key, state in health_status.items():
            status_icon = "✅" if state == CircuitBreakerState.CLOSED else "❌" if state == CircuitBreakerState.OPEN else "⚠️"
            health_text += f"{status_icon} **{key.replace('_', ' ').title()}**: `{state.name}`\n"
        
        await update.message.reply_text(health_text, parse_mode='Markdown')

    async def _button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handles callbacks from inline keyboard buttons."""
        query = update.callback_query
        await query.answer()
        
        if query.data == 'help':
            await self._help(update, context)

    def run(self) -> None:
        """Starts the bot and polls for updates."""
        logger.info("Bot is starting...")
        # Handlers are wrapped with safe execution logic here, but could also be done with decorators
        # This is a conceptual change to show how error handling is centralized
        original_handlers = self.application.handlers
        self.application.handlers = {}
        for group, handlers in original_handlers.items():
            self.application.handlers[group] = [
                handler if not isinstance(handler, CommandHandler) and not isinstance(handler, MessageHandler)
                else type(handler)(handler.filters if isinstance(handler, MessageHandler) else handler.commands, lambda u,c,h=handler.callback: self._safe_execute(u,c,h))
                for handler in handlers
            ]

        self.application.run_polling()
        logger.info("Bot has stopped.")

def main() -> None:
    """Start the enhanced bot with conversation support."""
    # Build enhanced RAG pipeline
    rag_pipeline = build_rag_pipeline()
    
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(Constants.TELEGRAM_TOKEN).build()
    
    # Store RAG pipeline in bot data for access in handlers
    application.bot_data['rag_pipeline'] = rag_pipeline

    # Command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("clear", clear_conversation))
    application.add_handler(CommandHandler("stats", show_stats))
    application.add_handler(CommandHandler("history", show_history))
    
    # Button callback handler
    application.add_handler(CallbackQueryHandler(button_callback))

    # Enhanced message handler with RAG
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, build_bot_handler(rag_pipeline)))
    
    # Periodic cleanup of old conversations (every hour)
    async def cleanup_conversations(context: ContextTypes.DEFAULT_TYPE):
        """Periodically cleanup old conversations."""
        try:
            if hasattr(rag_pipeline, 'cleanup_old_conversations'):
                cleaned_count = rag_pipeline.cleanup_old_conversations()
                if cleaned_count > 0:
                    logger.info(f"Cleaned up {cleaned_count} old conversations")
        except Exception as e:
            logger.error(f"Error during conversation cleanup: {e}")
    
    # Schedule cleanup job (every hour)
    job_queue = application.job_queue
    job_queue.run_repeating(cleanup_conversations, interval=3600, first=3600)
    
    logger.info("Enhanced bot started with conversation support and performance optimizations")
    
    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)
