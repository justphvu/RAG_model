from telegram import ForceReply, Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackQueryHandler
from src.config import Constants
from src.pipeline.build_rag_pipeline import build_rag_pipeline
from src.pipeline.ragpipeline import EnhancedRAGPipeline
from src.models.conversation import ConversationManager

import logging
import time

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
# set higher logging level for httpx to avoid all GET and POST requests being logged
logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


# Define a few command handlers. These usually take the two arguments update and context.
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_html(
        rf"Привет {user.mention_html()}! Я твой бот-ассистент по поступлению в университет 🤖. Задай мне любой вопрос!\n\n"
        "Доступные команды:\n"
        "/help - Показать подсказки\n"
        "/clear - Очистить историю\n"
        "/stats - Показать статистику работы бота\n"
        "/history - Показать последние сообщения из истории",
        reply_markup=ForceReply(selective=True),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    help_text = """
🤖 **University Assistant Bot Help**

**Основные команды:**
• Просто введите свой вопрос, и я поищу ответ в документах!
• Я запоминаю контекст нашей беседы для более точных ответов

**Available Commands:**
• `/start` - Запустить бота и увидеть список команд
• `/help` - Показать сообщение с подсказками
• `/clear` - Очистить историю твоих сообщений
• `/stats` - Показать статистику работы бота
• `/history` - Показать историю твоих недавних сообщений

**Советы:**
• Задавай уточняющие вопросы – я помню наш разговор
• Будь конкретен в вопросах, чтобы получить лучшие ответы
• Я могу помочь с вопросами о поступлении, программах, стоимости и многом другом!

**Примеры:**
• "Какие требования к поступлению на программу Computer Science?"
• "Сколько стоит магистерская программа?"
• "Какие есть программы аспирантуры?"
"""
    await update.message.reply_text(help_text, parse_mode='Markdown')

async def clear_conversation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Clear the user's conversation history."""
    user_id = str(update.effective_user.id)
    
    # Get the RAG pipeline from context
    rag_pipeline = context.bot_data.get('rag_pipeline')
    if rag_pipeline and hasattr(rag_pipeline, 'clear_conversation'):
        rag_pipeline.clear_conversation(user_id)
        await update.message.reply_text("✅ Your conversation history has been cleared!")
    else:
        await update.message.reply_text("❌ Unable to clear conversation history.")

async def show_stats(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show bot performance statistics."""
    rag_pipeline = context.bot_data.get('rag_pipeline')
    if not rag_pipeline:
        await update.message.reply_text("❌ Statistics not available.")
        return
    
    try:
        # Get various statistics
        cache_stats = rag_pipeline.get_cache_stats()
        perf_stats = rag_pipeline.get_performance_stats()
        conv_stats = rag_pipeline.get_conversation_stats()
        
        stats_text = f"""
📊 **Bot Statistics**

**Performance:**
• Total queries: {perf_stats.get('total_queries', 0)}
• Average processing time: {perf_stats.get('avg_processing_time', 0):.2f}s
• Total processing time: {perf_stats.get('total_processing_time', 0):.2f}s

**Cache Performance:**
• Cache hits: {cache_stats.get('hits', 0)}
• Cache misses: {cache_stats.get('misses', 0)}
• Hit rate: {cache_stats.get('hit_rate', 0):.1%}
• Cache size: {cache_stats.get('cache_size', 0)} entries

**Conversations:**
• Active conversations: {conv_stats.get('total_conversations', 0)}
• Total messages: {conv_stats.get('total_messages', 0)}
• Avg messages per conversation: {conv_stats.get('avg_messages_per_conversation', 0):.1f}
"""
        await update.message.reply_text(stats_text, parse_mode='Markdown')
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        await update.message.reply_text("❌ Error retrieving statistics.")

async def show_history(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the user's recent conversation history."""
    user_id = str(update.effective_user.id)
    
    rag_pipeline = context.bot_data.get('rag_pipeline')
    if not rag_pipeline or not hasattr(rag_pipeline, 'get_conversation_history'):
        await update.message.reply_text("❌ Conversation history not available.")
        return
    
    try:
        history = rag_pipeline.get_conversation_history(user_id, max_messages=10)
        
        if not history:
            await update.message.reply_text("📝 No conversation history found.")
            return
        
        history_text = "📝 **Your Recent Conversation History:**\n\n"
        
        for i, message in enumerate(history[-10:], 1):  # Show last 10 messages
            role_emoji = "👤" if message.role == "user" else "🤖"
            timestamp = time.strftime("%H:%M", time.localtime(message.timestamp))
            history_text += f"{i}. {role_emoji} **{message.role.title()}** ({timestamp}):\n"
            history_text += f"   {message.content[:100]}{'...' if len(message.content) > 100 else ''}\n\n"
        
        # Add clear button
        keyboard = [
            [InlineKeyboardButton("🗑️ Clear History", callback_data=f"clear_history_{user_id}")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            history_text, 
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
        
    except Exception as e:
        logger.error(f"Error getting history: {e}")
        await update.message.reply_text("❌ Error retrieving conversation history.")

async def button_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle button callbacks."""
    query = update.callback_query
    await query.answer()
    
    if query.data.startswith("clear_history_"):
        user_id = query.data.split("_")[-1]
        rag_pipeline = context.bot_data.get('rag_pipeline')
        
        if rag_pipeline and hasattr(rag_pipeline, 'clear_conversation'):
            rag_pipeline.clear_conversation(user_id)
            await query.edit_message_text("✅ Your conversation history has been cleared!")
        else:
            await query.edit_message_text("❌ Unable to clear conversation history.")

############################################################################
# Enhanced message handler using RAG pipeline with conversation support
def build_bot_handler(rag_pipeline: EnhancedRAGPipeline):
    async def bot_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Enhanced reply handler with conversation support and performance monitoring."""
        user_input = update.message.text.strip()
        user_id = str(update.effective_user.id)
        
        logger.info("Question from User %s: %s", user_id, user_input)
        
        if not user_input:
            await update.message.reply_text("Please type something.")
            return
        
        # Show typing indicator
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
        
        # Call enhanced RAG pipeline with conversation support
        try:
            start_time = time.time()
            llm_reply = rag_pipeline.generate_answer(
                query=user_input,
                user_id=user_id,
                use_conversation_history=True
            )
            processing_time = time.time() - start_time
            
            # Log performance
            logger.info(
                "Response generated for user %s in %.2fs: %s", 
                user_id, processing_time, llm_reply[:100] + "..." if len(llm_reply) > 100 else llm_reply
            )
            
        except Exception as e:
            logger.exception("Failed to generate response for user %s", user_id)
            llm_reply = "Sorry, I ran into an error while thinking. Please try again."

        await update.message.reply_text(llm_reply)
        
    return bot_reply


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
