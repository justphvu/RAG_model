from telegram import ForceReply, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from src.config import Constants
from src.pipeline.build_rag_pipeline import build_rag_pipeline
from src.pipeline.ragpipeline import RAGPipeline

import logging

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
        rf"Hi {user.mention_html()}!. I’m your university assistant bot 🤖. Ask me anything!",
        reply_markup=ForceReply(selective=True),
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text("Just type your question and I'll search the documents for you!")
    

############################################################################
# Message handler using RAG pipeline
def build_bot_handler(rag_pipeline: RAGPipeline):
    async def bot_reply(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Returns the reply to user after getting reply from server."""
        user_input = update.message.text.strip()
        logger.info("Question from User: %s", user_input)
        
        if not user_input:
            await update.message.reply_text("Please type something.")
            return
        
        # Call RAG pipeline
        try:
            llm_reply = rag_pipeline.generate_answer(user_input)
        except Exception as e:
            logger.exception("Failed to generate response")
            llm_reply = "Sorry, I ran into an error while thinking("

        await update.message.reply_text(llm_reply)
        
    return bot_reply


def main() -> None:
    """Start the bot."""
    # Build RAG pipeline
    rag_pipeline = build_rag_pipeline()
    
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(Constants.TELEGRAM_TOKEN).build()

    # on different commands - answer in Telegram
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # Message handler with RAG
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, build_bot_handler(rag_pipeline)))
    
    # Run the bot until the user presses Ctrl-C
    application.run_polling(allowed_updates=Update.ALL_TYPES)
