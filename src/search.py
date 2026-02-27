
from dotenv import load_dotenv
from sentence_transformers import CrossEncoder
import chainlit as cl
import logging
from pathlib import Path
from src.db import embed_text
from RAGService import RagService

logger = logging.getLogger(__name__)
rag_service = RagService()

@cl.on_chat_start
async def start():
    await cl.Message(content="PostgreSQL Support Bot is ready. Ask me anything!").send()

@cl.on_message
async def main(message: cl.Message):
    msg = cl.Message(content="")
    await msg.send()
    try:
        # Get RAG response and stream it back to the user if not empty
        response = rag_service.get_response(message.content)
        if isinstance(response, str):
            msg.content = response 
            await msg.update()               
        else:
            for token in response:
                await msg.stream_token(token)
    except Exception as e:
        logger.error(f"Error in main loop: {e}")
        msg.content = f"An error occurred: {str(e)}"
        await msg.update()
    await msg.update()
            