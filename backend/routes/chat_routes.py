from backend.utils.reference_links import get_collection_reference_link
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from backend.models.chat_model import ChatQuery, ChatResponse, LogEntry
from backend.db.mongo_utils import get_legal_db, get_chatbot_db, insert_log_entry
from backend.nlp.rag import generate_answer_with_rag, load_llm_and_models
from backend.nlp.model_loader import get_embedding_model
import logging
import os
import io
import pyttsx3
import base64
from datetime import datetime
from typing import Dict, Any, List, Optional
import re
import traceback


# Import cosine_similarity from similarity module
from backend.nlp.similarity import cosine_similarity

# --- Ensure LLM and embedding model are loaded at startup ---
load_llm_and_models()

logger = logging.getLogger(__name__)
router = APIRouter()

# Configuration
CONFIDENCE_THRESHOLD = 0.1  # lowered threshold to increase recall
KEYWORDS_COLLECTION = os.getenv("KEYWORDS_COLLECTION", "keywords")

# --- Updated ChatResponse model to include optional audio data ---
class ChatResponse(BaseModel):
    bot_response: str = Field(..., description="The chatbot's response text.")
    status: str = Field(..., description="Status of the query (e.g., 'answered', 'unanswered', 'error').")
    language: str = Field(..., description="Language of the bot's response.")
    query_id: Optional[str] = Field(None, description="Optional ID for the processed query.")
    similarity_score: Optional[float] = Field(None, description="Cosine similarity score if answered by NLP.")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio of the bot's response.")

async def get_synonyms_from_db(query_text: str, language: str) -> List[str]:
    '''Fetch synonyms from the chatbot database's keywords collection.'''
    try:
        db = get_chatbot_db()
        synonyms_collection = db[KEYWORDS_COLLECTION]
        query_words = [word.strip().lower() for word in query_text.split() if word.strip()]
        search_field = "english_synonyms" if language == 'en' else "hindi_synonyms"
        synonym_docs = list(synonyms_collection.find({search_field: {"$in": query_words}}))
        
        expanded_keywords = []
        for doc in synonym_docs:
            expanded_keywords.append(doc.get('english_keyword', ''))
            expanded_keywords.append(doc.get('hindi_keyword', ''))
            expanded_keywords.extend(doc.get('english_synonyms', []))
            expanded_keywords.extend(doc.get('hindi_synonyms', []))

        expanded_keywords = [kw for kw in set(expanded_keywords) if kw]
        return expanded_keywords
    except Exception as e:
        logger.error(f"Error fetching synonyms: {e}")
        return []

class ChatRequest(BaseModel):
    user_id: str = Field(...)
    query_text: str = Field(...)
    language: str = Field("en")
    chat_history: Optional[List[Dict[str, str]]] = None

def generate_tts_audio(text: str, lang: str) -> Optional[str]:
    '''Generates a encoded audio string from text using pyttsx3.'''
    try:
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if lang == 'hi':
            for voice in voices:
                if 'hi' in voice.languages or 'Hindi' in voice.name:
                    engine.setProperty('voice', voice.id)
                    break
        else:
            for voice in voices:
                if 'en' in voice.languages or 'English' in voice.name:
                    engine.setProperty('voice', voice.id)
                    break
        audio_stream = io.BytesIO()
        engine.save_to_file(text, 'temp_tts_output.mp3')
        engine.runAndWait()
        with open('temp_tts_output.mp3', 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        os.remove('temp_tts_output.mp3')
        return audio_base64
    except Exception as e:
        logger.error(f"Failed to generate TTS audio: {e}", exc_info=True)
        return None

@router.post("/chat", response_model=ChatResponse)
async def chat_with_bot(request: ChatRequest):
    user_query_text = request.query_text
    user_id = request.user_id
    language = request.language
    chat_history = request.chat_history

    logger.info(f"Received chat query from {user_id} ({language}): '{user_query_text}'")

    context_window = 5
    history_context = ""
    if chat_history:
        for msg in chat_history[-context_window:]:
            prefix = "User:" if msg.get("sender") == "user" else "Bot:"
            history_context += f"{prefix} {msg.get('text', '')}\n"

    try:
        # Use the new RAG+LLM pipeline for answer generation
        bot_response_text = generate_answer_with_rag(user_query_text)
        status_text = "answered" if bot_response_text and "couldn't find" not in bot_response_text.lower() else "unanswered"
        audio_data = generate_tts_audio(bot_response_text, lang=language)
    except Exception as e:
        logger.error(f"Error processing chat query '{user_query_text}': {e}\n{traceback.format_exc()}")
        bot_response_text = "An internal error occurred while processing your request. Please try again."
        status_text = "error"
        audio_data = None
        await insert_log_entry(LogEntry(
            timestamp=datetime.now(),
            user_id=user_id,
            query_text=user_query_text,
            bot_response_text=bot_response_text,
            status=status_text,
            language=language,
            similarity_score=None
        ).dict())
        raise HTTPException(status_code=500, detail="Internal server error.")

    if status_text != "error":
        await insert_log_entry(LogEntry(
            timestamp=datetime.now(),
            user_id=user_id,
            query_text=user_query_text,
            bot_response_text=bot_response_text,
            status=status_text,
            language=language,
            similarity_score=None
        ).dict())

    return ChatResponse(
        bot_response=bot_response_text,
        status=status_text,
        language=language,
        similarity_score=None,
        audio_data=audio_data
    )