from typing import List, Dict, Any
import os
import json
import logging
import re

# Import the new required libraries for Llama 3 and LangChain
from groq import Groq
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# --- Initial setup for the LLM ---
logger = logging.getLogger(__name__)


# --- Global LLM and embedding model variables ---
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
groq_client = None
embedding_model = None

def load_llm_and_models():
    """
    Loads the LLM and embedding model into global variables. Call this at app startup.
    """
    global groq_client, embedding_model
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables. Please set it.")
    if groq_client is None:
        groq_client = ChatGroq(
            model="llama-3.1-8b-instant", 
            api_key=GROQ_API_KEY,
            temperature=0.0
        )
    if embedding_model is None:
        from backend.nlp.model_loader import load_nlp_model, get_embedding_model
        load_nlp_model(os.environ.get("NLP_MODEL_NAME", "paraphrase-multilingual-MiniLM-L12-v2"))
        embedding_model = get_embedding_model()

# --- Your existing backend functions (no changes here) ---
# NOTE: These functions must be correctly implemented in your project.
from backend.nlp.similarity import get_embedding, cosine_similarity
from backend.db.mongo_utils import get_legal_db 

# --- New main function to orchestrate retrieval and generation ---
def generate_answer_with_rag(query: str, top_k: int = 5) -> str:
    """
    Orchestrates the RAG process: retrieves documents and then generates an answer using Llama 3.
    """
    try:
        # Step 1: Document Retrieval
        relevant_chunks = retrieve_relevant_faqs(query, top_k=top_k)

        if not relevant_chunks:
            logger.error("No relevant answer found for query: %s", query)
            return "Sorry, I couldn't find any relevant information to answer your question."

        # Step 2: Context Formatting
        context = format_context_for_generation(relevant_chunks)

        prompt_template = """
        You are a highly knowledgeable AI legal assistant specializing in Indian labour laws.
        Your task is to provide accurate and concise answers based ONLY on the following context.
        If the context does not contain enough information to answer the question, state that you cannot answer the question based on the provided information.
        Do not make up any information.
        Your task:
        - Answer the user's question in a clear non repetative, step-by-step, and detailed manner.
        - Use only the information in the CONTEXT below.
        - Do not make up information. If the answer is not present, say so.
        - At the end, provide a 'Reference Links' section listing each link only once, even if referenced multiple times.
        - Each source entry in the list should be in the format `[Source]: <unique_link_from_context>`.
        - Do not repeat the same link as a different source number.

        Context:
        {context}

        Question: {question}

        Answer:

        """

        prompt = ChatPromptTemplate.from_template(prompt_template)

        # Use LangChain to create a simple pipeline for RAG
        rag_chain = (
            {"context": lambda x: context, "question": lambda x: x["question"]}
            | prompt
            | groq_client
            | StrOutputParser()
        )

        try:
            answer = rag_chain.invoke({"question": query})
        except Exception as llm_error:
            logger.error(f"LLM API call failed: {llm_error}", exc_info=True)
            if not GROQ_API_KEY:
                return "LLM API key is missing. Please set GROQ_API_KEY in your environment."
            return f"An error occurred while calling the LLM API: {llm_error}"

        # --- Post-process: Only include links for sources actually cited in the answer ---
        import re
        # Build a mapping from Source number to link
        from backend.utils.reference_links import get_collection_reference_link
        context_chunks = []
        seen_links = set()
        for chunk in relevant_chunks:
            ref_link = ''
            if '_collection' in chunk:
                ref_link = get_collection_reference_link(chunk['_collection'])
            if not ref_link:
                ref_link = 'Not Available'
            if ref_link not in seen_links:
                seen_links.add(ref_link)
                context_chunks.append(ref_link)
        # Find all [Source X] cited in the answer
        cited = set(int(m) for m in re.findall(r'\[Source (\d+)\]', answer))
        # Build the reference section with only cited links
        ref_lines = []
        for idx, link in enumerate(context_chunks, 1):
            if idx in cited:
                ref_lines.append(f"[Source {idx}]: {link}")
        # Remove any existing Reference Links section from the answer
        answer = re.sub(r'Reference Links:.*', '', answer, flags=re.DOTALL)
        # Append the filtered Reference Links section
        if ref_lines:
            answer = answer.rstrip() + "\n\nReference Links:\n" + "\n".join(ref_lines)
        return answer

    except Exception as e:
        logger.error(f"Error in generate_answer_with_rag: {e}", exc_info=True)
        return f"An error occurred while trying to generate an answer. Details: {e}"

# --- Your original retrieval function (modified and simplified) ---
# NOTE: The complex keyword filtering and score combining logic has been removed.
# A pure semantic search is more robust and aligns better with LLM generation.
def retrieve_relevant_faqs(query: str, top_k: int = 3) -> List[Dict[str, Any]]:
    logger = logging.getLogger(__name__)
    try:
        db = get_legal_db() 
        # 1. Get all content collection names.
        all_db_collections = db.list_collection_names()
        exclude_collections = {
            'logs', 'users', 'admin_marking', 'admin_answers', 'keywords',
            'fs.files', 'fs.chunks', 'admin_users', 'system.views',
            'system.version', 'system.profile'
        }
        content_collections = [
            name for name in all_db_collections 
            if not name.startswith('system.') and not name.startswith('_') and name not in exclude_collections
        ]
        if not content_collections:
            logger.warning("No content collections found in the database to search.")
            return []
        # 2. Fetch all chunks.
        all_chunks = []
        for collection_name in content_collections:
            collection = db[collection_name]
            chunks = list(collection.find({}, {"_id": 1, "content_en": 1, "content_hi": 1, "embedding_en": 1, "embedding_hi": 1, "source": 1}))
            for chunk in chunks:
                chunk["_collection"] = collection_name
            all_chunks.extend(chunks)
        if not all_chunks:
            logger.warning("No documents (chunks) found in any of the content collections.")
            return []
        # 3. Determine query language and get embedding.
        is_hindi = re.search(r"[\u0900-\u097F]", query)
        emb_field = "embedding_hi" if is_hindi else "embedding_en"
        content_field = "content_hi" if is_hindi else "content_en"
        query_emb = get_embedding(query)
        # 4. Score each chunk based on cosine similarity.
        scored_chunks = []
        for chunk in all_chunks:
            chunk_emb = chunk.get(emb_field)
            if not chunk_emb:
                content = chunk.get(content_field) or chunk.get("content_en") or ""
                if content:
                    try:
                        chunk_emb = get_embedding(content)
                        db[chunk["_collection"]].update_one(
                            {"_id": chunk["_id"]},
                            {"$set": {emb_field: chunk_emb.tolist()}}
                        )
                    except Exception as e:
                        logger.error(f"Could not generate on-the-fly embedding for chunk {chunk.get('_id')}: {e}")
                        continue
                else:
                    continue
            try:
                sim = cosine_similarity(query_emb, chunk_emb)
                scored_chunks.append((sim, chunk))
            except Exception as e:
                logger.warning(f"Could not calculate similarity for chunk {chunk.get('_id')}: {e}")
        if not scored_chunks:
            logger.info("No chunks were scored successfully.")
            return []
        # 5. Sort by semantic similarity and return top results.
        scored_chunks.sort(reverse=True, key=lambda x: x[0])
        top_results = []
        for sim, chunk in scored_chunks[:top_k]:
            result = {
                "score": sim,
                "source": chunk.get("source"),
                "content_en": chunk.get("content_en"),
                "content_hi": chunk.get("content_hi"),
                "_collection": chunk.get("_collection")
            }
            top_results.append(result)
        logger.info(f"Top {len(top_results)} matches retrieved.")
        return top_results
    except Exception as e:
        logger.error(f"Error in retrieve_relevant_faqs: {e}", exc_info=True)
        return []

# --- Your original formatting function (modified for simplicity) ---
# NOTE: Removed the unused `language` parameter to simplify.
def format_context_for_generation(faqs: List[Dict[str, Any]]) -> str:
    """
    Formats the retrieved chunks into a string for the LLM, including reference links for each chunk.
    """
    from backend.utils.reference_links import get_collection_reference_link
    # Deduplicate by reference link
    seen_links = set()
    unique_chunks = []
    for chunk in faqs:
        ref_link = ''
        if '_collection' in chunk:
            ref_link = get_collection_reference_link(chunk['_collection'])
        if not ref_link:
            ref_link = 'Not Available'
        if ref_link not in seen_links:
            seen_links.add(ref_link)
            chunk['__ref_link'] = ref_link
            unique_chunks.append(chunk)
    lines = []
    for i, chunk in enumerate(unique_chunks, 1):
        text = chunk.get('content_en', chunk.get('content_hi', ''))
        topic = chunk.get('topic', '')
        if not topic and '_collection' in chunk:
            topic = chunk['_collection'].replace('_', ' ').title()
        ref_link = chunk.get('__ref_link', 'Not Available')
        ref_str = f"Reference Link: {ref_link}"
        lines.append(f"Source {i} (Topic: {topic}):\n{text}\n{ref_str}\n")
    return "\n".join(lines)