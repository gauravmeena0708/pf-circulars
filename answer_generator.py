# answer_generator.py

import logging
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace # <--- ADD THIS
from langchain_core.messages import HumanMessage             # <--- ADD THIS
import config

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)

def format_prompt(query, retrieved_chunks_data):
    # This function can remain the same, as it produces the string prompt
    if not retrieved_chunks_data:
        context_str = "No relevant information found in the documents."
    else:
        context_parts = []
        for i, chunk_data in enumerate(retrieved_chunks_data):
            source_info = f"[Source PDF: {chunk_data['metadata'].get('source_pdf', 'N/A')}, Page: {chunk_data['metadata'].get('page_number', 'N/A')}]"
            context_parts.append(f"Context Chunk {i+1} {source_info}:\n{chunk_data['text']}")
        context_str = "\n\n".join(context_parts)

    prompt = f"""
You are a helpful AI assistant. Answer the question based on the following context extracted from relevant documents.
The context may contain tables that span multiple original pages but are now presented as a single merged text.
Synthesize information from the entire provided context to answer comprehensively.
If the context does not contain the answer, state that the information is not found in the provided documents.

Context from documents:
-----------------------
{context_str}
-----------------------

Question: {query}

Helpful Answer:
"""
    return prompt

def get_llm_answer(query, retrieved_chunks_data, llm_instance): # llm_instance will now be ChatHuggingFace
    if not query:
        logger.warning("Query is empty. Cannot generate answer.")
        return "No query provided."
    if llm_instance is None:
        logger.error("LLM instance is not provided. Cannot generate answer.")
        return "LLM not available."

    prompt_string = format_prompt(query, retrieved_chunks_data) # Still get the formatted string
    logger.debug(f"Formatted Prompt String for Chat LLM:\n{prompt_string}")

    try:
        logger.info(f"Sending prompt to Chat LLM for query: '{query[:100]}...'")
        
        # For Chat models, input is typically a list of messages
        messages = [HumanMessage(content=prompt_string)]
        response_message = llm_instance.invoke(messages) # Invoke with the list of messages
        
        logger.info("Received response from Chat LLM.")
        
        # The response is an AIMessage object, extract its content
        if hasattr(response_message, 'content'):
            return response_message.content
        else:
            logger.error(f"Unexpected response type from Chat LLM: {type(response_message)}. Full response: {response_message}")
            return str(response_message) # Fallback

    except Exception as e:
        logger.error(f"Error during Chat LLM invocation: {e}", exc_info=True)
        return "An error occurred while trying to generate an answer from the language model."


def initialize_llm():
    if not config.HF_TOKEN:
        logger.error("Hugging Face API token (HF_TOKEN) is not set. LLM cannot be initialized.")
        raise ValueError("HF_TOKEN not found. LLM initialization failed.")

    try:
        logger.info(f"Initializing Chat LLM via HuggingFaceEndpoint: {config.LLM_REPO_ID}, Task: {config.LLM_TASK}")
        endpoint = HuggingFaceEndpoint(
            repo_id=config.LLM_REPO_ID,
            task=config.LLM_TASK, # Make sure this is "conversational" in config.py
            temperature=config.LLM_TEMPERATURE,
            max_new_tokens=config.LLM_MAX_NEW_TOKENS,
            huggingfacehub_api_token=config.HF_TOKEN
        )
        chat_model = ChatHuggingFace(llm=endpoint) # Try this first.
        logger.info("ChatHuggingFace LLM initialized successfully.")
        return chat_model
    except Exception as e:
        logger.error(f"Failed to initialize Chat LLM: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    # ... (This section remains the same for testing answer_generator.py directly) ...
    logger.info("Starting Answer Generator example...")
    try:
        llm_service = initialize_llm()
    except ValueError as e:
        logger.error(f"LLM Initialization Error: {e}")
        llm_service = None
    except Exception as e:
        logger.error(f"A general error occurred during LLM Initialization: {e}")
        llm_service = None

    if llm_service:
        example_query = "What are the main challenges in adopting renewable energy?"
        example_retrieved_data = [
            {"text": "Sample context about energy challenges.", "metadata": {"source_pdf": "dummy.pdf", "page_number": 1}}
        ]
        logger.info(f"Generating answer for query: '{example_query}'")
        answer = get_llm_answer(example_query, example_retrieved_data, llm_service)
        logger.info(f"\n--- Generated Answer ---\n{answer}\n------------------------")
    else:
        logger.warning("LLM service not available. Skipping answer generation examples.")
    logger.info("Answer Generator example finished.")