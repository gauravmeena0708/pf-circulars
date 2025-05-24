# answer_generator.py

import logging
#from langchain_huggingface import HuggingFaceHub # Using Langchain's HuggingFaceHub
from langchain_community.llms import HuggingFaceHub # Using Langchain's Community LLMs
# Assuming your config.py is in the same directory or accessible in PYTHONPATH
import config

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)

def format_prompt(query, retrieved_chunks_data):
    """
    Formats the prompt for the LLM using the query and retrieved chunks.

    Args:
        query (str): The user's original query.
        retrieved_chunks_data (list): A list of dictionaries, where each dict contains
                                      'text' (the retrieved chunk) and 'metadata'.

    Returns:
        str: A formatted prompt string.
    """
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
If the context does not contain the answer, state that the information is not found in the provided documents.
Do not make up information outside of the provided context.

Context from documents:
-----------------------
{context_str}
-----------------------

Question: {query}

Helpful Answer:
"""
    return prompt

def get_llm_answer(query, retrieved_chunks_data, llm_instance):
    """
    Generates an answer from the LLM using the query and retrieved chunks.

    Args:
        query (str): The user's original query.
        retrieved_chunks_data (list): Output from retriever.retrieve_relevant_chunks.
        llm_instance (HuggingFaceHub): An initialized HuggingFaceHub LLM instance.

    Returns:
        str: The answer from the LLM, or an error message if generation fails.
    """
    if not query:
        logger.warning("Query is empty. Cannot generate answer.")
        return "No query provided."
    if llm_instance is None:
        logger.error("LLM instance is not provided. Cannot generate answer.")
        return "LLM not available."

    prompt = format_prompt(query, retrieved_chunks_data)
    logger.debug(f"Formatted Prompt for LLM:\n{prompt}")

    try:
        logger.info(f"Sending prompt to LLM for query: '{query[:100]}...'")
        response = llm_instance.invoke(prompt)
        logger.info("Received response from LLM.")
        return response
    except Exception as e:
        logger.error(f"Error during LLM invocation: {e}", exc_info=True)
        return "An error occurred while trying to generate an answer from the language model."


def initialize_llm():
    """
    Initializes and returns the HuggingFaceHub LLM instance.
    """
    if not config.HF_TOKEN:
        logger.error("Hugging Face API token (HF_TOKEN) is not set in config/environment. LLM cannot be initialized.")
        # You might want to raise an error here or handle it more gracefully depending on application flow
        raise ValueError("HF_TOKEN not found. LLM initialization failed.")

    try:
        logger.info(f"Initializing LLM: {config.LLM_REPO_ID} with task: {config.LLM_TASK}")
        llm = HuggingFaceHub(
            repo_id=config.LLM_REPO_ID,
            task=config.LLM_TASK,
            model_kwargs={
                "temperature": config.LLM_TEMPERATURE,
                "max_new_tokens": config.LLM_MAX_NEW_TOKENS,
                # Add other specific kwargs if needed, e.g., "return_full_text": False for some models/tasks
            },
            huggingfacehub_api_token=config.HF_TOKEN
        )
        logger.info("LLM initialized successfully.")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
        raise # Re-raise the exception to be caught by the caller


if __name__ == '__main__':
    logger.info("Starting Answer Generator example...")

    # --- Initialize LLM (this would typically be done once in main.py) ---
    try:
        llm_service = initialize_llm()
    except ValueError as e: # Catch the specific error for missing token
        logger.error(f"LLM Initialization Error: {e}")
        llm_service = None # Ensure llm_service is defined for further checks
    except Exception as e: # Catch any other initialization errors
        logger.error(f"A general error occurred during LLM Initialization: {e}")
        llm_service = None


    if llm_service:
        # --- Example Query and Retrieved Chunks (simulate output from retriever.py) ---
        example_query = "What are the main challenges in adopting renewable energy?"
        example_retrieved_data = [
            {
                "text": "Adopting renewable energy sources like solar and wind power faces challenges such as intermittency, grid integration, and high initial investment costs. However, long-term benefits include reduced carbon emissions.",
                "metadata": {"source_pdf": "energy_report.pdf", "page_number": 5},
                "score": 0.85
            },
            {
                "text": "Energy storage solutions, like batteries, are crucial for addressing the intermittency of renewables but add to the overall system cost.",
                "metadata": {"source_pdf": "tech_review.pdf", "page_number": 12},
                "score": 0.78
            },
            {
                "text": "Policy support and public acceptance also play a significant role in the widespread adoption of green energy technologies.",
                "metadata": {"source_pdf": "policy_brief.pdf", "page_number": 2},
                "score": 0.70
            }
        ]

        logger.info(f"Generating answer for query: '{example_query}'")
        answer = get_llm_answer(example_query, example_retrieved_data, llm_service)
        logger.info(f"\n--- Generated Answer ---\n{answer}\n------------------------")

        # Test with no retrieved chunks
        logger.info("Generating answer for query with NO retrieved chunks...")
        answer_no_context = get_llm_answer(example_query, [], llm_service)
        logger.info(f"\n--- Answer (No Context) ---\n{answer_no_context}\n---------------------------")
    else:
        logger.warning("LLM service not available. Skipping answer generation examples.")

    logger.info("Answer Generator example finished.")