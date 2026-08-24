# answer_generator.py

import logging
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_core.messages import HumanMessage             
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


def _stream_llm_answer(llm_instance, messages):
    """Iterate the remote stream while keeping provider failures out of the UI."""
    try:
        yield from llm_instance.stream(messages)
    except Exception as e:
        logger.error(f"Error while streaming the Chat LLM response: {e}", exc_info=True)
        provider = getattr(config, "HF_INFERENCE_PROVIDER", "configured")
        yield (
            "The language-model service is temporarily unavailable. "
            f"Verify that the Hugging Face provider '{provider}' is enabled and "
            "that HF_TOKEN has Inference Providers permission."
        )


def format_prompt(query, retrieved_chunks_data):
    if not retrieved_chunks_data:
        context_str = "No relevant information found in the documents."
    else:
        context_parts = []
        for i, chunk_data in enumerate(retrieved_chunks_data):
            meta = chunk_data.get('metadata', {})
            title = meta.get('title') or "EPFO Document"
            circular_no = meta.get('circular_no') or "N/A"
            source_pdf = meta.get('english_pdf_link') or meta.get('source_pdf', 'N/A')
            page_no = meta.get('page_number', 'N/A')
            source_info = f"[Title: {title} | Identifier: {circular_no} | PDF: {source_pdf} | Page: {page_no}]"
            context_parts.append(f"Source [{i+1}] {source_info}:\n{chunk_data['text']}")
        context_str = "\n\n".join(context_parts)

    prompt = f"""You are a helpful and precise assistant specializing in Employees' Provident Fund Organisation (EPFO) rules, circulars, schemes, and manuals.
Answer the user's question based strictly on the context provided below.
Support factual claims with inline source numbers such as [1] or [2], matching the numbered sources below the answer.
Also mention relevant circular numbers, dates, or statutory sections when they are present in the context.
Never invent a source number or cite a source that does not support the claim.
If the provided context does not contain enough information to answer the question, state clearly that the information was not found in the documents.

Context from EPFO Documents:
-----------------------
{context_str}
-----------------------

Question: {query}

Helpful & Grounded Answer:"""
    return prompt


def get_llm_answer(query, retrieved_chunks_data, llm_instance, stream=False):
    if not query:
        logger.warning("Query is empty. Cannot generate answer.")
        return "No query provided."
    if llm_instance is None:
        logger.error("LLM instance is not provided. Cannot generate answer.")
        return "LLM not available."

    prompt_string = format_prompt(query, retrieved_chunks_data) 
    logger.debug(f"Formatted Prompt String for Chat LLM:\n{prompt_string}")

    logger.info(f"Sending prompt to Chat LLM for query: '{query[:100]}...'")
    messages = [HumanMessage(content=prompt_string)]

    if stream:
        return _stream_llm_answer(llm_instance, messages)

    try:
        response_message = llm_instance.invoke(messages)
        logger.info("Received response from Chat LLM.")
        if hasattr(response_message, 'content'):
            return response_message.content
        else:
            logger.error(f"Unexpected response type from Chat LLM: {type(response_message)}. Full response: {response_message}")
            return str(response_message)

    except Exception as e:
        logger.error(f"Error during Chat LLM invocation: {e}", exc_info=True)
        return "An error occurred while trying to generate an answer from the language model."


def initialize_llm(hf_token=None, max_new_tokens=None):
    token = hf_token or config.HF_TOKEN
    if not token:
        logger.error("Hugging Face API token (HF_TOKEN) is not set. LLM cannot be initialized.")
        raise ValueError("HF_TOKEN not found. LLM initialization failed.")

    try:
        logger.info(
            f"Initializing Chat LLM via HuggingFaceEndpoint: {config.LLM_REPO_ID}, "
            f"Task: {config.LLM_TASK}"
        )
        kwargs = {
            "repo_id": config.LLM_REPO_ID,
            "task": config.LLM_TASK,
            "temperature": config.LLM_TEMPERATURE,
            "max_new_tokens": max_new_tokens or getattr(config, "LLM_MAX_NEW_TOKENS", 2048),
            "huggingfacehub_api_token": token,
        }
        if getattr(config, "HF_INFERENCE_PROVIDER", None):
            kwargs["provider"] = config.HF_INFERENCE_PROVIDER

        endpoint = HuggingFaceEndpoint(**kwargs)
        chat_model = ChatHuggingFace(llm=endpoint)
        logger.info("ChatHuggingFace LLM initialized successfully.")
        return chat_model
    except Exception as e:
        logger.error(f"Failed to initialize Chat LLM: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    logger.info("Starting Answer Generator test...")
    try:
        llm_service = initialize_llm()
        sample_query = "What is the procedure for joint declaration?"
        sample_context = [{
            "text": "Joint declaration SOP outlines the procedure for member profile correction.",
            "metadata": {"title": "SOP Joint Declaration", "source_pdf": "Circular_JD.pdf", "page_number": "1"}
        }]
        print(get_llm_answer(sample_query, sample_context, llm_service))
    except Exception as e:
        logger.info(f"Test run completed: {e}")
