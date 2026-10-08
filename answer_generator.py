# answer_generator.py

import logging
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_core.messages import HumanMessage             
import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=config.LOG_LEVEL, format=config.LOG_FORMAT)


def describe_llm_error(e):
    """Maps a provider failure to a user-facing reason without leaking the raw
    exception text (which can include request URLs or account details)."""
    provider = getattr(config, "HF_INFERENCE_PROVIDER", "configured")
    response = getattr(e, "response", None)
    status = getattr(response, "status_code", None)
    if status == 401:
        return "The Hugging Face token was rejected (401). Check that HF_TOKEN is valid."
    if status == 402:
        return (
            "The Hugging Face account behind HF_TOKEN has used up its Inference "
            "Providers credits (402). Add credits/upgrade to PRO, or supply your own token."
        )
    if status == 403:
        return (
            "HF_TOKEN is not allowed to call Inference Providers (403). Enable "
            "'Make calls to Inference Providers' on the token."
        )
    if status == 404:
        return (
            f"Model '{config.LLM_REPO_ID}' is not served by provider '{provider}' (404). "
            "Change LLM_REPO_ID or HF_INFERENCE_PROVIDER."
        )
    if status == 429:
        return "The language-model provider is rate-limiting requests (429). Please retry shortly."
    suffix = f" (HTTP {status})" if status else ""
    return (
        f"The language-model service is temporarily unavailable{suffix}. "
        f"Verify that the Hugging Face provider '{provider}' is enabled and "
        "that HF_TOKEN has Inference Providers permission."
    )


def _stream_llm_answer(llm_instance, messages):
    """Iterate the remote stream while keeping provider failures out of the UI."""
    try:
        yield from llm_instance.stream(messages)
    except Exception as e:
        logger.error(f"Error while streaming the Chat LLM response: {e}", exc_info=True)
        yield describe_llm_error(e)


ANSWER_LANGUAGES = ("English", "Hindi")


def language_instruction(answer_language="English"):
    """Returns a prompt line asking for the answer in the chosen language, or
    "" for English (the prompts' default). Citation markers, circular numbers
    and statutory references stay verbatim so they still match the sources."""
    if answer_language == "Hindi":
        return (
            "Write the entire answer in Hindi (Devanagari script), even though the "
            "sources are in English. Keep source numbers such as [1], circular "
            "numbers, dates, section numbers, and official scheme names exactly as "
            "they appear in the sources."
        )
    return ""


def _sanitize_conversation_context(conversation_context):
    """Neutralizes text in conversation_context that could be mistaken for
    our own prompt delimiters.

    conversation_context is built from earlier turns' assistant answers --
    LLM output nominally grounded in retrieved documents, but those
    documents are bulk-ingested, untrusted PDFs. If a prior answer happens
    to echo (or hallucinate) text resembling the "PREVIOUS CONVERSATION"
    markers below, splicing it in verbatim could let it prematurely close
    that block and have the remainder read as new instructions.
    """
    if not conversation_context:
        return conversation_context
    return (
        conversation_context.replace(
            "--- END PREVIOUS CONVERSATION ---", "[conversation marker omitted]"
        ).replace("--- PREVIOUS CONVERSATION", "[conversation marker omitted]")
    )


def format_prompt(query, retrieved_chunks_data, conversation_context="", answer_language="English"):
    conversation_context = _sanitize_conversation_context(conversation_context)
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

    history_section = (
        f"\n--- PREVIOUS CONVERSATION (for follow-up context only; "
        f"ground every factual claim in the numbered sources above, not in "
        f"prior turns) ---\n{conversation_context}\n--- END PREVIOUS CONVERSATION ---\n"
        if conversation_context
        else ""
    )

    prompt = f"""You are a helpful and precise assistant specializing in Employees' Provident Fund Organisation (EPFO) rules, circulars, schemes, and manuals.
Answer the user's question based strictly on the context provided below.
Support factual claims with inline source numbers such as [1] or [2], matching the numbered sources below the answer.
Also mention relevant circular numbers, dates, or statutory sections when they are present in the context.
Never invent a source number or cite a source that does not support the claim.
If the provided context does not contain enough information to answer the question, state clearly that the information was not found in the documents.
{language_instruction(answer_language)}

Context from EPFO Documents:
-----------------------
{context_str}
-----------------------
{history_section}
Question: {query}

Helpful & Grounded Answer:"""
    return prompt


def get_llm_answer(query, retrieved_chunks_data, llm_instance, stream=False, conversation_context="", answer_language="English"):
    if not query:
        logger.warning("Query is empty. Cannot generate answer.")
        return "No query provided."
    if llm_instance is None:
        logger.error("LLM instance is not provided. Cannot generate answer.")
        return "LLM not available."

    prompt_string = format_prompt(
        query,
        retrieved_chunks_data,
        conversation_context=conversation_context,
        answer_language=answer_language,
    )
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
        return describe_llm_error(e)


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
        chat_kwargs = {}
        reasoning_effort = getattr(config, "LLM_REASONING_EFFORT", "")
        if reasoning_effort:
            # chat_completion() forwards extra_body verbatim to the provider.
            chat_kwargs["model_kwargs"] = {"extra_body": {"reasoning_effort": reasoning_effort}}
        chat_model = ChatHuggingFace(llm=endpoint, **chat_kwargs)
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
