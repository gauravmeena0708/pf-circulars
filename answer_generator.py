# answer_generator.py

import logging
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.chat_models import ChatHuggingFace
from langchain_core.messages import HumanMessage             
import config
import json
import os
import re
from citation_provenance import page_label

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
    if status == 413:
        return ("The evidence request exceeds the language-model provider's size or token limit (413). "
                "Try a narrower question or a shorter evidence history. Token permissions are not the cause of this error.")
    if status == 429:
        return "The language-model provider is rate-limiting requests (429). Please retry shortly."
    suffix = f" (HTTP {status})" if status else ""
    return (
        f"The language-model service is temporarily unavailable{suffix}. "
        f"Verify that the Hugging Face provider '{provider}' is enabled and "
        "that HF_TOKEN has Inference Providers permission."
    )


def _stream_llm_answer(llm_instance, messages, retry_messages=None):
    """Iterate the remote stream while keeping provider failures out of the UI."""
    emitted = False
    try:
        for part in llm_instance.stream(messages):
            emitted = True
            yield part
    except Exception as e:
        if not emitted and retry_messages and getattr(getattr(e, "response", None), "status_code", None) == 413:
            yield from _stream_llm_answer(llm_instance, retry_messages)
            return
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


def format_prompt(query, retrieved_chunks_data, conversation_context="", answer_language="English", policy_context=None):
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
            page_no = page_label(meta)
            evidence_status = meta.get("evidence_status", "retrieved source; applicability unverified")
            source_info = f"[Title: {title} | Identifier: {circular_no} | Date: {meta.get('date', 'unknown')} | PDF: {source_pdf} | Page: {page_no} | Authority: {meta.get('authority', 'not classified')} | Evidence: {evidence_status}]"
            context_parts.append(f"Source [{chunk_data.get('citation_number', i+1)}] {source_info}:\n{chunk_data['text']}")
        context_str = "\n\n".join(context_parts)

    policy_section = ""
    if policy_context:
        policy_section = (
            "\nPOLICY EVIDENCE REVIEW (curated metadata, not an additional numbered source):\n"
            + json.dumps(policy_context, ensure_ascii=False)
            + "\nExplain changes by issue date, accounting for later superseding directions and amendments. "
            "A numbered SOP or established commencement date does not establish the latest applicable policy. "
            "Never invent version numbers. Only source_checked relationships are verified. Never promote an unverified target or unknown commencement to fact; missing originals remain unverified. "
            "Describe explicit supersession even if timing is unresolved; do not call it merely implementation of an unchanged older SOP. "
            "Partial changes affect only their stated scope; established supersession displaces directions only within that scope and commencement. "
            "Distinguish Board approvals, press reports, software dates, processing limits and deadlines from operative rules and entitlement. "
            "Supplementary hybrid sources are unverified for amendment and applicability. "
            "State material gaps and unknown dates; never claim complete applicable rules, current validity or live implementation. "
            "Cite numbered source pages, never metadata alone.\n"

        )
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
Where page provenance is unverified, cite the document without inventing a page.
Relevance or recent issue date does not establish that a provision remains applicable.
If the provided context does not contain enough information to answer the question, state clearly that the information was not found in the documents.
{language_instruction(answer_language)}

Context from EPFO Documents:
-----------------------
{context_str}
-----------------------
{history_section}
{policy_section}
Question: {query}

Helpful & Grounded Answer:"""
    return prompt


def bounded_prompt(query, chunks, conversation_context="", answer_language="English", policy_context=None, limit=18000):
    """Bound the complete request, preserving original UI citation numbers.

    Omit whole source pages instead of cutting instructions out of context. The
    model is told about reduced coverage. Retry uses the same numbered sources.
    """
    if len(query) > 6000:
        raise ValueError("Please shorten the question to 6,000 characters.")
    selected = [{**c, "citation_number": c.get("citation_number", i + 1)} for i, c in enumerate(chunks)]
    latest_query = bool(re.search(r"\blatest\b|\bcurrent\b|most recent|नवीनतम|वर्तमान", query, re.I))
    if latest_query or policy_context:
        selected.sort(key=lambda c: (str(c.get("metadata", {}).get("date", "")),
                                    bool(c.get("metadata", {}).get("policy_change_evidence") or
                                         "relationship or commencement" in c.get("metadata", {}).get("evidence_status", ""))), reverse=True)
        # One page per document before further pages of the same document.
        first, rest, seen = [], [], set()
        for item in selected:
            meta = item.get("metadata", {})
            key = (meta.get("date"), meta.get("circular_no"), meta.get("title"))
            (rest if key in seen else first).append(item)
            seen.add(key)
        selected = first + rest
    context = None
    if policy_context:
        context = {**policy_context}
        # Page text already supplies the evidence. Do not repeat long OCR quotes.
        context["relationships"] = [{k: v for k, v in r.items() if k not in ("evidence", "effective_date_evidence")} for r in context.get("relationships", [])]
    if context and len(json.dumps(context, ensure_ascii=False).encode("utf-8")) > limit // 4:
        # Reserve the request for actual numbered evidence, including on 413 retry.
        context = {
            "as_of": policy_context.get("as_of"), "scope": policy_context.get("scope"),
            "conclusion": "complete_applicability_not_established",
            "documents": [{k: d.get(k) for k in ("id", "title", "issued_on", "status")}
                          for d in sorted(policy_context.get("documents", []), key=lambda d: d.get("issued_on", ""), reverse=True)[:4]],
            "relationships": [{k: r.get(k) for k in ("source", "target", "type", "scope", "verification", "status")}
                              for r in policy_context.get("relationships", [])[-3:]],
            "limitations": ["Review metadata condensed for request limits. Full inventory and gaps are displayed separately. Unknown commencement and missing originals remain unverified; complete applicability is not established."],
        }
    if context and limit <= 9000:
        context = {
            "as_of": context.get("as_of"), "conclusion": "complete_applicability_not_established",
            "recent_documents_id_date_status": [[d.get("id"), d.get("issued_on"), d.get("status")] for d in context.get("documents", [])],
            "checked_changes_source_target_type": [[r.get("source"), r.get("target"), r.get("type")] for r in context.get("relationships", []) if r.get("verification") == "source_checked"],
            "limitations": ["Inventory condensed; full review is in UI. Missing originals and commencement remain unverified. No complete applicability claim."],
        }
    history = conversation_context[-1500:]
    while True:
        prompt = format_prompt(query, selected, history, answer_language, context)
        if len(prompt.encode("utf-8")) <= limit:
            return prompt
        if history:
            history = ""
        elif selected:
            # Keep relationship/commencement pages whenever other pages can go.
            if latest_query or policy_context:
                # Keep the newest dated change pages before historical controls.
                drop = len(selected) - 1
            else:
                drop = next((i for i in range(len(selected)-1, -1, -1)
                             if "relationship or commencement" not in selected[i].get("metadata", {}).get("evidence_status", "")), len(selected)-1)
            selected.pop(drop)
            if context:
                context["prompt_coverage_reduced"] = True
                if "Source pages omitted to meet provider request limits; answer coverage is incomplete." not in context["limitations"]:
                    context["limitations"] = list(context["limitations"]) + ["Source pages omitted to meet provider request limits; answer coverage is incomplete."]
        elif context:
            context = {"as_of": context.get("as_of"), "scope": context.get("scope"),
                       "conclusion": "complete_applicability_not_established",
                       "limitations": ["Policy inventory omitted to meet request limits; no complete applicability claim is supported."]}
            if len(format_prompt(query, [], "", answer_language, context).encode("utf-8")) > limit:
                raise ValueError("The question is too long for the configured request limit.")
        else:
            raise ValueError("The question is too long for the configured request limit.")


def get_llm_answer(query, retrieved_chunks_data, llm_instance, stream=False, conversation_context="", answer_language="English", policy_context=None):
    if not query:
        logger.warning("Query is empty. Cannot generate answer.")
        return "No query provided."
    if llm_instance is None:
        logger.error("LLM instance is not provided. Cannot generate answer.")
        return "LLM not available."

    try:
        prompt_string = bounded_prompt(query, retrieved_chunks_data, conversation_context,
                                       answer_language, policy_context,
                                       limit=int(os.environ.get("LLM_MAX_PROMPT_BYTES", "18000")))
        retry_prompt = bounded_prompt(query, retrieved_chunks_data, "", answer_language,
                                     policy_context, limit=9000)
    except ValueError as error:
        return str(error)
    logger.debug(f"Formatted Prompt String for Chat LLM:\n{prompt_string}")

    logger.info(f"Sending prompt to Chat LLM for query: '{query[:100]}...'")
    messages = [HumanMessage(content=prompt_string)]

    if stream:
        return _stream_llm_answer(llm_instance, messages, [HumanMessage(content=retry_prompt)])

    try:
        try:
            response_message = llm_instance.invoke(messages)
        except Exception as error:
            if getattr(getattr(error, "response", None), "status_code", None) != 413:
                raise
            response_message = llm_instance.invoke([HumanMessage(content=retry_prompt)])
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
