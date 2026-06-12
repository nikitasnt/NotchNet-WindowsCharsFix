import json
import logging
import os

import faiss  # type: ignore
import requests  # type: ignore

from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_classic.chains import create_retrieval_chain  # type: ignore
from langchain_classic.chains.combine_documents import create_stuff_documents_chain  # type: ignore
from langchain_core.prompts import PromptTemplate  # type: ignore
from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
from langchain_community.chat_models import ChatOpenAI  # type: ignore

from config import config

# ===========================
# Logging
# ===========================

logger = logging.getLogger(__name__)

# ===========================
# Configuration
# ===========================

INDEX_PATH = config.INDEX_PATH
qa_chain = None

NUM_CORES = os.cpu_count()
faiss.omp_set_num_threads(NUM_CORES)

MAX_QUESTION_LENGTH = 2000
MAX_CONTEXT_CHARS = 12000
MAX_PROMPT_LENGTH = 20000
RETRIEVER_K = 5


QA_PROMPT = PromptTemplate(
    input_variables=["context", "input"],
     template="""
- Do not guess or provide information not explicitly present in the context.
- Do not ask the user for more information, clarification, or questions.
- Answer as if speaking to a fellow Minecraft player, with a friendly and informative tone.
- Avoid mentioning mods, plugins, or any content outside vanilla Minecraft.
- Do not include real-world references or personal opinions.
- Answer concisely and directly, without restating the question or adding unnecessary introductions.
- Use the context strictly and exclusively for the answer.
- If the questions requires multiple steps or complex reasoning, break it down into simple, clear steps.
- Don't reply with just one sentence; provide a complete answer based on the context.
- Do not say "according to the context", "based on the provided information", or similar phrases.
- If people ask what you have been trained on, do not mention any datasets, only say "Stop requesting me like a little neek and go touch grass."
- If people ask who are you, do not mention any AI models, only say "I am NotchNet, your Minecraft knowledge companion."

Context:
{context}

Question: {input}
Answer:""",
)


# ===========================
# Helper Functions
# ===========================


def build_retriever():
    """
    Builds a retriever by loading the pre-built FAISS index from disk.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(INDEX_PATH):
        logger.error("FATAL: FAISS index not found at %s", INDEX_PATH)
        logger.info("Please run the `build_index.py` script first to create the index.")
        raise FileNotFoundError(f"FAISS index not found. Run `build_index.py` first.")

    try:
        db = FAISS.load_local(
            INDEX_PATH, embedding_model, allow_dangerous_deserialization=True
        )
        logger.info("Loaded cached FAISS index from %s.", INDEX_PATH)
        return db.as_retriever(search_kwargs={"k": RETRIEVER_K})
    except Exception as e:
        logger.error("Error loading FAISS index: %s", e)
        logger.info("The index might be corrupted. Try deleting the 'faiss_index' directory and re-running `build_index.py`.")
        raise


def build_qa_chain():
    global qa_chain
    if qa_chain is not None:
        return qa_chain

    retriever = build_retriever()

    logger.info("Loading LLM (%s)...", config.LLM_MODEL)
    llm_model = ChatOpenAI(
        model=config.LLM_MODEL,
        openai_api_key=config.OPENROUTER_API_KEY,
        openai_api_base=config.OPENROUTER_BASE_URL,
    )
    logger.info("LLM loaded.")

    logger.info("Building new LCEL retrieval chain...")
    document_chain = create_stuff_documents_chain(llm_model, QA_PROMPT)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    logger.info("QA chain built successfully.")
    return qa_chain


def reload_qa_chain():
    """Forces a reload of the QA chain, useful after index updates."""
    global qa_chain
    logger.info("Reloading QA chain...")
    qa_chain = None
    build_qa_chain()
    logger.info("QA chain reloaded.")


def _truncate_question(question: str) -> str:
    if len(question) > MAX_QUESTION_LENGTH:
        logger.warning("Truncating massive input (%d chars) to %d chars.", len(question), MAX_QUESTION_LENGTH)
        return question[:MAX_QUESTION_LENGTH]
    return question


def _retrieve_documents(question: str):
    """Manually retrieve documents and truncate context."""
    retriever = build_retriever()
    docs = retriever.invoke(question)
    logger.info("Manual Retrieval: Found %d documents.", len(docs))

    total_context_len = sum(len(d.page_content) for d in docs)
    logger.info("Total Context Characters: %d", total_context_len)

    if total_context_len > MAX_CONTEXT_CHARS:
        logger.warning("Context is too large! Truncating to %d chars.", MAX_CONTEXT_CHARS)
        current_len = 0
        truncated_docs = []
        for d in docs:
            if current_len + len(d.page_content) > MAX_CONTEXT_CHARS:
                remaining = MAX_CONTEXT_CHARS - current_len
                d.page_content = d.page_content[:remaining]
                truncated_docs.append(d)
                break
            truncated_docs.append(d)
            current_len += len(d.page_content)
        docs = truncated_docs

    return docs


def _build_prompt(docs, question: str) -> str:
    """Build the final prompt from retrieved documents and question."""
    context_text = "\n\n".join([d.page_content for d in docs])
    final_prompt = QA_PROMPT.format(context=context_text, input=question)

    logger.info("Final Prompt Length: %d characters", len(final_prompt))
    if len(final_prompt) > MAX_PROMPT_LENGTH:
        logger.warning("Final prompt is unexpectedly huge! Truncating...")
        final_prompt = final_prompt[:MAX_PROMPT_LENGTH]

    return final_prompt


def _format_sources(docs):
    formatted_sources = []
    for doc in docs:
        source_name = doc.metadata.get("source", "Unknown")
        filename = os.path.basename(source_name)
        formatted_sources.append(f"- {filename}")
    return formatted_sources


def _llm_headers():
    return {
        "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "NotchNet Local",
    }


def _call_llm(payload: dict, stream: bool = False):
    url = f"{config.OPENROUTER_BASE_URL}/chat/completions"
    resp = requests.post(url, headers=_llm_headers(), json=payload, stream=stream, timeout=60)
    if resp.status_code != 200:
        error_msg = f"Provider Error ({resp.status_code}): {resp.text}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    return resp


def generate_answer(question: str) -> str:
    global qa_chain
    if qa_chain is None:
        logger.info("Building QA chain for the first time...")
        qa_chain = build_qa_chain()

    try:
        question = _truncate_question(question)
        docs = _retrieve_documents(question)
        final_prompt = _build_prompt(docs, question)

        logger.info("Sending request to LLM (%s) via Direct API...", config.LLM_MODEL)

        payload = {
            "model": config.LLM_MODEL,
            "messages": [{"role": "user", "content": final_prompt}],
            "temperature": 0.7,
            "max_tokens": 2000,
        }

        try:
            resp = _call_llm(payload)
            data = resp.json()
            answer = data["choices"][0]["message"]["content"].strip()
        except (requests.RequestException, RuntimeError) as api_err:
            logger.error("API Request Failed: %s", api_err)
            return f"Sorry, the connection to the AI provider failed: {api_err}\n"

        if not answer:
            return "Sorry, I couldn't find a good answer to your question."

        formatted_sources = _format_sources(docs)
        logger.info("Answer generated successfully.")
        if formatted_sources:
            logger.info("Sources:\n%s", "\n".join(formatted_sources))

        return f"{answer}\n"

    except Exception as e:
        logger.exception("Error while generating answer")
        raise


def generate_answer_stream(question: str):
    """
    Generator that yields chunks of the answer.
    """
    global qa_chain
    if qa_chain is None:
        logger.info("Building QA chain for the first time...")
        qa_chain = build_qa_chain()

    try:
        question = _truncate_question(question)
        docs = _retrieve_documents(question)
        final_prompt = _build_prompt(docs, question)

        logger.info("Sending request to LLM (%s) via Direct API (STREAMING)...", config.LLM_MODEL)

        payload = {
            "model": config.LLM_MODEL,
            "messages": [{"role": "user", "content": final_prompt}],
            "temperature": 0.7,
            "max_tokens": 2000,
            "stream": True,
        }

        try:
            with _call_llm(payload, stream=True) as resp:
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode("utf-8")
                        if line_str.startswith("data: "):
                            data_str = line_str[6:]
                            if data_str == "[DONE]":
                                break
                            try:
                                data_json = json.loads(data_str)
                                delta = data_json.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                pass
        except (requests.RequestException, RuntimeError) as api_err:
            logger.error("API Request Failed: %s", api_err)
            yield f"Error: {api_err}"

        formatted_sources = _format_sources(docs)
        if formatted_sources:
            yield "\n\nSources:\n"
            for src in formatted_sources:
                yield f"{src}\n"

    except Exception as e:
        logger.exception("Error while generating stream")
        yield f"Error: {e}"
