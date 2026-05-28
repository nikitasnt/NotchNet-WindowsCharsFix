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
# Configuration
# ===========================

INDEX_PATH = config.INDEX_PATH
qa_chain = None

NUM_CORES = os.cpu_count()
faiss.omp_set_num_threads(NUM_CORES)


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
    # check_ollama() # No longer using Ollama

    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    if not os.path.exists(INDEX_PATH):
        print(f"❌ FATAL: FAISS index not found at {INDEX_PATH}")
        print("Please run the `build_index.py` script first to create the index.")
        raise FileNotFoundError(f"FAISS index not found. Run `build_index.py` first.")

    try:
        db = FAISS.load_local(
            INDEX_PATH, embedding_model, allow_dangerous_deserialization=True
        )
        print(f"🔁 Loaded cached FAISS index from {INDEX_PATH}.")
        return db.as_retriever(search_kwargs={"k": 5})
    except Exception as e:
        print(f"❌ Error loading FAISS index: {e}")
        print(
            "The index might be corrupted. Try deleting the 'faiss_index' directory and re-running `build_index.py`."
        )
        raise e


def build_qa_chain():
    global qa_chain
    if qa_chain is not None:
        return qa_chain

    retriever = build_retriever()

    print(f"🔧 Loading LLM ({config.LLM_MODEL})...")
    llm_model = ChatOpenAI(
        model=config.LLM_MODEL,
        openai_api_key=config.OPENROUTER_API_KEY,
        openai_api_base=config.OPENROUTER_BASE_URL,
    )
    print("✅ LLM loaded.")

    print("🔧 Building new LCEL retrieval chain...")
    document_chain = create_stuff_documents_chain(llm_model, QA_PROMPT)
    qa_chain = create_retrieval_chain(retriever, document_chain)

    print("✅ QA chain built successfully.")
    return qa_chain


def reload_qa_chain():
    """Forces a reload of the QA chain, useful after index updates."""
    global qa_chain
    print("🔄 Reloading QA chain...")
    qa_chain = None
    build_qa_chain()
    print("✅ QA chain reloaded.")


def generate_answer(question: str) -> str:
    global qa_chain
    if qa_chain is None:
        print("🔧 Building QA chain for the first time...")
        qa_chain = build_qa_chain()

    try:
        if len(question) > 2000:
            print(f"⚠️ Truncating massive input ({len(question)} chars) to 2000 chars.")
            question = question[:2000]

        # 1. Manually retrieve documents
        # We need access to the vector store or retriever directly.
        # Since 'qa_chain' hides it, let's access the retriever from it if possible, 
        # OR better: just rebuild/access the retriever here directly or cache it.
        # Actually, let's just make 'retriever' global or accessible.
        
        # But for now, we can extract it from the chain if constructed that way, 
        # OR just instantiate a temporary retriever/db search since we loaded the index.
        # Let's rely on build_retriever() returning a new one (cheap enough) or cache it.
        
        retriever = build_retriever()
        docs = retriever.invoke(question)
        
        print(f"🔎 Manual Retrieval: Found {len(docs)} documents.")
        
        # 2. Check and Truncate Context
        total_context_len = sum(len(d.page_content) for d in docs)
        print(f"📊 Total Context Characters: {total_context_len}")
        
        # Hard limit: 12,000 chars (approx 3k tokens) to be super safe
        MAX_CTX_CHARS = 12000
        if total_context_len > MAX_CTX_CHARS:
             print(f"⚠️ Context is too large! Truncating to {MAX_CTX_CHARS} chars.")
             current_len = 0
             truncated_docs = []
             for d in docs:
                 if current_len + len(d.page_content) > MAX_CTX_CHARS:
                     # Add remaining budget from this doc
                     remaining = MAX_CTX_CHARS - current_len
                     d.page_content = d.page_content[:remaining]
                     truncated_docs.append(d)
                     break
                 truncated_docs.append(d)
                 current_len += len(d.page_content)
             docs = truncated_docs

        # 3. Generate Answer Manually (Bypassing Chains)
        # Manually join context
        context_text = "\n\n".join([d.page_content for d in docs])
        
        # Prepare final prompt
        final_prompt = QA_PROMPT.format(context=context_text, input=question)
        
        print(f"📝 Final Prompt Length: {len(final_prompt)} characters")
        if len(final_prompt) > 20000:
             print("⚠️ Final prompt is unexpectedly huge! Truncating...")
             final_prompt = final_prompt[:20000]
        
        print(f"🚀 Sending request to LLM ({config.LLM_MODEL}) via Direct API...")
        
        headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "NotchNet Local",
        }
        
        payload = {
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "user", "content": final_prompt}
            ],
            # Optional parameters to ensure safety
            "temperature": 0.7,
            "max_tokens": 2000, 
        }
        
        url = f"{config.OPENROUTER_BASE_URL}/chat/completions"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            
            if resp.status_code != 200:
                error_msg = f"❌ Provider Error ({resp.status_code}): {resp.text}"
                print(error_msg)
                return f"Sorry, I encountered an error from the AI provider: {error_msg}\n"
            
            data = resp.json()
            answer = data['choices'][0]['message']['content'].strip()
            
        except Exception as api_err:
            print(f"❌ API Request Failed: {api_err}")
            return f"Sorry, the connection to the AI provider failed: {api_err}\n"
        
        if not answer:
             return "❌ Sorry, I couldn't find a good answer to your question."

        
        if not answer:
             return "❌ Sorry, I couldn't find a good answer to your question."
             
        formatted_sources = []
        for doc in docs:
            source_name = doc.metadata.get("source", "Unknown")
            filename = os.path.basename(source_name)
            formatted_sources.append(f"- {filename}")

        print(f"\n💬 Answer: {answer}\n")
        if formatted_sources:
            print("📚 Sources:")
            for src in formatted_sources:
                print(src)

        return f"{answer}\n"

    except Exception as e:
        print(f"⚠️ Error while generating answer: {e}")
        import traceback
        traceback.print_exc()
        raise e


def generate_answer_stream(question: str):
    """
    Generator that yields chunks of the answer.
    """
    global qa_chain
    if qa_chain is None:
        print("🔧 Building QA chain for the first time...")
        qa_chain = build_qa_chain()

    try:
        if len(question) > 2000:
            print(f"⚠️ Truncating massive input ({len(question)} chars) to 2000 chars.")
            question = question[:2000]

        # 1. Manual Retrieval
        retriever = build_retriever()
        docs = retriever.invoke(question)
        
        print(f"🔎 Manual Retrieval: Found {len(docs)} documents.")
        
        # 2. Context Truncation
        total_context_len = sum(len(d.page_content) for d in docs)
        MAX_CTX_CHARS = 12000
        if total_context_len > MAX_CTX_CHARS:
             print(f"⚠️ Context is too large! Truncating to {MAX_CTX_CHARS} chars.")
             current_len = 0
             truncated_docs = []
             for d in docs:
                 if current_len + len(d.page_content) > MAX_CTX_CHARS:
                     remaining = MAX_CTX_CHARS - current_len
                     d.page_content = d.page_content[:remaining]
                     truncated_docs.append(d)
                     break
                 truncated_docs.append(d)
                 current_len += len(d.page_content)
             docs = truncated_docs

        # 3. Stream Generation
        context_text = "\n\n".join([d.page_content for d in docs])
        final_prompt = QA_PROMPT.format(context=context_text, input=question)
        
        if len(final_prompt) > 20000:
             print("⚠️ Final prompt is unexpectedly huge! Truncating...")
             final_prompt = final_prompt[:20000]
        
        print(f"🚀 Sending request to LLM ({config.LLM_MODEL}) via Direct API (STREAMING)...")
        
        headers = {
            "Authorization": f"Bearer {config.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "NotchNet Local",
        }
        
        payload = {
            "model": config.LLM_MODEL,
            "messages": [
                {"role": "user", "content": final_prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2000, 
            "stream": True # Enable streaming
        }
        
        url = f"{config.OPENROUTER_BASE_URL}/chat/completions"
        try:
            with requests.post(url, headers=headers, json=payload, stream=True, timeout=60) as resp:
                if resp.status_code != 200:
                     error_msg = f"❌ Provider Error ({resp.status_code}): {resp.text}"
                     print(error_msg)
                     yield f"Error: {error_msg}"
                     return

                import json
                for line in resp.iter_lines():
                    if line:
                        line_str = line.decode('utf-8')
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
            
        except Exception as api_err:
            print(f"❌ API Request Failed: {api_err}")
            yield f"Error: {str(api_err)}"
        
        # Determine if we should send sources
        # For simplicity, let's append sources at the end if possible, 
        # or maybe the client handles basic text appending.
        # Ideally, we send structured events, but for now we are just yielding text chunks.
        
        formatted_sources = []
        for doc in docs:
            source_name = doc.metadata.get("source", "Unknown")
            filename = os.path.basename(source_name)
            formatted_sources.append(f"- {filename}")

        if formatted_sources:
            yield "\n\nSources:\n"
            for src in formatted_sources:
                yield f"{src}\n"

    except Exception as e:
        print(f"⚠️ Error while generating stream: {e}")
        import traceback
        traceback.print_exc()
        yield f"Error: {str(e)}"
