
import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from config import config

INDEX_PATH = config.INDEX_PATH

def debug_retrieval():
    print(f"📂 Loading index from {INDEX_PATH}...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        db = FAISS.load_local(
            INDEX_PATH, embedding_model, allow_dangerous_deserialization=True
        )
        print("✅ Index loaded.")
    except Exception as e:
        print(f"❌ Failed to load index: {e}")
        return

    query = "How do I install Sodium?"
    print(f"\n🔍 Query: '{query}'")
    
    print("👉 Testing search with k=5...")
    # Directly test similarity search which is what retriever uses
    docs = db.similarity_search(query, k=5)
    
    print(f"✅ Retrieved {len(docs)} documents.")
    
    total_chars = 0
    for i, doc in enumerate(docs):
        content_len = len(doc.page_content)
        total_chars += content_len
        print(f"  📄 Doc {i+1}: Length = {content_len} chars")
        print(f"     Preview: {doc.page_content[:100]}...")
    
    print(f"\n📊 Total Context Size: {total_chars} chars")
    print(f"🔢 Estimated Tokens (chars / 4): {total_chars / 4}")
    
    if total_chars > 100000:
        print("⚠️ FAILURE: Context is massive!")
    else:
        print("✅ SUCCESS: Context is reasonable.")

if __name__ == "__main__":
    debug_retrieval()
