# agent/tool/rag.py
from langchain.tools import tool

# This will be set by VoiceAssistant
_vector_store = None

def set_vector_store(vs):
    """Set the vector store for the RAG tool"""
    global _vector_store
    _vector_store = vs

@tool(response_format="content_and_artifact")
def rag_context(query: str):
    """Retrieve information to help answer a query.
    
    Args:
        query: The search query to find relevant documents
    """
    if _vector_store is None:
        return "Vector store not initialized", []
    
    retrieved_docs = _vector_store.similarity_search(query, k=2)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs