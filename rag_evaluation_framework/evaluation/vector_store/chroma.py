
from rag_evaluation_framework.evaluation.vector_store.base import VectorStore
from typing import List

class ChromaVectorStore(VectorStore):

    def __init__(self):
        self.docs: List[str] = []
        self.embeddings: List[List[float]] = []

    def add_docs(self, docs: List[str], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the vector store."""
        self.docs.extend(docs)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: List[float], k: int) -> List[str]:
        """Search for similar documents using query embedding."""
        # TODO: Implement proper similarity search using ChromaDB
        # For now, return empty list as placeholder
        return []
