from typing import List
from abc import ABC, abstractmethod

class VectorStore(ABC):
    @abstractmethod
    def add_docs(self, docs: List[str], embeddings: List[List[float]]) -> None:
        """Add documents and their embeddings to the vector store."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: List[float], k: int) -> List[str]:
        """Search for similar documents using query embedding."""
        raise NotImplementedError
