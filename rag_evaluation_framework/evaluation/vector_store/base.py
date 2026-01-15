from typing import List, Optional, Dict, Any, Mapping

Metadata = Mapping[str, Any]
from abc import ABC, abstractmethod

class VectorStore(ABC):
    @abstractmethod
    def add_docs(
        self,
        docs: List[str],
        embeddings: List[List[float]],
        doc_ids: Optional[List[str]] = None,
        metadatas: Optional[List[Metadata]] = None,
    ) -> None:
        """Add documents and their embeddings to the vector store."""
        raise NotImplementedError

    @abstractmethod
    def search(self, query_embedding: List[float], k: int) -> List[Dict[str, Any]]:
        """Search for similar documents using query embedding."""
        raise NotImplementedError
