import logging
from typing import List, Optional, Mapping, Union

import chromadb
from chromadb.api.types import SparseVector

from rag_evaluation_framework.evaluation.vector_store.base import VectorStore

MetadataValue = Union[str, int, float, SparseVector, None]
Metadata = Mapping[str, MetadataValue]

logger = logging.getLogger(__name__)


class ChromaVectorStore(VectorStore):
    """
    Vector store implementation using ChromaDB.
    
    ChromaDB is an open-source embedding database for AI applications.
    See: https://github.com/chroma-core/chroma
    
    By default, uses in-memory storage for easy prototyping.
    For persistence, pass a persist_directory to the constructor.
    """

    def __init__(
        self, 
        collection_name: str = "default",
        persist_directory: Optional[str] = None
    ):
        """
        Initialize the ChromaDB vector store.
        
        Args:
            collection_name: Name of the collection to use
            persist_directory: Optional path to persist data. If None, uses in-memory storage.
        """
        self.collection_name = collection_name
        self._id_counter = 0
        
        # Initialize ChromaDB client
        if persist_directory:
            logger.debug("Initializing ChromaDB with persistence at: %s", persist_directory)
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            logger.debug("Initializing ChromaDB in-memory client")
            self._client = chromadb.Client()
        
        # Get or create collection
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Use cosine similarity
        )
        logger.debug("Using collection '%s' with cosine similarity", collection_name)

    def add_docs(
        self, 
        docs: List[str], 
        embeddings: List[List[float]], 
        doc_ids: Optional[List[str]] = None,
        metadatas: Optional[List[Metadata]] = None,
    ) -> None:
        """
        Add documents and their embeddings to the vector store.
        
        Args:
            docs: List of document texts
            embeddings: List of embedding vectors (must match length of docs)
            doc_ids: Optional list of document IDs. If not provided, auto-generated.
        """
        if len(docs) != len(embeddings):
            raise ValueError(f"Number of docs ({len(docs)}) must match number of embeddings ({len(embeddings)})")
        
        if not docs:
            return
        
        # Generate IDs if not provided
        if doc_ids is None:
            doc_ids = [f"doc_{self._id_counter + i}" for i in range(len(docs))]
            self._id_counter += len(docs)
        
        if metadatas is not None and len(metadatas) != len(docs):
            raise ValueError(f"Number of metadatas ({len(metadatas)}) must match number of docs ({len(docs)})")

        # Add to ChromaDB collection
        self._collection.add(
            documents=docs,
            embeddings=embeddings,  # type: ignore[arg-type]
            ids=doc_ids,
            metadatas=metadatas,
        )
        logger.debug("Added %d documents to collection '%s'", len(docs), self.collection_name)

    def search(self, query_embedding: List[float], k: int) -> List[dict]:
        """
        Search for similar documents using the query embedding.
        
        Args:
            query_embedding: The query vector to search with
            k: Number of results to return
            
        Returns:
            List of document texts, ordered by similarity (most similar first)
        """
        if self._collection.count() == 0:
            return []
        
        # Query ChromaDB
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count()),
            include=["documents", "metadatas"]
        )
        
        # Extract documents from results
        documents = results.get("documents", [[]]) or [[]]
        metadatas = results.get("metadatas", [[]]) or [[]]
        docs_list = documents[0] if documents else []
        metas_list = metadatas[0] if metadatas else []
        return [
            {"text": doc, "metadata": meta}
            for doc, meta in zip(docs_list, metas_list)
        ]

    def search_with_scores(
        self, 
        query_embedding: List[float], 
        k: int
    ) -> List[tuple[str, float]]:
        """
        Search for similar documents and return with similarity scores.
        
        Args:
            query_embedding: The query vector to search with
            k: Number of results to return
            
        Returns:
            List of (document_text, similarity_score) tuples.
            Note: ChromaDB returns distances, so we convert to similarity (1 - distance for cosine).
        """
        if self._collection.count() == 0:
            return []
        
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self._collection.count()),
            include=["documents", "distances"]
        )
        
        documents_list = results.get("documents") or [[]]
        distances_list = results.get("distances") or [[]]
        
        documents = documents_list[0] if documents_list else []
        distances = distances_list[0] if distances_list else []
        
        # Convert distances to similarity scores (for cosine: similarity = 1 - distance)
        return [(doc, 1.0 - dist) for doc, dist in zip(documents, distances)]

    def get_by_ids(self, doc_ids: List[str]) -> List[str]:
        """
        Retrieve documents by their IDs.
        
        Args:
            doc_ids: List of document IDs to retrieve
            
        Returns:
            List of document texts
        """
        results = self._collection.get(ids=doc_ids, include=["documents"])
        documents = results.get("documents")
        return documents if documents is not None else []

    def delete(self, doc_ids: List[str]) -> None:
        """
        Delete documents by their IDs.
        
        Args:
            doc_ids: List of document IDs to delete
        """
        self._collection.delete(ids=doc_ids)

    def clear(self) -> None:
        """Clear all documents from the collection."""
        logger.debug("Clearing collection '%s'", self.collection_name)
        # Delete and recreate the collection
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self._id_counter = 0

    def __len__(self) -> int:
        """Return the number of documents in the store."""
        return self._collection.count()
