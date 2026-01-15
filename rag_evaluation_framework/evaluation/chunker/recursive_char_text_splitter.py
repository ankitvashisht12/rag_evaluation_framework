import logging
from rag_evaluation_framework.evaluation.chunker.base import Chunker, Chunk
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class RecursiveCharTextSplitter(Chunker):
    chunk_size: int
    chunk_overlap: int

    def __init__(self, chunk_size: int=100, chunk_overlap: int=10):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk(self, text: str) -> list[Chunk]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = text_splitter.split_text(text)
        results: list[Chunk] = []
        search_start = 0

        for chunk_text in chunks:
            start_index = text.find(chunk_text, search_start)
            if start_index == -1:
                start_index = text.find(chunk_text)
            if start_index == -1:
                logger.warning("Chunk text not found in source text; skipping chunk")
                continue
            end_index = start_index + len(chunk_text)
            results.append(
                Chunk(
                    text=chunk_text,
                    start_index=start_index,
                    end_index=end_index,
                )
            )
            search_start = end_index

        return results
