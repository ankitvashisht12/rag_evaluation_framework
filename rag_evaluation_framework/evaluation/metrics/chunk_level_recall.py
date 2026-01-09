from rag_evaluation_framework.evaluation.metrics.base import Metrics
from typing import List, Dict

class ChunkLevelRecall(Metrics):
    def calculate(self, retrieved_chunk_ids: List[str], ground_truth_chunk_ids: str) -> float:

        retrieved_chunk_ids_set = set(retrieved_chunk_ids)
        ground_truth_chunk_ids_set = set(ground_truth_chunk_ids)

        if(len(ground_truth_chunk_ids_set) == 0 or len(retrieved_chunk_ids_set) == 0):
            return 0.0

        return len(retrieved_chunk_ids_set & ground_truth_chunk_ids_set) / len(ground_truth_chunk_ids_set)