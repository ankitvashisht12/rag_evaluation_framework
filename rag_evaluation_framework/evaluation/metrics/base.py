from abc import ABC, abstractmethod
from typing import List, Dict

class Metrics(ABC):
    @abstractmethod
    def calculate(self, retrieved_chunk_ids: List[str], ground_truth_chunk_ids: str) -> float:
        raise NotImplementedError