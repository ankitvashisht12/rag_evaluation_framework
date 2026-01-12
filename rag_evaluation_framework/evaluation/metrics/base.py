from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Callable
from langsmith import Evaluator, EvaluationResult
from langsmith.schemas import Example, Run

class Metrics(ABC):
    @abstractmethod
    def calculate(self, retrieved_chunk_ids: List[str], ground_truth_chunk_ids: List[str]) -> float:
        raise NotImplementedError

    @abstractmethod
    def extract_ground_truth_chunks_ids(self, example: Optional[Example]) -> List[str]:
        raise NotImplementedError

    @abstractmethod
    def extract_retrieved_chunks_ids(self, run: Run) -> List[str]:
        raise NotImplementedError

    def to_langsmith_evaluator(self, metric_name: Optional[str] = None, k: Optional[int] = None) -> Callable[[Run, Optional[Example]], EvaluationResult]:

        name = metric_name or self.__class__.__name__

        if k is not None:
            name = f"{name}@{k}"

        def evaluator(run: Run, example: Optional[Example]) -> EvaluationResult:
            retrieved_chunks_ids = self.extract_retrieved_chunks_ids(run)
            ground_truth_chunks_ids = self.extract_ground_truth_chunks_ids(example)



            score = self.calculate(retrieved_chunks_ids, ground_truth_chunks_ids)

            return EvaluationResult(
                key=name,
                score=score,
            )

        return evaluator

        