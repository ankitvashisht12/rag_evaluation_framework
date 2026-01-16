from typing import List, Dict, Any
import logging

from rag_evaluation_framework.evaluation.metrics.token_level_base import TokenLevelSpanMetric

logger = logging.getLogger(__name__)


class TokenLevelPrecision(TokenLevelSpanMetric):
    """
    Token-level precision metric for RAG evaluation (span overlap).
    """

    def __init__(self):
        pass

    def calculate(
        self,
        retrieved_chunk_ids: List[Dict[str, Any]],
        ground_truth_chunk_ids: List[Dict[str, Any]],
    ) -> float:
        logger.debug(
            "TokenLevelPrecision.calculate start: retrieved=%d ground_truth=%d",
            len(retrieved_chunk_ids),
            len(ground_truth_chunk_ids),
        )
        if not retrieved_chunk_ids:
            logger.debug("No retrieved chunks provided; returning 0.0")
            return 0.0

        if not ground_truth_chunk_ids:
            logger.debug("No ground truth chunks provided; returning 0.0")
            return 0.0

        ground_truth_ranges = self._ranges_by_doc(ground_truth_chunk_ids)
        retrieved_ranges = self._ranges_by_doc(retrieved_chunk_ids)

        _, total_retrieved_len, total_overlap_len = self._compute_lengths(
            ground_truth_ranges, retrieved_ranges
        )

        if total_retrieved_len == 0:
            logger.debug("Total retrieved length is 0; returning 0.0")
            return 0.0

        score = total_overlap_len / total_retrieved_len
        logger.debug(
            "TokenLevelPrecision.calculate end: total_retrieved_len=%d total_overlap_len=%d score=%.6f",
            total_retrieved_len,
            total_overlap_len,
            score,
        )
        return score
