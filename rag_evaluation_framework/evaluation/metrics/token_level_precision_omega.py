from typing import List, Dict, Any
import logging

from rag_evaluation_framework.evaluation.metrics.token_level_base import TokenLevelSpanMetric

logger = logging.getLogger(__name__)


class TokenLevelPrecisionOmega(TokenLevelSpanMetric):
    """
    Token-level precision Omega metric (span overlap).
    """

    def __init__(self):
        pass

    def calculate(
        self,
        retrieved_chunk_ids: List[Dict[str, Any]],
        ground_truth_chunk_ids: List[Dict[str, Any]],
    ) -> float:
        logger.debug(
            "TokenLevelPrecisionOmega.calculate start: retrieved=%d ground_truth=%d",
            len(retrieved_chunk_ids),
            len(ground_truth_chunk_ids),
        )
        if not retrieved_chunk_ids or not ground_truth_chunk_ids:
            logger.debug("Missing retrieved or ground truth chunks; returning 0.0")
            return 0.0

        ground_truth_ranges = self._ranges_by_doc(ground_truth_chunk_ids)
        retrieved_ranges = self._ranges_by_doc(retrieved_chunk_ids)

        total_reference_len, total_retrieved_len, total_overlap_len = self._compute_lengths(
            ground_truth_ranges, retrieved_ranges
        )

        missed_reference_len = max(total_reference_len - total_overlap_len, 0)
        denominator = total_retrieved_len + missed_reference_len
        if denominator == 0:
            logger.debug("PrecisionOmega denominator is 0; returning 0.0")
            return 0.0

        score = total_overlap_len / denominator
        logger.debug(
            "TokenLevelPrecisionOmega.calculate end: retrieved_len=%d missed_ref_len=%d overlap_len=%d score=%.6f",
            total_retrieved_len,
            missed_reference_len,
            total_overlap_len,
            score,
        )
        return score
