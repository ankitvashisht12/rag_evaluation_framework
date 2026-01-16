from typing import List, Dict, Any, Tuple
import logging

from rag_evaluation_framework.evaluation.metrics.token_level_base import TokenLevelSpanMetric

logger = logging.getLogger(__name__)


class TokenLevelMRR(TokenLevelSpanMetric):
    """
    Token-level Mean Reciprocal Rank using span overlap.
    """

    def __init__(self, min_overlap_ratio: float = 0.0):
        """
        Args:
            min_overlap_ratio: Minimum overlap ratio (relative to reference span length)
                for a retrieved chunk to be considered relevant.
        """
        self.min_overlap_ratio = min_overlap_ratio

    def calculate(
        self,
        retrieved_chunk_ids: List[Dict[str, Any]],
        ground_truth_chunk_ids: List[Dict[str, Any]],
    ) -> float:
        logger.debug(
            "TokenLevelMRR.calculate start: retrieved=%d ground_truth=%d min_overlap_ratio=%.3f",
            len(retrieved_chunk_ids),
            len(ground_truth_chunk_ids),
            self.min_overlap_ratio,
        )
        if not retrieved_chunk_ids or not ground_truth_chunk_ids:
            logger.debug("Missing retrieved or ground truth chunks; returning 0.0")
            return 0.0

        ground_truth_ranges = self._ranges_by_doc(ground_truth_chunk_ids)
        if not ground_truth_ranges:
            logger.debug("No valid ground truth ranges; returning 0.0")
            return 0.0

        ground_truth_union_by_doc = {
            doc_id: self._union_ranges(ranges)
            for doc_id, ranges in ground_truth_ranges.items()
        }
        ground_truth_len_by_doc = {
            doc_id: self._sum_ranges(ranges)
            for doc_id, ranges in ground_truth_union_by_doc.items()
        }

        for idx, chunk in enumerate(retrieved_chunk_ids, start=1):
            doc_id = chunk.get("doc_id")
            start = chunk.get("start_index")
            end = chunk.get("end_index")
            if doc_id is None or start is None or end is None:
                continue
            doc_id = str(doc_id)
            if doc_id not in ground_truth_union_by_doc:
                continue

            chunk_range = (int(start), int(end))
            overlap_len = 0
            for ref_range in ground_truth_union_by_doc[doc_id]:
                intersection = self._intersect_ranges(chunk_range, ref_range)
                if intersection is not None:
                    overlap_len += intersection[1] - intersection[0]

            ref_len = ground_truth_len_by_doc.get(doc_id, 0)
            if ref_len == 0:
                continue

            overlap_ratio = overlap_len / ref_len
            if overlap_ratio >= self.min_overlap_ratio and overlap_len > 0:
                score = 1.0 / idx
                logger.debug(
                    "TokenLevelMRR hit at rank %d (overlap_ratio=%.3f); score=%.3f",
                    idx,
                    overlap_ratio,
                    score,
                )
                return score

        logger.debug("TokenLevelMRR no relevant chunk found; returning 0.0")
        return 0.0
