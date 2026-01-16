from typing import Any, Dict, Iterable, List, Optional, Tuple
import logging

from langsmith.schemas import Example, Run

from rag_evaluation_framework.evaluation.metrics.base import Metrics

logger = logging.getLogger(__name__)


class TokenLevelSpanMetric(Metrics):
    """
    Shared span-based helpers for token-level metrics.
    """

    def _preview(self, value: Any, limit: int = 2000) -> str:
        """
        Return a bounded string preview for logging.
        """
        try:
            text = repr(value)
        except Exception as exc:  # pragma: no cover - defensive logging
            return f"<unreprable:{type(value).__name__} error={exc}>"
        if len(text) > limit:
            return text[:limit] + "...<truncated>"
        return text

    def _intersect_ranges(
        self, range1: Tuple[int, int], range2: Tuple[int, int]
    ) -> Optional[Tuple[int, int]]:
        start1, end1 = range1
        start2, end2 = range2
        intersect_start = max(start1, start2)
        intersect_end = min(end1, end2)
        if intersect_start < intersect_end:
            return (intersect_start, intersect_end)
        return None

    def _union_ranges(self, ranges: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
        sorted_ranges = sorted(ranges, key=lambda x: x[0])
        if not sorted_ranges:
            return []
        merged = [sorted_ranges[0]]
        for current_start, current_end in sorted_ranges[1:]:
            last_start, last_end = merged[-1]
            if current_start <= last_end:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        return merged

    def _sum_ranges(self, ranges: Iterable[Tuple[int, int]]) -> int:
        return sum(end - start for start, end in ranges)

    def _ranges_by_doc(self, items: List[Dict[str, Any]]) -> Dict[str, List[Tuple[int, int]]]:
        ranges_by_doc: Dict[str, List[Tuple[int, int]]] = {}
        for item in items:
            doc_id = item.get("doc_id")
            start = item.get("start_index")
            end = item.get("end_index")
            if doc_id is None or start is None or end is None:
                logger.debug(
                    "Skipping range with missing fields: doc_id=%s start_index=%s end_index=%s",
                    doc_id,
                    start,
                    end,
                )
                continue
            ranges_by_doc.setdefault(str(doc_id), []).append((int(start), int(end)))
        logger.debug(
            "Built ranges_by_doc: docs=%d total_ranges=%d",
            len(ranges_by_doc),
            sum(len(ranges) for ranges in ranges_by_doc.values()),
        )
        return ranges_by_doc

    def _compute_lengths(
        self,
        ground_truth_ranges: Dict[str, List[Tuple[int, int]]],
        retrieved_ranges: Dict[str, List[Tuple[int, int]]],
    ) -> Tuple[int, int, int]:
        total_reference_len = 0
        total_retrieved_len = 0
        total_overlap_len = 0

        for doc_id, ref_ranges in ground_truth_ranges.items():
            ref_union = self._union_ranges(ref_ranges)
            total_reference_len += self._sum_ranges(ref_union)

            overlap_ranges: List[Tuple[int, int]] = []
            for ref_range in ref_ranges:
                for chunk_range in retrieved_ranges.get(doc_id, []):
                    intersection = self._intersect_ranges(ref_range, chunk_range)
                    if intersection is not None:
                        overlap_ranges.append(intersection)

            overlap_union = self._union_ranges(overlap_ranges)
            total_overlap_len += self._sum_ranges(overlap_union)

        for doc_id, chunk_ranges in retrieved_ranges.items():
            chunk_union = self._union_ranges(chunk_ranges)
            total_retrieved_len += self._sum_ranges(chunk_union)

        return total_reference_len, total_retrieved_len, total_overlap_len

    def extract_ground_truth_chunks_ids(self, example: Optional[Example]) -> List[Dict[str, Any]]:
        if example is None:
            logger.debug("No example provided for ground truth extraction")
            return []

        logger.debug(
            "Example inputs preview: %s",
            self._preview(getattr(example, "inputs", None)),
        )
        logger.debug(
            "Example outputs preview: %s",
            self._preview(getattr(example, "outputs", None)),
        )

        if hasattr(example, "outputs") and example.outputs:
            if isinstance(example.outputs, dict):
                if (
                    "doc_id" in example.outputs
                    and "start_index" in example.outputs
                    and "end_index" in example.outputs
                ):
                    logger.debug("Ground truth found in example.outputs span dict")
                    return [example.outputs]
                if "references" in example.outputs:
                    logger.debug("Ground truth found in example.outputs['references']")
                    return example.outputs["references"]
                if "ground_truth" in example.outputs:
                    logger.debug("Ground truth found in example.outputs['ground_truth']")
                    return example.outputs["ground_truth"]
                if "spans" in example.outputs:
                    logger.debug("Ground truth found in example.outputs['spans']")
                    return example.outputs["spans"]
            elif isinstance(example.outputs, list):
                logger.debug("Ground truth found in example.outputs list")
                return example.outputs
        logger.debug("No ground truth found in example.outputs")
        return []

    def extract_retrieved_chunks_ids(self, run: Run) -> List[Dict[str, Any]]:
        logger.debug(
            "Run outputs preview: %s",
            self._preview(getattr(run, "outputs", None)),
        )
        if hasattr(run, "outputs"):
            if isinstance(run.outputs, list):
                return self._normalize_retrieved_chunks(run.outputs)
            if isinstance(run.outputs, dict):
                if "chunks" in run.outputs:
                    return self._normalize_retrieved_chunks(run.outputs["chunks"])
                if "retrieved_chunks" in run.outputs:
                    return self._normalize_retrieved_chunks(run.outputs["retrieved_chunks"])
                if "output" in run.outputs:
                    return self._normalize_retrieved_chunks(run.outputs["output"])
                return self._normalize_retrieved_chunks(run.outputs.get("chunk_ids", []))
        return []

    def _normalize_retrieved_chunks(self, chunks: Any) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        if not isinstance(chunks, list):
            return normalized
        for chunk in chunks:
            if isinstance(chunk, dict):
                if "metadata" in chunk and isinstance(chunk["metadata"], dict):
                    metadata = chunk["metadata"]
                    normalized.append(
                        {
                            "doc_id": metadata.get("doc_id"),
                            "start_index": metadata.get("start_index"),
                            "end_index": metadata.get("end_index"),
                        }
                    )
                else:
                    normalized.append(
                        {
                            "doc_id": chunk.get("doc_id"),
                            "start_index": chunk.get("start_index"),
                            "end_index": chunk.get("end_index"),
                        }
                    )
        return normalized
