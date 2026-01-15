from typing import List, Optional, Dict, Any, Iterable, Tuple

from langsmith.schemas import Example, Run

from rag_evaluation_framework.evaluation.metrics.base import Metrics


class TokenLevelRecall(Metrics):
    """
    Token-level recall metric for RAG evaluation.
    
    Measures the proportion of tokens in the ground truth that appear
    in the retrieved chunks. This is useful when you want to measure
    how much of the relevant content was retrieved, regardless of
    chunk boundaries.
    """

    def __init__(self):
        """
        Initialize TokenLevelRecall metric.
        """
        pass

    def calculate(
        self,
        retrieved_chunk_ids: List[Dict[str, Any]],
        ground_truth_chunk_ids: List[Dict[str, Any]],
    ) -> float:
        """
        Calculate token-level recall using span overlap.

        Ground truth and retrieved chunks are expected to include:
        - doc_id
        - start_index
        - end_index
        """
        if not ground_truth_chunk_ids:
            return 0.0

        if not retrieved_chunk_ids:
            return 0.0

        ground_truth_ranges = self._ranges_by_doc(ground_truth_chunk_ids)
        retrieved_ranges = self._ranges_by_doc(retrieved_chunk_ids)

        total_reference_len = 0
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

        if total_reference_len == 0:
            return 0.0

        return total_overlap_len / total_reference_len

    def _intersect_ranges(self, range1: Tuple[int, int], range2: Tuple[int, int]) -> Optional[Tuple[int, int]]:
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
                continue
            ranges_by_doc.setdefault(str(doc_id), []).append((int(start), int(end)))
        return ranges_by_doc

    def extract_ground_truth_chunks_ids(self, example: Optional[Example]) -> List[Dict[str, Any]]:
        """
        Extract ground truth content from Langsmith Example.
        
        Args:
            example: Langsmith Example containing ground truth data
            
        Returns:
            List of ground truth texts/chunk IDs
        """
        if example is None:
            return []
        
        # Expect span-based ground truth in outputs
        if hasattr(example, 'outputs') and example.outputs:
            if isinstance(example.outputs, dict):
                if "references" in example.outputs:
                    return example.outputs["references"]
                if "ground_truth" in example.outputs:
                    return example.outputs["ground_truth"]
                if "spans" in example.outputs:
                    return example.outputs["spans"]
            elif isinstance(example.outputs, list):
                return example.outputs
        
        return []

    def extract_retrieved_chunks_ids(self, run: Run) -> List[Dict[str, Any]]:
        """
        Extract retrieved content from Langsmith Run.
        
        Args:
            run: Langsmith Run containing retrieval results
            
        Returns:
            List of retrieved texts/chunk IDs
        """
        if hasattr(run, 'outputs'):
            if isinstance(run.outputs, list):
                return self._normalize_retrieved_chunks(run.outputs)
            if isinstance(run.outputs, dict):
                if "chunks" in run.outputs:
                    return self._normalize_retrieved_chunks(run.outputs["chunks"])
                if "retrieved_chunks" in run.outputs:
                    return self._normalize_retrieved_chunks(run.outputs["retrieved_chunks"])
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
