"""
RxPY Operator for simple cosine similarity search.
This operator uses FAISS index and filters results to only include piece_vector entries.
"""
from typing import Callable, List, Tuple, Dict, Any, Optional
import numpy as np
from dataclasses import dataclass
import polars as pl
import asyncio
from reactivex import Observable, defer, from_
from reactivex import operators as ops
from sklearn.metrics.pairwise import cosine_similarity  # pyright: ignore[reportMissingModuleSource]

from src.domain.faiss.repositories.faiss_index_repository import FaissIndexRepository
from src.domain.piece_vector.repositories.piece_vector_repository import PieceVectorRepository
from src.application.vector.use_cases.search_vector_impl.exceptions.cosine_similarity_search_exception import CosineSimilaritySearchException
from src.infraestructure import logger


@dataclass
class SearchResult:
    catalog_unique_id: str
    similarity: float
    method: str = "cosine_similarity"
    match_count: int = 1
    best_angle_similarity: float = 0.0
    mean_angle_similarity: float = 0.0

    def to_dict(self):
        return {
            "catalog_unique_id": self.catalog_unique_id,
            "similarity": float(self.similarity),
            "method": self.method
        }


def simple_cosine_search_operator(
    faiss_repository: FaissIndexRepository,
    piece_vector_repository: PieceVectorRepository,
    k: int = 100,
    threshold: float = 0.0,
    max_results: Optional[int] = None,
) -> Callable[[Observable], Observable]:
    """
    RxPY operator that performs simple cosine similarity search using FAISS.
    
    This operator:
    1. Takes vectors from encoded image (object_vectors if available, otherwise piece_vector)
    2. Gets all vectors from FAISS index
    3. Calculates cosine similarity between query vectors and all FAISS vectors
    4. For each FAISS vector, takes the best similarity across all query vectors
    5. Filters results to only include entries from piece_vector table (not piece_object)
    6. Groups by catalog_unique_id and aggregates similarities
    7. Filters by threshold and sorts by similarity
    8. Returns results with catalog_unique_id, similarity, and method
    
    Args:
        faiss_repository: Repository to access FAISS index
        piece_vector_repository: Repository to access piece_vector table
        k: Number of candidates to retrieve from FAISS (default: 100)
        threshold: Minimum similarity threshold (default: 0.0)
        max_results: Maximum number of results to return (default: None = all)
        
    Returns:
        RxPY operator function that transforms encoded vectors to search results
    """
    def operator(source: Observable) -> Observable:
        return source.pipe(
            ops.flat_map(lambda encoded_data: defer(
                lambda _: _perform_simple_cosine_search_async(
                    encoded_data,
                    faiss_repository,
                    piece_vector_repository,
                    k,
                    threshold,
                    max_results
                )
            ))
        )
    
    return operator


def _perform_simple_cosine_search_async(
    encoded_data: Dict[str, Any],
    faiss_repository: FaissIndexRepository,
    piece_vector_repository: PieceVectorRepository,
    k: int,
    threshold: float,
    max_results: Optional[int]
) -> Observable[List[Tuple[str, float, Optional[Dict[str, Any]]]]]:
    """
    Performs simple cosine similarity search using FAISS (async wrapper).
    
    Args:
        encoded_data: Dict containing 'piece_vector' and 'object_vectors'
        faiss_repository: Repository to access FAISS index
        piece_vector_repository: Repository to access piece_vector table
        k: Number of candidates to retrieve from FAISS
        threshold: Minimum similarity threshold
        max_results: Maximum number of results
        
    Returns:
        Observable that emits a list of tuples in the form
        (catalog_unique_id, similarity_score, euclidean_info_or_None)
        This matches the output shape of `search_vectors_operator` expected by downstream operators.
    """
    async def search_task():
        return await _perform_simple_cosine_search(
            encoded_data,
            faiss_repository,
            piece_vector_repository,
            k,
            threshold,
            max_results
        )
    
    try:
        # Run async task
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If event loop is running, use thread executor
            import concurrent.futures
            
            def run_in_new_loop():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    return new_loop.run_until_complete(search_task())
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_new_loop)
                result = future.result()
                return from_([result])
        else:
            result = asyncio.run(search_task())
            return from_([result])
    except Exception as e:
        from reactivex import throw
        logger.error(f"❌ [SIMPLE_COSINE] Error in async wrapper: {type(e).__name__}: {e}")
        return throw(e)


def _extract_query_vectors(encoded_data: Dict[str, Any], threshold: float) -> Tuple[List[np.ndarray], float]:
    """Extract query vectors and compute effective threshold."""
    object_vectors = encoded_data.get('object_vectors', [])
    piece_vector = encoded_data.get('piece_vector', [])

    if object_vectors and len(object_vectors) > 0:
        query_vectors = [np.array(vec, dtype=np.float32) for vec in object_vectors]
        logger.info(f"🎯 [SIMPLE_COSINE] Using {len(query_vectors)} object vectors for multi-piece search")
        effective_threshold = threshold
    elif piece_vector and len(piece_vector) > 0:
        query_vectors = [np.array(piece_vector, dtype=np.float32)]
        logger.info("🎯 [SIMPLE_COSINE] Using single piece vector")
        logger.warning("⚠️ [SIMPLE_COSINE] No object vectors detected. For images with multiple pieces, consider enabling object detection (enableObjectDetector=true) for better similarity scores.")
        effective_threshold = max(0.0, threshold - 0.2)
        logger.info(f"🎯 [SIMPLE_COSINE] Using adjusted threshold: {effective_threshold} (original: {threshold})")
    else:
        logger.error("❌ [SIMPLE_COSINE] No vectors found in encoded data")
        raise CosineSimilaritySearchException(
            message="No vectors available for search",
            detail={"reason": "Both piece_vector and object_vectors are empty or missing"}
        )

    logger.info(f"📊 [SIMPLE_COSINE] Query vectors shapes: {[v.shape for v in query_vectors]}")
    return query_vectors, effective_threshold


def _calculate_faiss_similarities(query_vectors: List[np.ndarray], all_vectors: List[Tuple[str, List[float]]], effective_threshold: float) -> Dict[str, float]:
    """Calculate similarities and filter by threshold."""
    db_faiss_ids = [str(faiss_id) for faiss_id, _ in all_vectors]
    db_vectors = np.vstack([np.array(vec, dtype=np.float32) for _, vec in all_vectors])

    region_vectors = np.vstack(query_vectors)
    similarities_matrix = cosine_similarity(region_vectors, db_vectors)
    logger.info(f"   ✅ Similarities matrix calculated: {similarities_matrix.shape}")

    masked = np.where(similarities_matrix >= effective_threshold, similarities_matrix, -np.inf)
    best_per_db = masked.max(axis=0)
    valid_mask = best_per_db > -np.inf
    faiss_similarities = {db_faiss_ids[i]: float(best_per_db[i]) for i in range(len(db_faiss_ids)) if valid_mask[i]}

    logger.info(f"📊 [SIMPLE_COSINE] Found {len(faiss_similarities)} unique vectors above threshold")
    if faiss_similarities:
        vals = list(faiss_similarities.values())
        logger.info(f"📈 [SIMPLE_COSINE] Similarity range: [{min(vals):.4f}, {max(vals):.4f}]")
    
    return faiss_similarities


def _build_result_dataframe(faiss_similarities: Dict[str, float], piece_vector_entries: List) -> pl.DataFrame:
    """Build Polars DataFrame from filtered piece vector entries."""
    rows = [
        {
            "faiss_id": str(faiss_id),
            "piece_id": piece_id,
            "catalog_unique_id": catalog_unique_id,
            "similarity": faiss_similarities.get(str(faiss_id))
        }
        for (faiss_id, piece_id, catalog_unique_id) in piece_vector_entries
        if str(faiss_id) in faiss_similarities and catalog_unique_id
    ]

    if not rows:
        logger.warning("⚠️ [SIMPLE_COSINE] After filtering, no valid piece entries remain")
        raise CosineSimilaritySearchException(
            message="No piece_vector entries matched FAISS candidates",
            detail={"candidates": len(faiss_similarities)}
        )

    return pl.DataFrame(rows)


def _apply_match_boosting(agg: pl.DataFrame, match_boost_rate: float, match_boost_cap: float) -> Tuple[List[str], np.ndarray]:
    """Apply vectorized match boosting to aggregated similarities."""
    catalog_ids = agg['catalog_unique_id'].to_list()
    bests = np.array(agg['best_similarity'].to_list(), dtype=np.float64)
    counts = np.array(agg['match_count'].to_list(), dtype=np.int32)

    boost_effect = np.zeros_like(bests)
    valid_boost_mask = (counts > 1) & (match_boost_cap > 0) & (match_boost_rate > 0)
    if np.any(valid_boost_mask):
        boost_effect[valid_boost_mask] = 1.0 - np.exp(-match_boost_rate * (counts[valid_boost_mask] - 1))

    aggregated = np.where(
        valid_boost_mask,
        bests + (1.0 - bests) * match_boost_cap * boost_effect,
        bests
    )

    return catalog_ids, aggregated


def _subscribe_sync(observable: Observable):
    """Helper to subscribe to Observable and get value synchronously."""
    result = []
    error_holder = []
    
    def on_next(value):
        result.append(value)
    
    def on_error(e):
        error_holder.append(e)
    
    observable.subscribe(on_next=on_next, on_error=on_error)
    
    if error_holder:
        raise error_holder[0]
    
    if not result:
        return []  # Return empty list instead of None for list operations
    
    return result[0]


async def _perform_simple_cosine_search(
    encoded_data: Dict[str, Any],
    faiss_repository: FaissIndexRepository,
    piece_vector_repository: PieceVectorRepository,
    k: int,
    threshold: float,
    max_results: Optional[int]
) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
    """
    Performs simple cosine similarity search using FAISS and filters by piece_vector table.
    
    Uses the optimized batch calculation logic from search_vectors_operator:
    1. Calculate all similarities in batch using cosine_similarity
    2. Filter by threshold
    3. Group by faiss_id and take best similarity per vector
    4. Filter to only piece_vector entries
    5. Group by catalog_unique_id and aggregate similarities
    
    Args:
        encoded_data: Dict containing 'piece_vector' and 'object_vectors'
        faiss_repository: Repository to access FAISS index
        piece_vector_repository: Repository to access piece_vector table
        k: Number of candidates to retrieve from FAISS
        threshold: Minimum similarity threshold
        max_results: Maximum number of results
        
    Returns:
        Dict with search results including method='cosine_similarity'
    """
    try:
        logger.info("🔍 [SIMPLE_COSINE] Starting simple cosine similarity search with FAISS...")

        # Step 1: Extract query vectors
        query_vectors, effective_threshold = _extract_query_vectors(encoded_data, threshold)

        # Step 2: Get all vectors from FAISS index (now returns Observable)
        logger.info("🔍 [SIMPLE_COSINE] Getting all vectors from FAISS index...")
        all_vectors_obs = faiss_repository.get_all_vectors()
        all_vectors = _subscribe_sync(all_vectors_obs)

        if not all_vectors:
            logger.warning("⚠️ [SIMPLE_COSINE] No vectors found in FAISS index")
            raise CosineSimilaritySearchException(
                message="No vectors found in FAISS index",
                detail={"reason": "FAISS get_all_vectors returned empty results"}
            )

        # Step 3: Calculate similarities
        faiss_similarities = _calculate_faiss_similarities(query_vectors, all_vectors, effective_threshold)

        # Step 4: Filter to only piece_vector entries
        faiss_ids = list(faiss_similarities.keys())
        logger.info(f"🗃️ [SIMPLE_COSINE] Filtering to piece_vector entries from {len(faiss_ids)} candidates...")
        piece_vector_entries = await piece_vector_repository.get_by_ids(faiss_ids)

        if not piece_vector_entries:
            logger.warning("⚠️ [SIMPLE_COSINE] No piece_vector entries found for FAISS IDs")
            raise CosineSimilaritySearchException(
                message="No piece_vector entries found",
                detail={"reason": "All FAISS results are from piece_object table"}
            )

        logger.info(f"✅ [SIMPLE_COSINE] Found {len(piece_vector_entries)} piece_vector entries")

        # Step 5: Build DataFrame and aggregate
        df = _build_result_dataframe(faiss_similarities, piece_vector_entries)
        agg = df.group_by(['catalog_unique_id']).agg([
            pl.col('similarity').max().alias('best_similarity'),
            pl.col('similarity').mean().alias('mean_similarity'),
            pl.count('similarity').alias('match_count')
        ])

        # Step 6: Apply boosting
        search_params = encoded_data.get('search_params', {}) or {}
        match_boost_rate = float(search_params.get('match_boost_rate', 0.5))
        match_boost_cap = float(search_params.get('match_boost_cap', 0.25))

        catalog_ids, aggregated = _apply_match_boosting(agg, match_boost_rate, match_boost_cap)
        bests = np.array(agg['best_similarity'].to_list(), dtype=np.float64)
        means = np.array(agg['mean_similarity'].to_list(), dtype=np.float64)
        counts = np.array(agg['match_count'].to_list(), dtype=np.int32)

        # Build SearchResult objects
        results = [
            SearchResult(
                catalog_unique_id=cid,
                similarity=float(sim),
                match_count=int(mc),
                best_angle_similarity=float(b),
                mean_angle_similarity=float(m)
            )
            for cid, sim, mc, b, m in zip(catalog_ids, aggregated.tolist(), counts.tolist(), bests.tolist(), means.tolist())
        ]

        # Sort and apply limits
        results.sort(key=lambda r: r.similarity, reverse=True)

        if k > 0 and len(results) > k:
            results = results[:k]
            logger.info(f"✂️ [SIMPLE_COSINE] Limited to top {k} results")

        if max_results is not None and max_results > 0 and len(results) > max_results:
            results = results[:max_results]
            logger.info(f"✂️ [SIMPLE_COSINE] Limited to max {max_results} results")

        if not results:
            logger.warning(f"⚠️ [SIMPLE_COSINE] No results above effective threshold {effective_threshold}")
            raise CosineSimilaritySearchException(
                message="No similar vectors found above threshold",
                detail={"threshold": threshold, "effective_threshold": effective_threshold, "faiss_candidates": len(faiss_ids)}
            )

        logger.info(f"✅ [SIMPLE_COSINE] Returning {len(results)} unique pieces")

        # Convert to tuples compatible with `search_vectors_operator` and `format_results_operator`
        tuple_results: List[Tuple[str, float, Optional[Dict[str, Any]]]] = [
            (r.catalog_unique_id, float(r.similarity), None) for r in results
        ]

        return tuple_results

    except CosineSimilaritySearchException:
        raise
    except Exception as e:
        logger.error(f"❌ [SIMPLE_COSINE] Unexpected error: {type(e).__name__}: {e}")
        raise CosineSimilaritySearchException(
            message="Error performing simple cosine similarity search",
            detail={"error": str(e)}
        )
