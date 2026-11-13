"""
RxPY Operator for searching similar vectors.
Optimized version with vectorized operations and efficient data structures.
"""
from typing import List, Tuple, Callable, Optional, Dict, Any
from dataclasses import dataclass
from reactivex import Observable
from reactivex import operators as ops

import numpy as np
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances # pyright: ignore[reportMissingModuleSource]

from src.common.exceptions.custom_server_exception import CustomServerException
from src.domain.faiss.repositories.faiss_index_repository import FaissIndexRepository
from src.domain.piece_vector.repositories.piece_vector_repository import PieceVectorRepository
from src.application.vector.use_cases.search_vector_impl.exceptions.cosine_similarity_search_exception import CosineSimilaritySearchException
from src.infraestructure import logger


@dataclass
class RegionMatch:
    """Dataclass for efficient region match storage."""
    region_meta: Dict[str, Any]
    faiss_id: str
    real_piece_id: str
    similarity: float


@dataclass
class PieceMatch:
    """Dataclass for efficient piece match storage."""
    piece_id: str
    real_piece_id: str
    similarity: float
    original_similarity: float
    match_count: int
    same_piece_count: int
    region: Dict[str, Any]
    has_relative_penalty: bool = False


@dataclass
class VectorDistanceResult:
    """Dataclass para almacenar distancias euclidianas entre vectores."""
    piece_id: str
    cosine_similarity: float
    euclidean_distance: float  # Distancia euclidiana entre vectores de embedding
    combined_score: float  # Score combinado (cosine + euclidean)


def _calculate_combined_score(
    cosine_sim: float,
    euclidean_sim: float,
    consistency_ratio: float,
    weights_high: Tuple[float, float] = (0.7, 0.3),
    weights_low: Tuple[float, float] = (0.5, 0.5)
) -> float:
    """
    Calcula score combinado basado en consistencia.
    
    Args:
        cosine_sim: Similitud de coseno [0, 1]
        euclidean_sim: Similitud euclidiana [0, 1]
        consistency_ratio: Ratio de consistencia [0, 1]
        weights_high: Pesos (cosine, euclidean) si consistency >= 0.5
        weights_low: Pesos (cosine, euclidean) si consistency < 0.5
        
    Returns:
        Score combinado normalizado [0, 1]
    """
    if consistency_ratio >= 0.5:
        return weights_high[0] * cosine_sim + weights_high[1] * euclidean_sim
    else:
        return weights_low[0] * cosine_sim + weights_low[1] * euclidean_sim


def _calculate_magnitude_ratios(
    candidate_array: np.ndarray,
    query_vectors: List[np.ndarray]
) -> np.ndarray:
    """Calculate magnitude ratios between candidates and query vectors."""
    magnitude_ratios = np.zeros(len(candidate_array))
    for i, candidate_vector in enumerate(candidate_array):
        candidate_norm = np.linalg.norm(candidate_vector)
        if candidate_norm == 0:
            magnitude_ratios[i] = np.inf
            continue
        query_norms = [np.linalg.norm(qv.flatten()) for qv in query_vectors]
        ratios = [max(qn / candidate_norm, candidate_norm / qn) for qn in query_norms if qn > 0]
        magnitude_ratios[i] = max(ratios) if ratios else np.inf
    return magnitude_ratios


def _get_suspicious_mask(
    best_distances: np.ndarray,
    best_cosine_sims: np.ndarray,
    magnitude_filter_mask: np.ndarray,
    total_regions: int,
    apply_strict_filter: bool
) -> np.ndarray:
    """Determine which candidates should be filtered out."""
    if not apply_strict_filter:
        return magnitude_filter_mask
    
    if total_regions <= 2:
        # Imágenes simples: filtro calibrado para permitir variaciones de perspectiva e iluminación
        return ((best_distances > 1.15) & (best_cosine_sims < 0.35)) | magnitude_filter_mask
    else:
        # Imágenes complejas: filtro calibrado
        return (
            ((best_distances > 1.25) & (best_cosine_sims < 0.30)) |
            magnitude_filter_mask
        )


def calculate_vector_euclidean_distances_multi_region(
    query_vectors: List[np.ndarray],
    candidate_vectors: List[Tuple[str, np.ndarray, float, int]],
    apply_strict_filter: bool = True,
    total_regions: int = 1
) -> List[VectorDistanceResult]:
    """Calcula distancias euclidianas entre vectores de query y candidatos."""
    if not candidate_vectors or not query_vectors:
        logger.warning("⚠️ [EUCLIDEAN-MULTI] No hay vectores candidatos o query")
        return []
    
    logger.info(f"\n🔍 [EUCLIDEAN-MULTI] Calculando distancias contra {len(query_vectors)} regiones...")
    
    # Extraer datos
    piece_ids = [pid for pid, _, _, _ in candidate_vectors]
    candidate_array = np.vstack([vec for _, vec, _, _ in candidate_vectors])
    cosine_sims = [sim for _, _, sim, _ in candidate_vectors]
    match_counts = [count for _, _, _, count in candidate_vectors]
    
    # Calcular mejores distancias por región
    best_distances = np.full(len(candidate_vectors), np.inf)
    best_region_indices = np.zeros(len(candidate_vectors), dtype=int)
    best_cosine_sims = np.array(cosine_sims)
    
    for region_idx, query_vector in enumerate(query_vectors):
        qv = query_vector.reshape(1, -1) if query_vector.ndim == 1 else query_vector
        region_distances = euclidean_distances(qv, candidate_array)[0]
        region_cosine_sims = cosine_similarity(qv, candidate_array)[0]
        
        for i in range(len(candidate_vectors)):
            if region_distances[i] < best_distances[i]:
                best_distances[i] = region_distances[i]
                best_region_indices[i] = region_idx
                best_cosine_sims[i] = region_cosine_sims[i]
    
    # Convertir a similitudes y calcular scores
    best_euclidean_sims = 1.0 / (1.0 + best_distances)
    combined_scores = np.array([
        _calculate_combined_score(best_cosine_sims[i], best_euclidean_sims[i], match_counts[i] / max(total_regions, 1))
        for i in range(len(best_cosine_sims))
    ])
    
    logger.info(f"   ✅ Distancias: [{best_distances.min():.4f}, {best_distances.max():.4f}]")
    
    # Calcular máscaras de filtro
    magnitude_ratios = _calculate_magnitude_ratios(candidate_array, query_vectors)
    magnitude_filter_mask = magnitude_ratios > 3.0
    
    if np.any(magnitude_filter_mask):
        logger.warning(f"   ⚠️ [MAGNITUDE_FILTER] Rechazando {np.sum(magnitude_filter_mask)} candidatos")
    
    # Determinar máscara de validez
    suspicious_mask = _get_suspicious_mask(best_distances, best_cosine_sims, magnitude_filter_mask, total_regions, apply_strict_filter)
    valid_mask = ~suspicious_mask
    
    if np.any(suspicious_mask) and apply_strict_filter:
        logger.warning(f"   🚫 Filtro estricto: Rechazando {np.sum(suspicious_mask)} candidatos")
    
    # Construir resultados
    results = _build_distance_results(piece_ids, list(best_cosine_sims), best_distances, combined_scores, valid_mask)
    
    if not results:
        logger.warning("⚠️ No hay resultados válidos")
    
    return results


def _build_distance_results(
    piece_ids: List[str],
    cosine_sims: List[float],
    euclidean_dists: np.ndarray,
    combined_scores: np.ndarray,
    valid_mask: np.ndarray
) -> List[VectorDistanceResult]:
    """Helper to build VectorDistanceResult list from valid indices."""
    results = []
    for i in range(len(piece_ids)):
        if valid_mask[i]:
            logger.debug(f"      ✅ {piece_ids[i]}: cosine={cosine_sims[i]:.4f}, "
                        f"euclidean={euclidean_dists[i]:.4f}, combined={combined_scores[i]:.4f}")
            results.append(VectorDistanceResult(
                piece_id=piece_ids[i],
                cosine_similarity=cosine_sims[i],
                euclidean_distance=float(euclidean_dists[i]),
                combined_score=float(combined_scores[i])
            ))
    return results


def calculate_vector_euclidean_distances(
    query_vector: np.ndarray,
    candidate_vectors: List[Tuple[str, np.ndarray, float, int]],
    apply_strict_filter: bool = True,
    total_regions: int = 1
) -> List[VectorDistanceResult]:
    """
    Calcula distancias euclidianas entre el vector de consulta y los vectores candidatos.
    Combina similitud de coseno (ya calculada) con distancia euclidiana para ranking mejorado.
    """
    if not candidate_vectors:
        logger.warning("⚠️ [EUCLIDEAN] No hay vectores candidatos")
        return []
    
    logger.info("\n🔍 [EUCLIDEAN] Calculando distancias euclidianas entre vectores...")
    logger.info(f"   📊 Query vector dim: {query_vector.shape}, Candidatos: {len(candidate_vectors)}")
    
    # Asegurar shape correcto
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)
    
    # Extraer datos
    piece_ids = [pid for pid, _, _, _ in candidate_vectors]
    vectors_array = np.vstack([vec for _, vec, _, _ in candidate_vectors])
    cosine_sims = [sim for _, _, sim, _ in candidate_vectors]
    match_counts = [count for _, _, _, count in candidate_vectors]
    
    # Cálculo vectorizado
    euclidean_dists = euclidean_distances(query_vector, vectors_array)[0]
    logger.info(f"   ✅ Distancias calculadas - Rango: [{euclidean_dists.min():.4f}, {euclidean_dists.max():.4f}]")
    
    # Convertir a similitud
    euclidean_similarity = 1.0 / (1.0 + euclidean_dists)
    logger.info(f"   📐 Similitudes euclidianas - Rango: [{euclidean_similarity.min():.4f}, {euclidean_similarity.max():.4f}]")
    
    # Combinar scores
    combined_scores = np.array([
        _calculate_combined_score(cosine_sims[i], euclidean_similarity[i], match_counts[i] / max(total_regions, 1))
        for i in range(len(cosine_sims))
    ])
    
    # Determinar máscara de validez
    if apply_strict_filter:
        suspicious_mask = (euclidean_dists > 1.15) & (np.array(cosine_sims) < 0.35)
        if np.any(suspicious_mask):
            logger.warning(f"   🚫 Filtro estricto: Rechazando {np.sum(suspicious_mask)} candidatos")
        valid_mask = ~suspicious_mask
    else:
        valid_mask = np.ones(len(piece_ids), dtype=bool)
    
    # Construir resultados
    results = _build_distance_results(piece_ids, cosine_sims, euclidean_dists, combined_scores, valid_mask)
    
    if not results:
        logger.warning("   ⚠️ No hay resultados válidos")
        return []
    
    # Ordenar y log
    results.sort(key=lambda x: x.combined_score, reverse=True)
    logger.info(f"   🎯 Mejor match: {results[0].piece_id} - "
               f"cosine={results[0].cosine_similarity:.4f}, "
               f"euclidean={results[0].euclidean_distance:.4f}, "
               f"combined={results[0].combined_score:.4f}")
    
    return results


def calculate_similarities_batch(
    regions: List[Tuple[np.ndarray, Dict[str, Any]]],
    database_vectors: List[Tuple[str, np.ndarray]],
    min_similarity: float = 0.35
) -> List[RegionMatch]:
    """
    Calcula similitudes en batch usando operaciones vectorizadas.
    
    Args:
        regions: Lista de tuplas (vector, metadata)
        database_vectors: Lista de tuplas (faiss_id, vector) de la BD
        min_similarity: Umbral mínimo de similitud
        
    Returns:
        Lista de RegionMatch con todos los matches válidos
    """
    if not regions or not database_vectors:
        return []
    
    logger.info(f"🚀 [BATCH_CALC] Calculando similitudes en batch: {len(regions)} regiones × {len(database_vectors)} vectores")
    
    # Apilar todos los vectores de regiones en una matriz
    region_vectors = np.vstack([region_vector for region_vector, _ in regions])
    region_metas = [meta for _, meta in regions]
    
    # Apilar todos los vectores de BD en una matriz
    db_vectors = np.vstack([vec for _, vec in database_vectors])
    db_faiss_ids = [faiss_id for faiss_id, _ in database_vectors]
    
    # 🚀 CÁLCULO EN BATCH: Una sola operación matricial para TODAS las similitudes
    # Resultado: matriz de (n_regions, n_db_vectors)
    similarities_matrix = cosine_similarity(region_vectors, db_vectors)
    
    logger.info(f"   ✅ Matriz de similitudes calculada: {similarities_matrix.shape}")
    
    # 🎯 FILTRADO VECTORIZADO: Usar máscaras de NumPy en lugar de loops
    valid_mask = similarities_matrix >= min_similarity
    valid_indices = np.nonzero(valid_mask)
    
    # Extraer matches válidos usando list comprehension
    matches = [RegionMatch(
        region_meta=region_metas[idx[0]],
        faiss_id=db_faiss_ids[idx[1]],
        real_piece_id=db_faiss_ids[idx[1]],  # Se actualizará después con el mapeo
        similarity=float(similarities_matrix[idx[0], idx[1]])
    ) for idx in zip(valid_indices[0], valid_indices[1])]
    
    logger.info(f"   📊 Matches válidos encontrados: {len(matches)} (umbral >= {min_similarity})")
    return matches


def apply_consistency_adjustments_vectorized(
    df: pl.DataFrame,
    total_regions: int,
    consistency_boost: float = 1.10,
    low_consistency_penalty: float = 0.80
) -> pl.DataFrame:
    """
    Aplica ajustes de consistencia usando operaciones vectorizadas de Polars.
    
    Args:
        df: DataFrame de Polars con columnas ['faiss_id', 'similarity', 'count', 'same_piece_count']
        total_regions: Número total de regiones
        consistency_boost: Factor de boost para alta consistencia
        low_consistency_penalty: Factor de penalty para baja consistencia
        
    Returns:
        DataFrame con similitudes ajustadas
    """
    logger.info("🔧 [CONSISTENCY] Aplicando ajustes vectorizados...")
    
    # Calcular ratio de consistencia vectorizado
    df = df.with_columns(
        (pl.col("count") / total_regions).alias("consistency_ratio")
    )
    
    # 🚀 BOOST VECTORIZADO: Alta consistencia (≥50% de regiones) Y similitud MUY fuerte (≥0.85)
    # CAMBIO: Solo dar boost si la similitud es realmente muy fuerte
    high_consistency_mask = (
        (pl.col("consistency_ratio") >= 0.5) &
        (pl.col("similarity") >= 0.85)
    )
    
    # 🚨 WEAK MATCH: Alta consistencia pero similitud débil (<0.75)
    # Esto detecta cuando todas las regiones coinciden pero con match débil
    # CAMBIO: Solo aplicar penalty si consistencia < 100% (no todas las regiones)
    weak_consistent_match_mask = (
        (pl.col("consistency_ratio") >= 0.5) &
        (pl.col("consistency_ratio") < 1.0) &  # 🆕 No penalizar si 100% consistencia
        (pl.col("similarity") < 0.75)
    )
    
    # 📉 PENALTY VECTORIZADO: Baja consistencia (≤20% con mínimo 3 regiones)
    low_consistency_mask = (
        (pl.col("consistency_ratio") <= 0.2) & 
        (total_regions >= 3)
    )
    
    # 🚨 PENALTY ANTI-FALSOS POSITIVOS: Solo 1 región coincide (muy sospechoso)
    # Si solo matchea 1 región de muchas, es probable un falso positivo
    single_match_suspicious_mask = (
        (pl.col("count") == 1) & 
        (total_regions >= 3)
    )
    
    # Aplicar ajustes usando when/then vectorizado (equivalente a np.where)
    df = df.with_columns([
        pl.col("similarity").alias("original_similarity"),
        pl.when(high_consistency_mask)
        .then(pl.col("similarity") * consistency_boost)
        .when(weak_consistent_match_mask)
        .then(pl.col("similarity") * 0.70)  # 🚨 Penalty agresivo: -30% para matches débiles pero consistentes
        .when(single_match_suspicious_mask)
        .then(pl.col("similarity") * 0.60)  # 🚨 Penalty agresivo: -40% para matches de 1 sola región
        .when(low_consistency_mask)
        .then(pl.col("similarity") * low_consistency_penalty)
        .otherwise(pl.col("similarity"))
        .clip(upper_bound=1.0)  # Asegurar que no exceda 1.0
        .alias("adjusted_similarity"),
        
        # Crear columna de razón de ajuste
        pl.when(high_consistency_mask)
        .then(pl.lit(f"boost x{consistency_boost}"))
        .when(weak_consistent_match_mask)
        .then(pl.lit("penalty x0.70 (match débil pero consistente - posible falso positivo)"))
        .when(single_match_suspicious_mask)
        .then(pl.lit("penalty x0.60 (falso positivo sospechoso - solo 1 región)"))
        .when(low_consistency_mask)
        .then(pl.lit(f"penalty x{low_consistency_penalty}"))
        .otherwise(pl.lit("ninguno"))
        .alias("adjustment_reason")
    ])
    
    # Log de ajustes aplicados
    boost_count = df.filter(high_consistency_mask).height
    weak_count = df.filter(weak_consistent_match_mask).height
    penalty_count = df.filter(low_consistency_mask).height
    suspicious_count = df.filter(single_match_suspicious_mask).height
    
    logger.info(f"   ✅ Ajustes aplicados: {boost_count} boosts, {weak_count} weak, {penalty_count} penalties, {suspicious_count} sospechosos")
    
    if suspicious_count > 0:
        logger.warning(f"   🚨 {suspicious_count} matches sospechosos detectados (solo 1 región coincide)")
    
    if weak_count > 0:
        logger.warning(f"   ⚠️ {weak_count} matches débiles detectados (similitud < 0.75 - posibles falsos positivos)")
    
    return df


def apply_relative_penalties_vectorized(
    df: pl.DataFrame
) -> pl.DataFrame:
    """
    Aplica penalties relativos usando operaciones vectorizadas de Polars.
    Solo penaliza matches débiles cuando hay un match dominante.
    """
    if df.height <= 1:
        # Si hay 1 o menos filas, agregar columna final_similarity sin penalties
        return df.with_columns([
            pl.col("adjusted_similarity").alias("final_similarity"),
            pl.lit(False).alias("has_relative_penalty")
        ])
    
    # Obtener la mejor similitud
    best_similarity = df.select(pl.col("adjusted_similarity").max()).item()
    
    if best_similarity < 0.90:
        # No aplicar penalty relativo si no hay match dominante
        return df.with_columns([
            pl.col("adjusted_similarity").alias("final_similarity"),
            pl.lit(False).alias("has_relative_penalty")
        ])
    
    logger.info(f"⚡ [RELATIVE_PENALTY] Match dominante detectado: {best_similarity:.4f}")
    
    # Calcular gap vectorizado
    df = df.with_columns(
        (best_similarity - pl.col("adjusted_similarity")).alias("gap")
    )
    
    # 🆕 PENALTY AGRESIVO: Match casi perfecto (≥0.95)
    if best_similarity >= 0.95:
        # Penalty muy agresivo para matches <0.90 con gap ≥0.15
        very_aggressive_mask = (
            (pl.col("adjusted_similarity") < 0.90) & 
            (pl.col("gap") >= 0.15)
        )
        
        # Penalty agresivo para matches <0.90 con gap 0.05-0.15
        aggressive_mask = (
            (pl.col("adjusted_similarity") < 0.90) & 
            (pl.col("gap") >= 0.05) & 
            (pl.col("gap") < 0.15)
        )
        
        df = df.with_columns([
            pl.when(very_aggressive_mask)
            .then(pl.col("adjusted_similarity") * 0.50)
            .when(aggressive_mask)
            .then(pl.col("adjusted_similarity") * 0.65)
            .otherwise(pl.col("adjusted_similarity"))
            .alias("final_similarity"),
            
            pl.when(very_aggressive_mask | aggressive_mask)
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("has_relative_penalty")
        ])
    else:
        # ⚠️ PENALTY ORIGINAL: Solo para matches débiles (<0.85)
        weak_high_gap_mask = (
            (pl.col("adjusted_similarity") < 0.85) & 
            (pl.col("gap") >= 0.3)
        )
        
        weak_medium_gap_mask = (
            (pl.col("adjusted_similarity") < 0.85) & 
            (pl.col("gap") >= 0.2) & 
            (pl.col("gap") < 0.3)
        )
        
        df = df.with_columns([
            pl.when(weak_high_gap_mask)
            .then(pl.col("adjusted_similarity") * 0.60)
            .when(weak_medium_gap_mask)
            .then(pl.col("adjusted_similarity") * 0.75)
            .otherwise(pl.col("adjusted_similarity"))
            .alias("final_similarity"),
            
            pl.when(weak_high_gap_mask | weak_medium_gap_mask)
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("has_relative_penalty")
        ])
    
    return df


def find_best_matches_global(
    regions: List[Tuple[np.ndarray, Dict[str, Any]]],
    database_vectors: List[Tuple[str, np.ndarray]],
    faiss_to_piece_map: Dict[str, str],
    top_k: int = 10,
    min_similarity: float = 0.35,
    consistency_boost: float = 1.10,
    low_consistency_penalty: float = 0.80
) -> List[PieceMatch]:
    """
    Versión optimizada que encuentra los mejores matches usando operaciones vectorizadas.
    
    Implementa búsqueda multi-región optimizada:
    1. 🚀 Cálculo en BATCH: Todas las similitudes en una operación matricial
    2. 🎯 Filtrado vectorizado: Máscaras de NumPy para threshold
    3. 📊 Agrupación con Polars: Ultra-rápida con .groupby().agg()
    4. ⚡ Ajustes vectorizados: Boost/penalty aplicados con when/then
    5. 🔄 Consolidación eficiente: Mejor match por piece_id
    
    Args:
        regions: Lista de tuplas (vector, metadata)
        database_vectors: Lista de tuplas (faiss_id, vector) de la BD
        faiss_to_piece_map: Mapeo de faiss_id a piece_id real
        top_k: Número de mejores matches a retornar
        min_similarity: Umbral mínimo de similitud
        consistency_boost: Factor boost para alta consistencia (default: 1.10)
        low_consistency_penalty: Factor penalty para baja consistencia (default: 0.80)
        
    Returns:
        Lista de PieceMatch con los mejores matches optimizados
    """
    logger.info("\n🔍 [GLOBAL_SEARCH_OPTIMIZED] Búsqueda vectorizada ultra-rápida...")
    logger.info(f"   📊 Comparando {len(regions)} regiones contra {len(database_vectors)} vectores BD")
    
    # 🚀 PASO 1: Cálculo en BATCH de todas las similitudes
    all_matches = calculate_similarities_batch(regions, database_vectors, min_similarity)
    
    # Actualizar real_piece_id usando list comprehension
    for match in all_matches:
        match.real_piece_id = faiss_to_piece_map.get(match.faiss_id, match.faiss_id)
    
    logger.info(f"   📈 Total de matches encontrados: {len(all_matches)}")
    
    # 📊 PASO 2: Agrupación ultra-rápida con Polars
    if not all_matches:
        logger.info("   ⚠️ No hay matches para agrupar")
        return []
    
    # Convertir a DataFrame de Polars para operaciones eficientes
    matches_data = [
        {
            "faiss_id": match.faiss_id,
            "real_piece_id": match.real_piece_id,
            "similarity": match.similarity,
            "region_label": match.region_meta.get('label', 'unknown'),
            "region_type": match.region_meta.get('type', 'unknown')
        }
        for match in all_matches
    ]
    
    df = pl.DataFrame(matches_data)
    
    # 🚀 AGRUPACIÓN VECTORIZADA: Una sola operación para todas las estadísticas
    # IMPORTANTE: Agrupar solo por faiss_id para calcular consistencia correctamente.
    # Si 3 regiones matchean la misma pieza, count=3 → consistency_ratio alto → BOOST.
    grouped_df = df.group_by("faiss_id").agg([
        pl.col("similarity").max().alias("best_similarity"),
        pl.col("similarity").count().alias("count"),
        pl.col("real_piece_id").first().alias("real_piece_id"),
        pl.col("region_label").first().alias("best_region_label"),
        pl.col("region_type").first().alias("best_region_type"),
        # Contar matches de la misma pieza
        (pl.col("real_piece_id") == pl.col("real_piece_id").first()).sum().alias("same_piece_count")
    ])
    
    # � PASO 3: Ajustes de consistencia vectorizados
    total_regions = len(regions)
    adjusted_df = apply_consistency_adjustments_vectorized(
        grouped_df.rename({"best_similarity": "similarity"}),
        total_regions,
        consistency_boost,
        low_consistency_penalty
    )
    
    # 📈 PASO 4: Penalties relativos vectorizados
    final_df = apply_relative_penalties_vectorized(adjusted_df)
    
    # 🔄 PASO 5: Ordenar y convertir a PieceMatch
    final_df = final_df.sort("final_similarity", descending=True)
    
    # Convertir a PieceMatch usando list comprehension
    best_matches = [PieceMatch(
        piece_id=row["faiss_id"],
        real_piece_id=row["real_piece_id"],
        similarity=row["final_similarity"],
        original_similarity=row["original_similarity"],
        match_count=row["count"],
        same_piece_count=row["same_piece_count"],
        region={
            "label": row["best_region_label"],
            "type": row["best_region_type"]
        },
        has_relative_penalty=row.get("has_relative_penalty", False)
    ) for row in final_df.iter_rows(named=True)]
    
    logger.info(f"   🎯 Piezas únicas encontradas: {len(best_matches)}")
    
    #  NO LIMITAR AQUÍ: Retornar TODOS los matches para que la agrupación posterior
    # pueda consolidar múltiples ángulos y LUEGO aplicar el top_k
    # Esto evita cortar piezas diferentes que tienen múltiples ángulos en BD
    logger.info(f"   🔄 Retornando {len(best_matches)} matches optimizados")
    
    # Log de los mejores resultados
    if best_matches:
        logger.info("\n   📋 Top 10 matches optimizados:")
        for i, match in enumerate(best_matches[:10], 1):
            logger.info(f"      {i}. {match.piece_id} (piece: {match.real_piece_id[:8]}...): {match.similarity:.4f} "
                       f"(mejor de {match.match_count} matches, "
                       f"mejor región: {match.region.get('label', 'unknown')})")
    
    return best_matches


def _prepare_regions(
    piece_vector: List[float],
    object_vectors: List[Dict[str, Any]]
) -> List[Tuple[np.ndarray, Dict[str, Any]]]:
    """Prepare regions for global search from piece and object vectors."""
    regions: List[Tuple[np.ndarray, Dict[str, Any]]] = []
    
    if piece_vector and len(piece_vector) > 0:
        regions.append((
            np.array(piece_vector, dtype=np.float32),
            {'label': 'piece', 'type': 'piece'}
        ))
        logger.info(f"   ✅ Agregada región: piece (dim={len(piece_vector)})")
    
    if object_vectors:
        valid_objects = [
            (np.array(obj.get('vector', []), dtype=np.float32), 
             {'label': f"{obj.get('type', 'object')}_{i}", 'type': 'object'})
            for i, obj in enumerate(object_vectors)
            if obj.get('vector') and len(obj.get('vector', [])) > 0
        ]
        regions.extend(valid_objects)
        for _, obj_meta in valid_objects:
            logger.info(f"   ✅ Agregada región: {obj_meta['label']}")
    
    return regions


def _calculate_geometric_penalty(regions: List[Tuple[np.ndarray, Dict[str, Any]]]) -> float:
    """Calculate geometric penalty based on angle between piece and object vectors."""
    if len(regions) <= 1:
        return 1.0
    
    try:
        piece_vec = regions[0][0]
        obj_vecs = [r[0] for r in regions[1:]]
        
        if not obj_vecs:
            return 1.0
        
        avg_obj = np.mean(obj_vecs, axis=0)
        angle_rad = np.arccos(np.clip(
            np.dot(piece_vec, avg_obj) / (np.linalg.norm(piece_vec) * np.linalg.norm(avg_obj)),
            -1, 1
        ))
        angle_deg = np.degrees(angle_rad)
        
        logger.info(f"   📐 [GEOMETRIC_CHECK] Ángulo pieza vs objetos: {angle_deg:.1f}°")
        
        if angle_deg > 70:
            logger.warning("   🚨 [GEOMETRIC_CHECK] Ángulo alto: penalty x0.7")
            return 0.7
        return 1.0
    except Exception as e:
        logger.warning(f"   ⚠️ [GEOMETRIC_CHECK] Error: {e}")
        return 1.0


def _consolidate_candidates(candidate_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Consolidate candidates from multiple regions into unique candidate dict."""
    unique_candidates = {}
    for result in candidate_results:
        piece_id = result['piece_id']
        if piece_id not in unique_candidates:
            unique_candidates[piece_id] = {
                'similarity': result['similarity'],
                'regions': [result['region_meta']],
                'match_count': 1
            }
        else:
            unique_candidates[piece_id]['similarity'] = max(
                unique_candidates[piece_id]['similarity'],
                result['similarity']
            )
            unique_candidates[piece_id]['regions'].append(result['region_meta'])
            unique_candidates[piece_id]['match_count'] += 1
    return unique_candidates


def _build_piece_mappings(
    faiss_ids_in_db: List[str],
    piece_vector_repository: 'PieceVectorRepository'
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build faiss_id to piece_id and catalog_unique_id mappings."""
    faiss_to_catalog_map = {}
    faiss_to_piece_map = {}
    
    try:
        piece_vectors_data = piece_vector_repository.get_by_ids_sync(faiss_ids_in_db)
        
        if piece_vectors_data:
            faiss_to_piece_map = {faiss_id: piece_id for faiss_id, piece_id, _ in piece_vectors_data}
            faiss_to_catalog_map = {faiss_id: catalog_unique_id for faiss_id, _, catalog_unique_id in piece_vectors_data}
        
        found_ids = set(faiss_to_piece_map.keys())
        missing_ids = set(faiss_ids_in_db) - found_ids
        
        if missing_ids:
            logger.warning(f"      ⚠️ {len(missing_ids)} faiss_ids no encontrados en BD")
            faiss_to_piece_map.update({faiss_id: faiss_id for faiss_id in missing_ids})
            faiss_to_catalog_map.update({faiss_id: faiss_id for faiss_id in missing_ids})
        
        logger.info(f"   ✅ Mapeo completado: {len(piece_vectors_data) if piece_vectors_data else 0} vectores")
    except Exception as e:
        logger.error(f"      ❌ Error obteniendo mapeo: {e}")
        faiss_to_piece_map = {faiss_id: faiss_id for faiss_id in faiss_ids_in_db}
        faiss_to_catalog_map = {faiss_id: faiss_id for faiss_id in faiss_ids_in_db}
    
    return faiss_to_piece_map, faiss_to_catalog_map


def _group_results_by_catalog(
    results: List[Tuple],
    geometric_penalty: float
) -> List[Tuple[str, float, Optional[Dict]]]:
    """Group results by catalog_unique_id using Polars."""
    if not results:
        return []
    
    df = pl.DataFrame({
        "catalog_unique_id": [r[0] for r in results],
        "final_score": [r[1] for r in results],
        "real_piece_id": [r[2] for r in results],
        "match_count": [r[3] for r in results],
        "same_piece_count": [r[4] for r in results],
        "euclidean_info": [r[5] for r in results]
    })
    
    grouped_df = df.group_by("catalog_unique_id").agg([
        pl.col("final_score").max().alias("best_score"),
        pl.col("real_piece_id").first().alias("piece_id"),
        pl.col("match_count").sum().alias("total_matches"),
        pl.col("same_piece_count").sum().alias("total_same_piece"),
        pl.struct([
            pl.col("final_score"),
            pl.col("euclidean_info")
        ]).sort_by("final_score", descending=True).first().struct.field("euclidean_info").alias("best_euclidean")
    ])
    
    final_results = [
        (row["catalog_unique_id"], row["best_score"], row["best_euclidean"])
        for row in grouped_df.iter_rows(named=True)
    ]
    
    if geometric_penalty < 1.0:
        logger.info(f"   🚨 Aplicando penalty geométrico x{geometric_penalty}")
        final_results = [(cid, score * geometric_penalty, euc) for cid, score, euc in final_results]
    
    final_results.sort(key=lambda x: x[1], reverse=True)
    return final_results


def _search_faiss_candidates(
    regions: List[Tuple[np.ndarray, Dict[str, Any]]],
    faiss_repository: 'FaissIndexRepository',
    _subscribe_sync: Callable,
    prefilter_k: int = 100
) -> List[Dict[str, Any]]:
    """Search FAISS for top candidates per region."""
    candidate_results = []
    
    for i, (region_vector, region_meta) in enumerate(regions):
        try:
            logger.info(f"   🔍 Región {i+1}/{len(regions)}: {region_meta.get('label', 'unknown')}")
            faiss_candidates_obs = faiss_repository.search_similar_sync(region_vector.tolist(), k=prefilter_k)
            faiss_candidates = _subscribe_sync(faiss_candidates_obs)
            
            for piece_id, similarity in faiss_candidates:
                candidate_results.append({
                    'piece_id': piece_id,
                    'similarity': similarity,
                    'region_meta': region_meta,
                    'region_index': i
                })
            logger.info(f"      ✅ {len(faiss_candidates)} candidatos")
        except Exception as e:
            logger.warning(f"      ⚠️ Error: {e}")
            continue
    
    return candidate_results


def _fetch_candidate_vectors(
    unique_candidate_ids: List[str],
    faiss_repository: 'FaissIndexRepository',
    _subscribe_sync: Callable
) -> List[Tuple[str, np.ndarray]]:
    """Fetch actual vectors for candidate IDs."""
    database_vectors = []
    for piece_id in unique_candidate_ids:
        try:
            vector_obs = faiss_repository.get_vector_by_id(piece_id)
            vector = _subscribe_sync(vector_obs)
            database_vectors.append((piece_id, np.array(vector, dtype=np.float32)))
        except Exception as e:
            logger.warning(f"      ⚠️ No vector for {piece_id}: {e}")
            continue
    return database_vectors


def _calculate_euclidean_map(
    regions: List[Tuple[np.ndarray, Dict[str, Any]]],
    best_matches: List['PieceMatch'],
    database_vectors: List[Tuple[str, np.ndarray]],
    min_similarity: float
) -> Tuple[Dict[str, 'VectorDistanceResult'], bool]:
    """Calculate euclidean distances and return result map and whether all were rejected."""
    euclidean_results_map = {}
    
    if not best_matches or not regions:
        return euclidean_results_map, False
    
    try:
        query_vectors = [region_vector for region_vector, _ in regions]
        db_vector_dict = dict(database_vectors)
        
        candidate_vectors = [
            (match.piece_id, db_vector_dict[match.piece_id], match.similarity, match.match_count)
            for match in best_matches
            if match.piece_id in db_vector_dict
        ]
        
        if not candidate_vectors:
            return euclidean_results_map, False
        
        apply_strict_filter = min_similarity >= 0.5 and len(query_vectors) <= 2
        euclidean_results = calculate_vector_euclidean_distances_multi_region(
            query_vectors, candidate_vectors,
            apply_strict_filter=apply_strict_filter,
            total_regions=len(regions)
        )
        
        euclidean_results_map = {er.piece_id: er for er in euclidean_results}
        
        if apply_strict_filter and len(euclidean_results) == 0:
            logger.warning("🚫 [EUCLIDEAN] Filtro estricto rechazó todos")
            return euclidean_results_map, True
            
    except Exception as e:
        logger.warning(f"⚠️ [EUCLIDEAN] Error: {e}")
    
    return euclidean_results_map, False


def _convert_to_final_results(
    best_matches: List['PieceMatch'],
    euclidean_results_map: Dict[str, 'VectorDistanceResult'],
    faiss_to_catalog_map: Dict[str, str]
) -> List[Tuple]:
    """Convert matches to final result tuples."""
    if not euclidean_results_map:
        return [(
            faiss_to_catalog_map.get(m.piece_id, m.piece_id),
            m.similarity,
            m.real_piece_id,
            m.match_count,
            m.same_piece_count,
            {
                'euclidean_distance': 0.0,
                'combined_score': m.similarity
            }
        ) for m in best_matches]

    filtered_matches = [m for m in best_matches if m.piece_id in euclidean_results_map]
    return [(
        faiss_to_catalog_map.get(m.piece_id, m.piece_id),
        euclidean_results_map[m.piece_id].combined_score,
        m.real_piece_id,
        m.match_count,
        m.same_piece_count,
        {
            'euclidean_distance': euclidean_results_map[m.piece_id].euclidean_distance,
            'combined_score': euclidean_results_map[m.piece_id].combined_score
        }
    ) for m in filtered_matches]


def search_vectors_operator(
    faiss_repository: FaissIndexRepository,
    piece_vector_repository: PieceVectorRepository,
    k: Optional[int] = None,
    threshold: Optional[float] = None,
    max_results: Optional[int] = None,
) -> Callable[[Observable], Observable]:
    """
    RxPY operator that searches for similar vectors using advanced cosine similarity.
    Now supports multi-vector search (piece + detected objects) and euclidean distance ranking.
    
    🚀 OPTIMIZACIÓN FAISS COMPLETA: En lugar de cargar TODOS los vectores de la BD (brute force),
    usa faiss_index.search() para pre-filtrar top candidatos por región y luego get_vector_by_id()
    para obtener solo los vectores necesarios usando FAISS reconstruct.
    
    El flujo optimizado es:
    1. faiss_index.search(k) → obtiene piece_ids de candidatos (no índices FAISS)
    2. get_vector_by_id(piece_id) → mapea piece_id → índice FAISS real → reconstruct
    
    Esto reduce dramáticamente memoria y cómputo vs el approach anterior.
    
    Combina dos métricas complementarias:
    1. Similitud de Coseno: Mide la dirección/orientación entre vectores (0-1)
    2. Distancia Euclidiana: Mide la distancia real en el espacio vectorial
    
    El score combinado (70% cosine + 30% euclidean normalizado) mejora el ranking
    al considerar ambas métricas, reduciendo falsos positivos.
    
    Args:
        faiss_repository: FAISS index repository
        piece_vector_repository: Repository to get piece_id from vector_id
        k: Number of similar vectors to return (default: 5)
        threshold: Minimum similarity threshold to include results
        max_results: Maximum number of results to return
        
    Returns:
        RxPY operator function that transforms Dict[str, Any] to List[Tuple[str, float, Optional[Dict]]]
        
        Input: {
            'piece_vector': List[float],
            'object_vectors': List[Dict]
        }
        
        Output: List of (catalog_unique_id, combined_score, euclidean_info) tuples
            - catalog_unique_id: str
            - combined_score: float (0.0 to 1.0) - Combinación de cosine y euclidean
            - euclidean_info: Optional[Dict] = {
                'euclidean_distance': float,  # Distancia en espacio vectorial
                'combined_score': float       # Score final combinado
              }
    """
    
    def _subscribe_sync(observable: Observable):
        """Helper to subscribe to Observable and get value synchronously using run()."""
        try:
            # Use run() which blocks until the observable completes and returns the last value
            # For observables that emit a single value, this is the correct approach
            result = observable.run()
            return result
        except Exception as e:
            logger.error(f"🔍 [_subscribe_sync] run() error: {e}")
            raise
    
    def search_similar_vectors_step(vectors_data: Dict[str, Any]) -> List[Tuple[str, float, Optional[Dict[str, Any]]]]:
        """Search for similar vectors using multiple query vectors with euclidean distance ranking."""
        try:
            num_results = k if k else 5
            min_similarity = threshold if threshold is not None else 0.35
            
            piece_vector = vectors_data.get('piece_vector', [])
            object_vectors = vectors_data.get('object_vectors', [])
            
            logger.info(f"🔧 [SEARCH_VECTORS] k={num_results}, threshold={min_similarity}")
            
            # Prepare regions using helper
            regions = _prepare_regions(piece_vector, object_vectors)
            if not regions:
                raise CosineSimilaritySearchException(
                    message="No vectors to search",
                    detail={"reason": "No valid vectors provided"}
                )
            
            # Calculate geometric penalty using helper
            geometric_penalty = _calculate_geometric_penalty(regions)
            
            # Search FAISS for candidates using helper
            logger.info("\n🚀 [FAISS] Searching candidates...")
            candidate_results = _search_faiss_candidates(regions, faiss_repository, _subscribe_sync)
            
            # Consolidate candidates using helper
            unique_candidates = _consolidate_candidates(candidate_results)
            unique_candidate_ids = list(unique_candidates.keys())
            logger.info(f"   📊 {len(unique_candidate_ids)} unique candidates")
            
            if not unique_candidate_ids:
                raise CosineSimilaritySearchException(
                    message="No candidates found",
                    detail={"reason": "No candidates found in any region"}
                )
            
            # Fetch vectors using helper
            database_vectors = _fetch_candidate_vectors(unique_candidate_ids, faiss_repository, _subscribe_sync)
            if not database_vectors:
                raise CosineSimilaritySearchException(
                    message="No candidate vectors found",
                    detail={"reason": "No vectors found for prefiltered candidates"}
                )
            
            # Build mappings using helper
            faiss_ids_in_db = [piece_id for piece_id, _ in database_vectors]
            faiss_to_piece_map, faiss_to_catalog_map = _build_piece_mappings(faiss_ids_in_db, piece_vector_repository)
            
            # Find best matches
            best_matches = find_best_matches_global(
                regions=regions,
                database_vectors=database_vectors,
                faiss_to_piece_map=faiss_to_piece_map,
                top_k=9999,
                min_similarity=min_similarity,
                consistency_boost=1.10,
                low_consistency_penalty=0.80
            )
            
            # Calculate euclidean distances using helper
            euclidean_results_map, all_rejected = _calculate_euclidean_map(
                regions, best_matches, database_vectors, min_similarity
            )
            if all_rejected:
                return []
            
            # Convert to final results using helper
            results = _convert_to_final_results(best_matches, euclidean_results_map, faiss_to_catalog_map)
            
            # Group by catalog using helper
            logger.info("\n🔍 [GROUPING] By catalog_unique_id...")
            final_results = _group_results_by_catalog(results, geometric_penalty)
            
            # Apply adaptive filter and limits
            final_results = adaptive_gap_filter(final_results, min_similarity, max_results=9999, total_regions=len(regions))
            
            if len(final_results) > num_results:
                final_results = final_results[:num_results]
            
            if max_results is not None:
                if max_results <= 0:
                    return []
                final_results = final_results[:max_results]
            
            logger.info(f"✅ [SEARCH_VECTORS] Returning {len(final_results)} results")
            return final_results
            
        except Exception as e:
            logger.error(f"❌ [SEARCH_VECTORS] Error: {e}")
            raise CosineSimilaritySearchException(detail={"error": str(e)})
    
    return ops.map(search_similar_vectors_step)



def adaptive_gap_filter(
    results: List[Tuple[str, float, Optional[Dict]]],
    min_similarity: float,
    max_results: int = 10,
    total_regions: int = 1
) -> List[Tuple[str, float, Optional[Dict]]]:
    """
    FILTRO INTELIGENTE:
    - Filtra por min_similarity
    - Si hay un "gap claro" entre top 1 y top 2 → confiar en top 1
    - Si hay muchos matches fuertes → bajar umbral ligeramente
    - Para imágenes multi-pieza (>2 regiones), ser menos agresivo con el gap
    - NUNCA elimina un match válido
    """
    if not results:
        return []
    
    # 1. Filtrar por umbral fijo
    filtered = [r for r in results if r[1] >= min_similarity]
    if not filtered:
        return []
    
    # 2. Ordenar
    filtered.sort(key=lambda x: x[1], reverse=True)
    
    n = len(filtered)
    if n == 1:
        logger.info("[GAP_FILTER] Solo 1 resultado → devolviendo")
        return filtered[:max_results]
    
    scores = np.array([s for _, s, _ in filtered])
    top1, top2 = scores[0], scores[1]
    gap = top1 - top2
    q3 = np.percentile(scores, 75)
    
    logger.info(f"   [GAP_FILTER] top1={top1:.4f}, top2={top2:.4f}, gap={gap:.4f}, Q3={q3:.4f}")
    
    # 3. DECISIÓN DE FILTRO
    final_threshold = min_similarity
    
    # Ajustar gap threshold basado en complejidad de la imagen
    gap_threshold = 0.05 if total_regions > 2 else 0.08  # Más agresivo para multi-pieza
    
    # Caso 1: Gap grande + top1 fuerte → confiar, eliminar débiles (para imágenes con pocos objetos)
    if total_regions <= 3 and gap > gap_threshold and top1 >= 0.80:
        final_threshold = max(min_similarity, top2 + 0.05)  # eliminar todo por debajo de top2 + margen
        logger.info(f"   [GAP_FILTER] Gap grande (>{gap_threshold:.2f}) → subiendo umbral a {final_threshold:.4f} (elimina falsos)")
    
    # Caso 2: Muchos matches fuertes → bajar umbral para incluir más
    elif n >= 5 and q3 > min_similarity + 0.08:
        final_threshold = max(min_similarity * 0.92, q3 - 0.10)
        logger.info(f"   [GAP_FILTER] Masa fuerte → bajando umbral a {final_threshold:.4f}")
    
    # Aplicar
    final_results = [r for r in filtered if r[1] >= final_threshold]
    
    logger.info(f"   [GAP_FILTER] {n} → {len(final_results)} resultados (umbral: {final_threshold:.4f})")
    
    return final_results[:max_results]
