# src/infrastructure/vector_store/qdrant_store_optimized.py - VERSIÓN CORREGIDA
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct,
    Filter, FieldCondition, MatchValue,
    OptimizersConfigDiff, HnswConfigDiff
)
import uuid
from typing import List, Dict, Optional


class QdrantOptimizedStore:
    def __init__(self, host="localhost", port=6333):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = "indra_rag_optimized"

    def initialize_collection_pro(self, vector_size: int = 768):
        """Configuración PROFESIONAL - VERSIÓN CORREGIDA"""

        # Eliminar colección vieja si existe
        try:
            self.client.delete_collection(self.collection_name)
            print(f"🗑️ Colección anterior eliminada")
        except:
            pass

        # Crear colección OPTIMIZADA - SINTAXIS CORREGIDA
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
                # HNSW optimizado para mejor accuracy
                hnsw_config=HnswConfigDiff(
                    m=32,  # Más conexiones = mejor accuracy
                    ef_construct=200,  # Más esfuerzo en construcción
                    full_scan_threshold=10000
                ),
                # Quantización - SINTAXIS CORREGIDA
                quantization_config={
                    "scalar": {
                        "type": "int8",
                        "quantile": 0.99,
                        "always_ram": True
                    }
                }
            ),
            # Optimizadores para Mac M1
            optimizers_config=OptimizersConfigDiff(
                memmap_threshold=50000,
                indexing_threshold=10000,
                flush_interval_sec=5
            )
        )

        print(f"✅ Colección '{self.collection_name}' creada con configuración ÓPTIMA")
        print(f"   - HNSW con m=32 para máxima accuracy")
        print(f"   - Quantización int8 para velocidad")
        print(f"   - Optimizada para Apple M1")

        # Crear índices después de crear la colección
        try:
            # Índice para página
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="page",
                field_schema="integer"
            )
            print(f"   ✅ Índice 'page' creado")

            # Índice para tipo
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="type",
                field_schema="keyword"
            )
            print(f"   ✅ Índice 'type' creado")

            # Índice para has_image
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name="has_image",
                field_schema="bool"
            )
            print(f"   ✅ Índice 'has_image' creado")

        except Exception as e:
            print(f"   ⚠️ Índices no creados (no crítico): {e}")

    def add_documents_batch(self,
                            texts: List[str],
                            embeddings: List[List[float]],
                            metadata: List[Dict],
                            batch_size: int = 100) -> List[str]:
        """Inserción optimizada en batches"""

        all_ids = []
        total = len(texts)

        for i in range(0, total, batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]

            points = []
            batch_ids = []

            for text, embedding, meta in zip(batch_texts, batch_embeddings, batch_metadata):
                point_id = str(uuid.uuid4())
                batch_ids.append(point_id)

                # Payload enriquecido
                payload = {
                    "content": text,
                    "text": text[:500],  # Preview
                    "page": meta.get("page", 0),
                    "type": meta.get("type", "text"),
                    "has_image": meta.get("has_image", False),
                    "image_path": meta.get("image_path"),
                    "chunk_id": meta.get("chunk_id", 0),
                    "char_count": len(text),
                    "source": meta.get("source", "unknown")
                }

                points.append(PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload
                ))

            # Insertar batch
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True
            )

            all_ids.extend(batch_ids)
            print(f"   📦 Batch {i // batch_size + 1}: {len(batch_ids)} documentos agregados")

        print(f"✅ Total: {len(all_ids)} documentos indexados")
        return all_ids

    def hybrid_search(self,
                      query_embedding: List[float],
                      query_text: str,
                      top_k: int = 5,
                      filters: Optional[Dict] = None) -> List[Dict]:
        """Búsqueda híbrida: vectorial + filtros + scoring mejorado"""

        # Construir filtros
        must_conditions = []

        if filters:
            if "type" in filters:
                must_conditions.append(
                    FieldCondition(
                        key="type",
                        match=MatchValue(value=filters["type"])
                    )
                )
            if "has_image" in filters:
                must_conditions.append(
                    FieldCondition(
                        key="has_image",
                        match=MatchValue(value=filters["has_image"])
                    )
                )

        # Búsqueda vectorial con filtros
        vector_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            query_filter=Filter(must=must_conditions) if must_conditions else None,
            limit=top_k * 2,  # Buscar más para luego filtrar
            with_payload=True,
            score_threshold=0.3
        )

        # Re-scoring con texto
        final_results = []
        keywords = query_text.lower().split() if query_text else []

        for result in vector_results:
            # Score vectorial
            vector_score = result.score

            # Score de texto
            text_score = 0
            if keywords:
                content = result.payload.get("content", "").lower()
                matches = sum(1 for kw in keywords if kw in content)
                text_score = matches / len(keywords) if keywords else 0

            # Score combinado
            combined_score = (vector_score * 0.8) + (text_score * 0.2)

            final_results.append({
                "id": str(result.id),
                "score": combined_score,
                "vector_score": vector_score,
                "text_score": text_score,
                "text": result.payload.get("text", ""),
                "metadata": result.payload
            })

        # Ordenar por score combinado
        final_results.sort(key=lambda x: x["score"], reverse=True)

        return final_results[:top_k]

    def get_statistics(self) -> Dict:
        """Obtener estadísticas detalladas"""
        try:
            info = self.client.get_collection(self.collection_name)

            return {
                "total_vectors": info.vectors_count,
                "indexed_vectors": info.indexed_vectors_count,
                "points_count": info.points_count,
                "segments_count": info.segments_count,
                "status": info.status,
                "optimizer_status": info.optimizer_status,
                "config": {
                    "vector_size": info.config.params.vectors.size,
                    "distance": str(info.config.params.vectors.distance)
                }
            }
        except Exception as e:
            return {"error": str(e)}