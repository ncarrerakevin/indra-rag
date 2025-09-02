import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from src.infrastructure.vector_store.qdrant_store_optimized import QdrantOptimizedStore

# Inicializar Qdrant
store = QdrantOptimizedStore()
store.initialize_collection_pro(vector_size=768)

# Cargar chunks con embeddings
with open('output/chunks_with_embeddings.json', 'r') as f:
    chunks = json.load(f)

# Preparar datos
texts = [c['content'] for c in chunks]
embeddings = [c['embedding'] for c in chunks]
metadata = [{'chunk_id': c['id'], 'page': i//3 + 1, 'has_image': False} for i, c in enumerate(chunks)]

# Cargar
store.add_documents_batch(texts, embeddings, metadata)
print("✅ Datos cargados en Qdrant")