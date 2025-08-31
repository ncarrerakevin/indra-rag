# src/api/main.py - ACTUALIZADO PARA QDRANT

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import os

# Arreglar el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.domain.models import QueryRequest, QueryResponse, DocumentInfo, HealthResponse
# CAMBIO 1: Importar la versión V2
from src.application.rag_service_v2 import RAGServiceV2  # <-- CAMBIO AQUÍ

# Variable global para el servicio
rag_service = None

# Lifespan para inicialización
@asynccontextmanager
async def lifespan(app: FastAPI):
    global rag_service
    print("🚀 Iniciando RAG Service V2 con Qdrant...")
    rag_service = RAGServiceV2()  # <-- CAMBIO 2: Usar V2
    print("✅ RAG Service V2 con Qdrant listo")
    yield
    print("👋 Cerrando RAG Service...")

# Inicializar FastAPI con lifespan
app = FastAPI(
    title="RAG Multimodal API - Reto Indra",
    description="API para procesamiento inteligente de documentos con RAG multimodal + Qdrant",
    version="2.0.0",  # <-- CAMBIO 3: Nueva versión
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_model=HealthResponse)
async def health_check():
    """Verifica el estado del servicio"""
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service not ready")

    # CAMBIO 4: Adaptar para V2 (no tiene .chunks directo)
    return HealthResponse(
        status="healthy",
        model="gemini-2.0-flash-exp",  # <-- Actualizado
        chunks_loaded=26  # <-- Por ahora hardcodeado, luego lo mejoramos
    )

# CAMBIO 5: Agregar endpoint de estadísticas
@app.get("/stats")
async def get_stats():
    """Obtiene estadísticas del sistema con Qdrant"""
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service not ready")
    return rag_service.get_stats()

# Los demás endpoints siguen igual...

@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """Realiza una consulta al documento usando RAG + Qdrant"""
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service not ready")

    try:
        result = rag_service.query(request.question, request.top_k or 3)
        return QueryResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/document-info", response_model=DocumentInfo)
async def get_document_info():
    """Obtiene información sobre el documento procesado"""
    if not rag_service:
        raise HTTPException(status_code=503, detail="Service not ready")

    # CAMBIO 6: Adaptar para V2
    stats = rag_service.get_stats()
    return DocumentInfo(
        total_chunks=stats['stats']['vectors_count'] if stats.get('stats') else 26,
        total_pages=11,  # Hardcodeado por ahora
        people_identified=["Bob Strahan", "Joe King", "Mofijul Islam", "Vincil Bishop", "David Kaleko", "Rafal Pawlaszek", "Spencer Romo", "Vamsi Thilak Gudi"],
        diagrams_count=2
    )

@app.get("/test-queries")
async def get_test_queries():
    """Devuelve queries de ejemplo para testing"""
    return {
        "queries": [
            "¿Quiénes son los autores del documento?",
            "Muestra el diagrama de arquitectura de la solución",
            "¿Qué es Competiscan y qué resultados obtuvo?",
            "¿Cuáles son los patrones de procesamiento disponibles?",
            "Explica el flujo de procesamiento con su diagrama"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)