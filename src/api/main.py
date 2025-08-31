# src/api/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import sys
import os

# Arreglar el path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.domain.models import QueryRequest, QueryResponse, DocumentInfo, HealthResponse
from src.application.rag_service import RAGService

# Variable global para el servicio
rag_service = None


# Lifespan para inicialización (reemplaza @app.on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global rag_service
    print("🚀 Iniciando RAG Service...")
    rag_service = RAGService()
    print("✅ RAG Service listo")
    yield
    # Shutdown
    print("👋 Cerrando RAG Service...")


# Inicializar FastAPI con lifespan
app = FastAPI(
    title="RAG Multimodal API - Reto Indra",
    description="API para procesamiento inteligente de documentos con RAG multimodal",
    version="1.0.0",
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

    return HealthResponse(
        status="healthy",
        model="gemini-2.5-pro",
        chunks_loaded=len(rag_service.chunks)
    )


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """Realiza una consulta al documento usando RAG"""
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

    return DocumentInfo(**rag_service.get_document_info())


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