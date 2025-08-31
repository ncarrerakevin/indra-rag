# src/domain/models.py
from pydantic import BaseModel
from typing import List, Optional, Dict

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    images: Optional[List[Dict]] = []
    confidence: Optional[float] = None

class DocumentInfo(BaseModel):
    total_chunks: int
    total_pages: int
    people_identified: List[str]
    diagrams_count: int

class HealthResponse(BaseModel):
    status: str
    model: str
    chunks_loaded: int