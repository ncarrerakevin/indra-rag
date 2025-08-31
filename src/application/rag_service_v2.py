# src/application/rag_service_v2.py - VERSIÓN CORREGIDA COMPLETA
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import base64

from src.infrastructure.vector_store.qdrant_store_optimized import QdrantOptimizedStore

load_dotenv()


class RAGServiceV2:
    def __init__(self):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.embed_model = 'models/text-embedding-004'
        self.llm_model = genai.GenerativeModel('gemini-2.0-flash-exp')

        # Usar Qdrant Optimizado
        self.vector_store = QdrantOptimizedStore()

        # Verificar conexión - FORMA CORRECTA
        try:
            stats = self.vector_store.get_statistics()
            if 'error' in stats:
                raise Exception(f"Error en Qdrant: {stats['error']}")
            print(f"✅ RAG Service V2 con Qdrant Optimizado")
            print(f"   - Colección: {self.vector_store.collection_name}")
            print(f"   - Vectores: {stats.get('total_vectors', 'N/A')}")
        except Exception as e:
            raise Exception(f"❌ No se puede conectar a Qdrant: {e}")

        # Cargar metadata adicional
        self.document_analysis = self._load_document_analysis()
        self.images_metadata = self._load_images_metadata()

    def _load_document_analysis(self) -> Dict:
        try:
            with open('output/complete_document_analysis.json', 'r') as f:
                data = json.load(f)
                return data.get('complete_analysis', {})
        except:
            return {}

    def _load_images_metadata(self) -> List[Dict]:
        try:
            with open('output/images_with_context.json', 'r') as f:
                return json.load(f)
        except:
            return []

    def query(self, question: str, top_k: int = 3) -> Dict:
        """Query mejorado usando Qdrant Optimizado"""

        print(f"\n🤔 Pregunta: {question}")

        # Generar embedding de la pregunta
        query_result = genai.embed_content(
            model=self.embed_model,
            content=question,
            task_type="retrieval_query"
        )

        # Detectar si necesita imágenes
        image_keywords = ['diagrama', 'arquitectura', 'imagen', 'foto', 'muestra', 'visualiza']
        query_lower = question.lower()
        wants_image = any(keyword in query_lower for keyword in image_keywords)

        # USAR BÚSQUEDA HÍBRIDA OPTIMIZADA
        print("🔍 Búsqueda híbrida en Qdrant Optimizado...")

        filters = {"has_image": True} if wants_image else None

        relevant_chunks = self.vector_store.hybrid_search(
            query_embedding=query_result['embedding'],
            query_text=question,  # Para búsqueda de texto
            top_k=top_k,
            filters=filters
        )

        print(f"📊 Encontrados {len(relevant_chunks)} chunks relevantes")
        scores = ['{:.3f}'.format(r['score']) for r in relevant_chunks]
        print(f"   Scores híbridos: {scores}")

        # Buscar imágenes relevantes
        relevant_images = self.find_relevant_images(question, relevant_chunks)

        # Generar respuesta
        answer = self.generate_answer(question, relevant_chunks, relevant_images)

        # Preparar imágenes
        images_data = self._prepare_images(relevant_images)

        return {
            'question': question,
            'answer': answer,
            'sources': [f"Chunk {chunk['metadata']['chunk_id']}" for chunk in relevant_chunks],
            'images': images_data,
            'confidence': relevant_chunks[0]['score'] if relevant_chunks else 0.0,
            'chunks_used': len(relevant_chunks),
            'search_type': 'hybrid_optimized'
        }

    def find_relevant_images(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Encuentra imágenes relevantes"""
        relevant_images = []

        # Detectar si piden imágenes
        image_keywords = ['diagrama', 'arquitectura', 'imagen', 'foto', 'muestra', 'visualiza']
        query_lower = query.lower()
        wants_image = any(keyword in query_lower for keyword in image_keywords)

        if not wants_image:
            return []

        # Buscar imágenes por página - AJUSTADO PARA HYBRID SEARCH
        relevant_pages = set()
        for chunk in chunks:
            page = chunk.get('metadata', {}).get('page', 0)
            relevant_pages.add(page)

        for img in self.images_metadata:
            if img.get('page', 0) in relevant_pages:
                relevant_images.append(img)
                if len(relevant_images) >= 2:
                    break

        return relevant_images

    def generate_answer(self, query: str, chunks: List[Dict], images: List[Dict]) -> str:
        """Genera respuesta con contexto optimizado"""

        if not chunks:
            return "No encontré información relevante en el documento."

        # Contexto con scores híbridos
        context = "\n\n---\n\n".join([
            f"[Score híbrido: {chunk['score']:.2%}]\n{chunk['text']}"
            for chunk in chunks
        ])

        # Mencionar imágenes si existen
        image_context = ""
        if images:
            image_context = "\n\nImágenes relacionadas encontradas:\n"
            for img in images:
                image_context += f"- {img.get('description', 'Imagen')} (página {img.get('page', '?')})\n"

        prompt = f"""
        Eres un asistente experto analizando el documento AWS GenAI IDP Accelerator.

        CONTEXTO RECUPERADO (búsqueda híbrida):
        {context}
        {image_context}

        PREGUNTA: {query}

        INSTRUCCIONES:
        - Responde basándote SOLO en el contexto
        - Si hay imágenes relevantes, menciónalas
        - Sé específico y preciso
        - Si no tienes certeza, dilo

        RESPUESTA:
        """

        response = self.llm_model.generate_content(prompt)
        return response.text

    def _prepare_images(self, images: List[Dict]) -> List[Dict]:
        """Prepara imágenes para respuesta"""
        images_data = []
        for img in images:
            try:
                path = img.get('path', '')
                if os.path.exists(path):
                    with open(path, 'rb') as f:
                        img_base64 = base64.b64encode(f.read()).decode('utf-8')
                        images_data.append({
                            'filename': img.get('filename', ''),
                            'data': img_base64,
                            'description': img.get('description', ''),
                            'page': img.get('page', 0)
                        })
            except Exception as e:
                print(f"⚠️ Error cargando imagen: {e}")
        return images_data

    def get_stats(self) -> Dict:
        """Estadísticas del sistema"""
        stats = self.vector_store.get_statistics()

        return {
            'vector_db': 'Qdrant',
            'collection': self.vector_store.collection_name,
            'stats': stats if 'error' not in stats else None,
            'model': 'gemini-2.0-flash-exp',
            'embedding_model': 'text-embedding-004',
            'search_type': 'hybrid_optimized'
        }