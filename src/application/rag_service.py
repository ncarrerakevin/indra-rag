# src/application/rag_service.py
import google.generativeai as genai
import json
import numpy as np
import os
from dotenv import load_dotenv
from typing import List, Dict, Optional
import base64

load_dotenv()


class RAGService:
    def __init__(self):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.embed_model = 'models/text-embedding-004'
        self.llm_model = genai.GenerativeModel('gemini-2.5-pro')

        # Cargar datos
        self.chunks = self._load_chunks()
        self.document_analysis = self._load_document_analysis()
        self.images_metadata = self._load_images_metadata()

        print(f"✅ RAG Service inicializado")
        print(f"   - Chunks: {len(self.chunks)}")
        print(f"   - Personas: {len(self.document_analysis.get('people', []))}")
        print(f"   - Diagramas: {len(self.document_analysis.get('diagrams', []))}")

    def _load_chunks(self) -> List[Dict]:
        with open('output/chunks_with_embeddings.json', 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_document_analysis(self) -> Dict:
        with open('output/complete_document_analysis.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return data.get('complete_analysis', {})

    def _load_images_metadata(self) -> List[Dict]:
        try:
            with open('output/images_with_context.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []

    def find_similar_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """Encuentra chunks similares"""
        query_result = genai.embed_content(
            model=self.embed_model,
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = np.array(query_result['embedding'])

        similarities = []
        for chunk in self.chunks:
            chunk_embedding = np.array(chunk['embedding'])
            similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append((chunk, similarity))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in similarities[:top_k]]

    def find_relevant_images(self, query: str, chunks: List[Dict]) -> List[Dict]:
        """Encuentra imágenes relevantes basadas en la pregunta y chunks"""
        relevant_images = []

        # Palabras clave para detectar solicitudes de imágenes
        image_keywords = ['diagrama', 'arquitectura', 'imagen', 'foto', 'muestra', 'visualiza']
        query_lower = query.lower()

        # Si piden específicamente imágenes
        if any(keyword in query_lower for keyword in image_keywords):
            # Buscar en el análisis del documento
            if 'arquitectura' in query_lower or 'diagrama' in query_lower:
                for diagram in self.document_analysis.get('diagrams', []):
                    # Encontrar la imagen correspondiente
                    page = diagram['page']
                    for img in self.images_metadata:
                        if img['page'] == page:
                            relevant_images.append({
                                'filename': img['filename'],
                                'path': img['path'],
                                'description': diagram['description'],
                                'type': 'diagram'
                            })
                            break

            # Si piden personas
            if 'quien' in query_lower or 'autor' in query_lower:
                for person in self.document_analysis.get('people', []):
                    if person.get('image_description'):
                        page = person['page']
                        for img in self.images_metadata:
                            if img['page'] == page and 'person' in img.get('description', '').lower():
                                relevant_images.append({
                                    'filename': img['filename'],
                                    'path': img['path'],
                                    'description': person['image_description'],
                                    'person_name': person['name'],
                                    'type': 'person'
                                })

        return relevant_images[:2]  # Máximo 2 imágenes

    def generate_answer(self, query: str, chunks: List[Dict], images: List[Dict]) -> str:
        """Genera respuesta con contexto e imágenes"""
        context = "\n\n---\n\n".join([
            f"[Fuente {i + 1}]:\n{chunk['content']}"
            for i, chunk in enumerate(chunks)
        ])

        # Si hay imágenes relevantes, mencionarlas
        image_context = ""
        if images:
            image_context = "\n\nImágenes relevantes disponibles:\n"
            for img in images:
                image_context += f"- {img['description']}\n"

        prompt = f"""
        Eres un asistente experto analizando un documento sobre AWS GenAI IDP Accelerator.
        Usa el contexto para responder de forma precisa y completa.

        CONTEXTO:
        {context}
        {image_context}

        PREGUNTA: {query}

        INSTRUCCIONES:
        - Responde basándote en el contexto proporcionado
        - Si hay imágenes relevantes, menciónalas en tu respuesta
        - Sé específico y detallado
        - Si la información no está en el contexto, dilo claramente

        RESPUESTA:
        """

        response = self.llm_model.generate_content(prompt)
        return response.text

    def query(self, question: str, top_k: int = 3) -> Dict:
        """Pipeline completo de RAG multimodal"""

        # 1. Buscar chunks relevantes
        relevant_chunks = self.find_similar_chunks(question, top_k)

        # 2. Buscar imágenes relevantes
        relevant_images = self.find_relevant_images(question, relevant_chunks)

        # 3. Generar respuesta
        answer = self.generate_answer(question, relevant_chunks, relevant_images)

        # 4. Preparar imágenes para respuesta
        images_data = []
        for img in relevant_images:
            try:
                with open(img['path'], 'rb') as f:
                    img_base64 = base64.b64encode(f.read()).decode('utf-8')
                    images_data.append({
                        'filename': img['filename'],
                        'data': img_base64,
                        'description': img['description'],
                        'type': img.get('type', 'unknown')
                    })
            except Exception as e:
                print(f"Error cargando imagen {img['path']}: {e}")

        return {
            'question': question,
            'answer': answer,
            'sources': [f"Chunk {chunk['id']}" for chunk in relevant_chunks],
            'images': images_data,
            'confidence': 0.95 if relevant_chunks else 0.5
        }

    def get_document_info(self) -> Dict:
        """Obtiene información del documento procesado"""
        people_names = [p['name'] for p in self.document_analysis.get('people', [])]
        return {
            'total_chunks': len(self.chunks),
            'total_pages': len(self.document_analysis.get('pages', [])),
            'people_identified': people_names,
            'diagrams_count': len(self.document_analysis.get('diagrams', []))
        }