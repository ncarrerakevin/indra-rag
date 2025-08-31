# src/application/rag_pipeline.py
import google.generativeai as genai
import json
import numpy as np
import os
from dotenv import load_dotenv
from typing import List, Dict

load_dotenv()


class RAGPipeline:
    def __init__(self):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.embed_model = 'models/text-embedding-004'
        self.llm_model = genai.GenerativeModel('gemini-2.5-pro')

        # Cargar chunks con embeddings
        with open('output/chunks_with_embeddings.json', 'r', encoding='utf-8') as f:
            self.chunks = json.load(f)
        print(f"✅ Cargados {len(self.chunks)} chunks con embeddings")

    def find_similar_chunks(self, query: str, top_k: int = 5) -> List[Dict]:
        """Encuentra los chunks más similares a la pregunta"""

        # Generar embedding de la pregunta
        query_result = genai.embed_content(
            model=self.embed_model,
            content=query,
            task_type="retrieval_query"
        )
        query_embedding = np.array(query_result['embedding'])

        # Calcular similitud con cada chunk
        similarities = []
        for chunk in self.chunks:
            chunk_embedding = np.array(chunk['embedding'])

            # Similitud coseno
            similarity = np.dot(query_embedding, chunk_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
            )
            similarities.append((chunk, similarity))

        # Ordenar por similitud y tomar top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_chunks = [chunk for chunk, _ in similarities[:top_k]]

        print(
            f"📊 Top {top_k} chunks encontrados (similitud: {similarities[0][1]:.3f} - {similarities[top_k - 1][1]:.3f})")

        return top_chunks

    def generate_answer(self, query: str, chunks: List[Dict]) -> str:
        """Genera respuesta usando los chunks relevantes"""

        # Construir contexto
        context = "\n\n---\n\n".join([
            f"[Chunk {i + 1}]:\n{chunk['content']}"
            for i, chunk in enumerate(chunks)
        ])

        # Prompt para el LLM
        prompt = f"""
        Eres un asistente experto. Usa el siguiente contexto para responder la pregunta.
        Si la respuesta no está en el contexto, dilo claramente.

        CONTEXTO:
        {context}

        PREGUNTA: {query}

        RESPUESTA:
        """

        # Generar respuesta
        response = self.llm_model.generate_content(prompt)
        return response.text

    def query(self, question: str) -> Dict:
        """Pipeline completo de RAG"""
        print(f"\n🤔 Pregunta: {question}")

        # 1. Buscar chunks relevantes
        print("🔍 Buscando chunks relevantes...")
        relevant_chunks = self.find_similar_chunks(question, top_k=3)

        # 2. Generar respuesta
        print("💭 Generando respuesta...")
        answer = self.generate_answer(question, relevant_chunks)

        return {
            "question": question,
            "answer": answer,
            "sources": [f"Chunk {chunk['id']}" for chunk in relevant_chunks]
        }


# Test del RAG completo
if __name__ == "__main__":
    rag = RAGPipeline()

    # Preguntas de prueba
    test_questions = [
        "¿Quiénes son los autores del documento?",
        "¿Qué es Competiscan y qué logró?",
        "¿Cuál es la arquitectura de la solución?"
    ]

    for question in test_questions:
        result = rag.query(question)
        print(f"\n{'=' * 60}")
        print(f"❓ {result['question']}")
        print(f"\n📝 Respuesta:\n{result['answer']}")
        print(f"\n📚 Fuentes: {', '.join(result['sources'])}")