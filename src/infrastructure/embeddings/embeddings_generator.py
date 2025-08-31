# src/infrastructure/embeddings/embeddings_generator.py
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
import numpy as np

load_dotenv()


class EmbeddingsGenerator:
    def __init__(self):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        # Modelo de embeddings de Google
        self.model = 'models/text-embedding-004'

    def generate_embeddings(self):
        """Genera embeddings para cada chunk"""

        # Cargar chunks
        with open('output/chunks.json', 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"📊 Generando embeddings para {len(chunks)} chunks...")

        # Generar embedding para cada chunk
        for i, chunk in enumerate(chunks):
            text = chunk['content']

            # Generar embedding
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )

            # Agregar embedding al chunk
            chunk['embedding'] = result['embedding']

            print(f"✅ Chunk {i + 1}/{len(chunks)} - Vector de {len(result['embedding'])} dimensiones")

        # Guardar chunks con embeddings
        with open('output/chunks_with_embeddings.json', 'w', encoding='utf-8') as f:
            json.dump(chunks, f, indent=2)

        print(f"\n✅ Embeddings generados y guardados")
        print(f"📁 Archivo: output/chunks_with_embeddings.json")

        return chunks


# Test
if __name__ == "__main__":
    generator = EmbeddingsGenerator()
    chunks_with_embeddings = generator.generate_embeddings()

    # Verificar
    print(f"\n🔍 Verificación:")
    print(f"  - Chunks con embeddings: {len(chunks_with_embeddings)}")
    print(f"  - Dimensiones del vector: {len(chunks_with_embeddings[0]['embedding'])}")