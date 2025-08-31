# src/infrastructure/document/text_chunker.py
import json
from typing import List, Dict


class TextChunker:
    def __init__(self, chunk_size: int = 1000, overlap: int = 200):
        """
        chunk_size: caracteres por chunk
        overlap: caracteres que se solapan entre chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap

    def create_chunks(self, text: str) -> List[Dict]:
        """Divide el texto en chunks con metadata"""
        chunks = []

        # Dividir por saltos de línea para mantener párrafos completos
        lines = text.split('\n')
        current_chunk = ""
        chunk_id = 0

        for line in lines:
            # Si agregar esta línea excede el tamaño, crear nuevo chunk
            if len(current_chunk) + len(line) > self.chunk_size and current_chunk:
                chunks.append({
                    "id": chunk_id,
                    "content": current_chunk.strip(),
                    "char_count": len(current_chunk.strip())
                })

                # Overlap: incluir parte del chunk anterior
                overlap_text = current_chunk[-self.overlap:] if len(current_chunk) > self.overlap else current_chunk
                current_chunk = overlap_text + "\n" + line
                chunk_id += 1
            else:
                current_chunk += "\n" + line

        # Agregar el último chunk
        if current_chunk.strip():
            chunks.append({
                "id": chunk_id,
                "content": current_chunk.strip(),
                "char_count": len(current_chunk.strip())
            })

        return chunks


# Test
def test_chunking():
    # Cargar el texto extraído
    with open('output/extracted_text.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Crear chunks
    chunker = TextChunker(chunk_size=1000, overlap=200)
    chunks = chunker.create_chunks(data['full_text'])

    # Guardar chunks
    with open('output/chunks.json', 'w', encoding='utf-8') as f:
        json.dump(chunks, f, indent=2, ensure_ascii=False)

    # Mostrar estadísticas
    print(f"📊 Estadísticas de Chunking:")
    print(f"  - Total chunks: {len(chunks)}")
    print(f"  - Promedio caracteres/chunk: {sum(c['char_count'] for c in chunks) // len(chunks)}")
    print(f"  - Chunk más pequeño: {min(c['char_count'] for c in chunks)} chars")
    print(f"  - Chunk más grande: {max(c['char_count'] for c in chunks)} chars")
    print(f"\n📁 Chunks guardados en: output/chunks.json")

    # Mostrar ejemplo
    print(f"\n📝 Ejemplo del primer chunk:")
    print("-" * 50)
    print(chunks[0]['content'][:300] + "...")

    return chunks


if __name__ == "__main__":
    test_chunking()