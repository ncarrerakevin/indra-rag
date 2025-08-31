# scripts/fix_image_metadata.py
"""
Script para arreglar la metadata de chunks en Qdrant
Conecta chunks con imágenes por número de página
"""

import json
import sys
import os
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Path fix
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def fix_metadata():
    """Arregla metadata de chunks para conectar con imágenes"""

    print("🔧 INICIANDO CORRECCIÓN DE METADATA")
    print("=" * 50)

    # 1. Conectar a Qdrant
    client = QdrantClient(host="localhost", port=6333)
    collection_name = "indra_rag_optimized"

    # 2. Cargar datos necesarios
    print("\n📁 Cargando archivos...")

    # Chunks originales con texto completo
    with open('output/chunks.json', 'r') as f:
        original_chunks = json.load(f)
    print(f"   ✓ {len(original_chunks)} chunks originales cargados")

    # Análisis del documento (tiene páginas)
    try:
        with open('output/complete_document_analysis.json', 'r') as f:
            doc_analysis = json.load(f)
            pages_data = doc_analysis.get('complete_analysis', {}).get('pages', [])
    except:
        pages_data = []
    print(f"   ✓ {len(pages_data)} páginas analizadas")

    # Imágenes con sus páginas
    with open('output/images_with_context.json', 'r') as f:
        images_data = json.load(f)

    # Crear mapa de páginas con imágenes
    pages_with_images = set()
    image_by_page = {}
    for img in images_data:
        page = img.get('page', 0)
        pages_with_images.add(page)
        if page not in image_by_page:
            image_by_page[page] = []
        image_by_page[page].append(img)

    print(f"   ✓ Imágenes en páginas: {sorted(pages_with_images)}")

    # 3. Mapear chunks a páginas
    print("\n🔍 Mapeando chunks a páginas...")

    # Estrategia: usar el contenido para determinar página
    chunk_to_page = {}

    # Si tenemos análisis por página
    if pages_data:
        for chunk_idx, chunk in enumerate(original_chunks):
            chunk_text = chunk['content'].lower()[:200]  # Primeros 200 chars

            # Buscar en qué página aparece este texto
            best_page = 1
            best_match = 0

            for page_data in pages_data:
                page_num = page_data.get('page_number', 1)
                page_text = page_data.get('text_content', '').lower()

                # Contar palabras coincidentes
                chunk_words = chunk_text.split()[:10]  # Primeras 10 palabras
                matches = sum(1 for word in chunk_words if word in page_text)

                if matches > best_match:
                    best_match = matches
                    best_page = page_num

            chunk_to_page[chunk_idx] = best_page
            if best_page in pages_with_images:
                print(f"   ✓ Chunk {chunk_idx} → Página {best_page} (¡CON IMAGEN!)")

    # Fallback: distribuir chunks uniformemente
    if not chunk_to_page:
        print("   ⚠️ Sin análisis por página, usando distribución estimada")
        total_chunks = len(original_chunks)
        total_pages = 11  # Sabemos que el PDF tiene 11 páginas
        chunks_per_page = total_chunks // total_pages

        for i, chunk in enumerate(original_chunks):
            estimated_page = min((i // chunks_per_page) + 1, total_pages)
            chunk_to_page[i] = estimated_page
            if estimated_page in pages_with_images:
                print(f"   ✓ Chunk {i} → Página {estimated_page} (¡CON IMAGEN!)")

    # 4. Obtener todos los puntos de Qdrant
    print("\n📥 Obteniendo puntos de Qdrant...")

    # Scroll para obtener TODOS los puntos
    all_points = []
    offset = None

    while True:
        result = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        points, next_offset = result
        all_points.extend(points)

        if next_offset is None:
            break
        offset = next_offset

    print(f"   ✓ {len(all_points)} puntos obtenidos")

    # 5. Actualizar metadata de cada punto
    print("\n🔄 Actualizando metadata...")

    updated_count = 0
    image_chunks_count = 0

    for point in all_points:
        point_id = point.id
        payload = point.payload

        # Obtener chunk_id original
        chunk_id = payload.get('chunk_id', 0)

        # Determinar página
        page = chunk_to_page.get(chunk_id, 1)

        # Determinar si tiene imagen
        has_image = page in pages_with_images

        # Preparar payload actualizado
        updated_payload = {
            **payload,  # Mantener todo lo existente
            'page': page,
            'has_image': has_image,
            'pages_with_images': list(pages_with_images) if has_image else []
        }

        # Si tiene imagen, agregar paths
        if has_image:
            images_in_page = image_by_page.get(page, [])
            if images_in_page:
                updated_payload['image_paths'] = [img['path'] for img in images_in_page]
                updated_payload['image_descriptions'] = [img.get('description', '') for img in images_in_page]
                image_chunks_count += 1

        # Actualizar en Qdrant
        client.set_payload(
            collection_name=collection_name,
            payload=updated_payload,
            points=[point_id]
        )

        updated_count += 1
        if updated_count % 5 == 0:
            print(f"   ✓ Actualizados {updated_count}/{len(all_points)} puntos...")

    print(f"\n✅ CORRECCIÓN COMPLETADA")
    print(f"   - Total puntos actualizados: {updated_count}")
    print(f"   - Chunks con imágenes: {image_chunks_count}")
    print(f"   - Páginas con imágenes: {sorted(pages_with_images)}")

    # 6. Verificar corrección
    print("\n🔍 Verificando corrección...")

    # Buscar chunks con imágenes
    results = client.scroll(
        collection_name=collection_name,
        scroll_filter=Filter(
            must=[
                FieldCondition(
                    key="has_image",
                    match=MatchValue(value=True)
                )
            ]
        ),
        limit=100
    )

    points_with_images, _ = results
    print(f"   ✓ Chunks marcados con has_image=True: {len(points_with_images)}")

    # Mostrar ejemplos
    if points_with_images:
        print("\n📸 Ejemplos de chunks con imágenes:")
        for point in points_with_images[:3]:
            print(f"   - Chunk {point.payload.get('chunk_id')} en página {point.payload.get('page')}")
            print(f"     Texto: {point.payload.get('text', '')[:100]}...")
            if point.payload.get('image_paths'):
                print(f"     Imágenes: {point.payload.get('image_paths')}")

    print("\n✨ ¡Metadata corregida! Ahora las búsquedas de imágenes funcionarán.")
    print("🚀 Ejecuta 'python scripts/test_full_system.py' para verificar")


if __name__ == "__main__":
    fix_metadata()