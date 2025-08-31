# scripts/test_full_system.py
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import json
import time


def test_full_system():
    """Prueba completa del sistema RAG optimizado"""

    print("🚀 TEST COMPLETO DEL SISTEMA RAG OPTIMIZADO")
    print("=" * 50)

    # 1. Verificar que el API esté corriendo
    print("\n1️⃣ Verificando API...")
    try:
        response = requests.get("http://localhost:8000/")
        if response.status_code == 200:
            print("   ✅ API funcionando")
            data = response.json()
            print(f"   - Modelo: {data.get('model', 'N/A')}")
            print(f"   - Chunks: {data.get('chunks_loaded', 'N/A')}")
        else:
            print("   ❌ API no responde correctamente")
            return
    except:
        print("   ❌ API no está corriendo. Ejecuta: python src/api/main.py")
        return

    # 2. Verificar estadísticas
    print("\n2️⃣ Verificando Qdrant optimizado...")
    try:
        response = requests.get("http://localhost:8000/stats")
        if response.status_code == 200:
            stats = response.json()
            print("   ✅ Qdrant conectado")
            print(f"   - Vector DB: {stats.get('vector_db', 'N/A')}")
            print(f"   - Colección: {stats.get('collection', 'N/A')}")
            if stats.get('stats'):
                print(f"   - Vectores: {stats['stats'].get('vectors_count', 'N/A')}")
        else:
            print("   ⚠️ Stats no disponibles")
    except Exception as e:
        print(f"   ⚠️ Error obteniendo stats: {e}")

    # 3. Pruebas de queries
    print("\n3️⃣ Probando queries...")

    test_cases = [
        {
            "query": "¿Quiénes son los autores del documento?",
            "expects_images": False,
            "should_find": ["Bob Strahan", "Joe King", "Mofijul Islam"]
        },
        {
            "query": "¿Qué es Competiscan y qué logró?",
            "expects_images": False,
            "should_find": ["35,000", "45,000", "campañas", "marketing"]
        },
        {
            "query": "Muestra el diagrama de arquitectura de la solución",
            "expects_images": True,
            "should_find": ["Bedrock", "Lambda", "DynamoDB"]
        },
        {
            "query": "¿Cuántos documentos procesó Ricoh?",
            "expects_images": False,
            "should_find": ["10,000", "70,000", "healthcare"]
        }
    ]

    results = []

    for i, test in enumerate(test_cases, 1):
        print(f"\n   Test {i}: {test['query'][:50]}...")

        try:
            response = requests.post(
                "http://localhost:8000/query",
                json={"question": test['query'], "top_k": 3}
            )

            if response.status_code == 200:
                data = response.json()

                # Verificar respuesta
                answer = data.get('answer', '')
                sources = data.get('sources', [])
                images = data.get('images', [])
                confidence = data.get('confidence', 0)

                # Verificar contenido esperado
                found_keywords = [kw for kw in test['should_find'] if kw.lower() in answer.lower()]

                success = len(found_keywords) > 0
                has_images = len(images) > 0

                result = {
                    'query': test['query'],
                    'success': success,
                    'confidence': confidence,
                    'found_keywords': found_keywords,
                    'has_images': has_images,
                    'expected_images': test['expects_images']
                }

                results.append(result)

                # Mostrar resultado
                if success:
                    print(f"      ✅ Respuesta correcta (confianza: {confidence:.2%})")
                    print(f"      📝 Encontró: {', '.join(found_keywords)}")
                else:
                    print(f"      ⚠️ No encontró keywords esperados")

                if test['expects_images']:
                    if has_images:
                        print(f"      🖼️ {len(images)} imagen(es) encontrada(s)")
                    else:
                        print(f"      ❌ No encontró imágenes (esperadas)")

                print(f"      📚 Fuentes: {', '.join(sources[:3])}")

            else:
                print(f"      ❌ Error: {response.status_code}")
                results.append({'query': test['query'], 'success': False})

        except Exception as e:
            print(f"      ❌ Error: {e}")
            results.append({'query': test['query'], 'success': False})

    # 4. Resumen
    print("\n" + "=" * 50)
    print("📊 RESUMEN DE RESULTADOS")
    print("=" * 50)

    total = len(results)
    successful = sum(1 for r in results if r.get('success', False))
    avg_confidence = sum(r.get('confidence', 0) for r in results) / total if total > 0 else 0

    print(f"\n✅ Exitosos: {successful}/{total} ({successful / total * 100:.0f}%)")
    print(f"📊 Confianza promedio: {avg_confidence:.2%}")

    # Problemas detectados
    problems = []
    for r in results:
        if not r.get('success'):
            problems.append(f"- Query fallida: {r['query'][:50]}...")
        if r.get('expected_images') and not r.get('has_images'):
            problems.append(f"- Sin imágenes en: {r['query'][:50]}...")

    if problems:
        print(f"\n⚠️ Problemas detectados:")
        for p in problems:
            print(p)
    else:
        print(f"\n🎉 ¡TODO FUNCIONANDO PERFECTAMENTE!")

    return successful == total


if __name__ == "__main__":
    # Primero asegúrate que el API esté corriendo
    print("⚠️ Asegúrate que el API esté corriendo:")
    print("   python src/api/main.py")
    print("\nPresiona Enter cuando esté listo...")
    input()

    success = test_full_system()

    if success:
        print("\n✅ SISTEMA LISTO PARA PRODUCCIÓN")
    else:
        print("\n⚠️ HAY PROBLEMAS QUE RESOLVER")