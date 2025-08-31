# src/infrastructure/document/complete_processor.py
import google.generativeai as genai
import json
import os
from dotenv import load_dotenv
import re

load_dotenv()


class CompleteDocumentProcessor:
    def __init__(self):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-2.5-pro')

    def process_document_with_context(self, pdf_path: str) -> dict:
        """Procesa el PDF manteniendo TODAS las relaciones"""

        print(f"📄 Procesando documento completo: {pdf_path}")

        # Subir PDF
        uploaded_file = genai.upload_file(pdf_path)

        # Prompt más explícito sobre formato JSON
        prompt = """
        Analiza este documento PDF y devuelve ÚNICAMENTE un JSON válido (sin markdown, sin explicaciones).

        El JSON debe tener esta estructura:
        {
            "pages": [
                {
                    "page_number": 1,
                    "text_content": "texto completo de la página",
                    "images": [
                        {
                            "position": "top/middle/bottom",
                            "description": "descripción de la imagen",
                            "related_text": "texto relacionado si existe"
                        }
                    ]
                }
            ],
            "people": [
                {
                    "name": "nombre si se menciona",
                    "page": 1,
                    "role": "cargo/rol",
                    "image_description": "descripción visual"
                }
            ],
            "diagrams": [
                {
                    "page": 1,
                    "type": "architecture/flow/table",
                    "description": "qué muestra"
                }
            ]
        }

        IMPORTANTE: Responde SOLO con el JSON, sin ```json``` ni texto adicional.
        """

        response = self.model.generate_content([uploaded_file, prompt])

        # Limpiar la respuesta
        response_text = response.text.strip()

        # Remover markdown si existe
        if response_text.startswith("```"):
            response_text = re.sub(r'^```json?\n?', '', response_text)
            response_text = re.sub(r'\n?```$', '', response_text)

        # Intentar parsear JSON
        try:
            analysis = json.loads(response_text)
        except json.JSONDecodeError as e:
            print(f"⚠️ No se pudo parsear JSON. Guardando respuesta raw...")
            print(f"Error: {e}")
            print(f"\nPrimeros 500 caracteres de la respuesta:")
            print(response_text[:500])

            # Guardar respuesta raw para debug
            with open('output/raw_response.txt', 'w', encoding='utf-8') as f:
                f.write(response_text)

            # Crear estructura básica
            analysis = {
                "error": "No se pudo parsear como JSON",
                "raw_text": response_text
            }

        # Guardar resultado
        result = {
            "complete_analysis": analysis,
            "source": pdf_path
        }

        with open('output/complete_document_analysis.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print("✅ Análisis completo guardado")
        return result


# Test con manejo de errores
if __name__ == "__main__":
    try:
        processor = CompleteDocumentProcessor()
        result = processor.process_document_with_context("data/rag-challenge.pdf")

        if "error" in result["complete_analysis"]:
            print("\n⚠️ Hubo un problema con el formato JSON")
            print("Revisa output/raw_response.txt para ver la respuesta completa")
        else:
            print("\n✅ Documento procesado exitosamente")
            print(f"📊 Páginas analizadas: {len(result['complete_analysis'].get('pages', []))}")
            print(f"👥 Personas identificadas: {len(result['complete_analysis'].get('people', []))}")
            print(f"📐 Diagramas encontrados: {len(result['complete_analysis'].get('diagrams', []))}")

    except Exception as e:
        print(f"❌ Error: {e}")