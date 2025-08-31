# src/infrastructure/document/pdf_processor.py
import google.generativeai as genai
from pathlib import Path
import json
import os
from dotenv import load_dotenv

load_dotenv()


class PDFProcessor:
    def __init__(self):
        # Configurar Gemini
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def extract_text_from_pdf(self, pdf_path: str) -> dict:
        """Extrae texto del PDF usando Gemini Vision - igual que el chat"""

        print(f"📄 Procesando: {pdf_path}")

        # Subir el archivo a Gemini (como cuando lo arrastras al chat)
        uploaded_file = genai.upload_file(pdf_path)
        print(f"✅ Archivo subido a Gemini")

        # Prompt para extraer TODO el contenido
        prompt = """
        Extrae TODO el texto de este documento PDF.
        Incluye TODOS los títulos, párrafos, listas, tablas y cualquier texto visible.
        Mantén la estructura y formato original lo más posible.
        """

        # Generar respuesta
        response = self.model.generate_content([uploaded_file, prompt])

        return {
            "full_text": response.text,
            "source_file": pdf_path
        }


# Test
def test_extraction():
    processor = PDFProcessor()
    result = processor.extract_text_from_pdf("data/rag-challenge.pdf")

    # Guardar resultado
    output_path = Path("output/extracted_text.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Texto extraído: {len(result['full_text'])} caracteres")
    print(f"📁 Guardado en: {output_path}")

    return result


if __name__ == "__main__":
    test_extraction()