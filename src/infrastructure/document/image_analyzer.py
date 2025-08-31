# src/infrastructure/document/image_analyzer.py
import google.generativeai as genai
from PIL import Image
import json
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()


class ImageAnalyzer:
    def __init__(self):
        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        self.model = genai.GenerativeModel('gemini-1.5-flash')

    def analyze_images(self) -> list:
        """Analiza cada imagen y obtiene su contexto/descripción"""

        # Cargar metadata de imágenes
        with open('output/images_metadata.json', 'r') as f:
            images_metadata = json.load(f)

        analyzed_images = []

        print("🔍 Analizando imágenes con Gemini...")

        for img_info in images_metadata:
            img_path = img_info['path']
            print(f"\n📸 Analizando: {img_info['filename']}")

            # Cargar imagen
            img = Image.open(img_path)

            # Prompt para Gemini
            prompt = """
            Describe esta imagen de forma detallada:
            1. ¿Qué tipo de contenido es? (diagrama, foto de persona, tabla, screenshot, etc.)
            2. Si es un diagrama: ¿qué arquitectura o proceso muestra?
            3. Si es una persona: describe características visibles
            4. Si es una tabla: ¿qué información contiene?
            5. Cualquier texto visible en la imagen

            Sé específico y conciso.
            """

            try:
                response = self.model.generate_content([img, prompt])
                description = response.text
                print(f"  ✅ Analizada")
            except Exception as e:
                description = f"Error al analizar: {str(e)}"
                print(f"  ❌ Error: {e}")

            # Agregar análisis
            analyzed_images.append({
                **img_info,  # Mantener info original
                "description": description,
                "analyzed": True
            })

        # Guardar análisis
        output_path = 'output/images_with_context.json'
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(analyzed_images, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Análisis completo guardado en: {output_path}")

        # Mostrar resumen
        self.print_summary(analyzed_images)

        return analyzed_images

    def print_summary(self, analyzed_images):
        """Imprime un resumen del análisis"""
        print("\n📊 RESUMEN DE IMÁGENES:")
        print("-" * 50)

        for img in analyzed_images[:3]:  # Mostrar solo las primeras 3
            print(f"\n📄 {img['filename']} (Página {img['page']}):")
            print(f"   {img['description'][:200]}...")

        print("\n(...)")


# Test
def test_image_analysis():
    analyzer = ImageAnalyzer()
    results = analyzer.analyze_images()

    print(f"\n🎯 Total imágenes analizadas: {len(results)}")
    print("📁 Revisa output/images_with_context.json para ver todas las descripciones")

    return results


if __name__ == "__main__":
    test_image_analysis()