# src/infrastructure/document/image_extractor.py
import fitz  # PyMuPDF
from pathlib import Path
import json


class ImageExtractor:
    def extract_images_from_pdf(self, pdf_path: str) -> dict:
        """Extrae todas las imágenes del PDF"""

        # Crear carpeta para imágenes
        Path("output/images").mkdir(parents=True, exist_ok=True)

        doc = fitz.open(pdf_path)
        images_info = []

        print(f"📄 Procesando PDF: {pdf_path}")
        print(f"📑 Total de páginas: {len(doc)}")

        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()

            if image_list:
                print(f"\n📃 Página {page_num + 1}: {len(image_list)} imágenes encontradas")

            for img_index, img in enumerate(image_list):
                xref = img[0]
                pix = fitz.Pixmap(doc, xref)

                # Convertir CMYK a RGB si es necesario
                if pix.n - pix.alpha > 3:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                # Guardar imagen
                img_name = f"page_{page_num + 1}_img_{img_index + 1}.png"
                img_path = f"output/images/{img_name}"

                pix.save(img_path)

                images_info.append({
                    "page": page_num + 1,
                    "image_index": img_index + 1,
                    "filename": img_name,
                    "path": img_path,
                    "width": pix.width,
                    "height": pix.height
                })

                print(f"  ✅ Extraída: {img_name} ({pix.width}x{pix.height}px)")
                pix = None  # Liberar memoria

        doc.close()

        # Guardar metadata de las imágenes
        metadata_path = 'output/images_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(images_info, f, indent=2)

        print(f"\n📊 Resumen:")
        print(f"  - Total imágenes extraídas: {len(images_info)}")
        print(f"  - Guardadas en: output/images/")
        print(f"  - Metadata en: {metadata_path}")

        return images_info


# Test
def test_image_extraction():
    extractor = ImageExtractor()
    images = extractor.extract_images_from_pdf("data/rag-challenge.pdf")

    if images:
        print("\n🎯 Verificación:")
        print("  Abre la carpeta output/images/ para ver las imágenes extraídas")
        print("  Revisa output/images_metadata.json para ver la información")
    else:
        print("\n⚠️ No se encontraron imágenes en el PDF")

    return images


if __name__ == "__main__":
    test_image_extraction()