# src/ui/ui_with_images.py
import gradio as gr
import requests
import base64
from PIL import Image
import io


def query_and_show_images(question):
    """Consulta al backend y devuelve texto + imágenes"""
    try:
        response = requests.post(
            "http://localhost:8000/query",
            json={"question": question}
        )

        if response.status_code == 200:
            data = response.json()

            # Preparar texto
            text_output = f"### Respuesta:\n{data['answer']}\n\n"
            text_output += f"**Fuentes:** {', '.join(data['sources'])}\n"

            # Procesar imágenes si existen
            images = []
            if data.get('images'):
                for img_data in data['images']:
                    try:
                        # Decodificar base64 a imagen
                        img_bytes = base64.b64decode(img_data['data'])
                        img = Image.open(io.BytesIO(img_bytes))
                        images.append(img)
                    except Exception as e:
                        print(f"Error procesando imagen: {e}")

            # Devolver texto y primera imagen (si existe)
            if images:
                return text_output, images[0]
            else:
                # Crear imagen placeholder si no hay imágenes
                placeholder = Image.new('RGB', (400, 100), color='white')
                return text_output, placeholder

        else:
            return f"Error: {response.status_code}", None

    except Exception as e:
        return f"Error: {str(e)}", None


# Interfaz con imagen
with gr.Blocks(title="RAG Multimodal") as demo:
    gr.Markdown("# 🤖 RAG Multimodal - Reto Indra")
    gr.Markdown("Sistema de Q&A con soporte de imágenes del documento AWS GenAI IDP")

    with gr.Row():
        with gr.Column():
            input_text = gr.Textbox(
                label="Pregunta",
                placeholder="Ej: Muestra el diagrama de arquitectura",
                lines=2
            )
            submit = gr.Button("🔍 Buscar", variant="primary")

            gr.Examples(
                examples=[
                    "Muestra el diagrama de arquitectura de la solución",
                    "¿Quiénes son los autores del documento?",
                    "¿Qué es Competiscan?",
                    "Explica el flujo de procesamiento con su diagrama"
                ],
                inputs=input_text
            )

    with gr.Row():
        with gr.Column():
            output_text = gr.Textbox(
                label="Respuesta",
                lines=10,
                max_lines=20
            )
            output_image = gr.Image(
                label="Imagen relacionada (si existe)",
                type="pil"
            )

    submit.click(
        query_and_show_images,
        inputs=input_text,
        outputs=[output_text, output_image]
    )

if __name__ == "__main__":
    demo.launch(share=True)