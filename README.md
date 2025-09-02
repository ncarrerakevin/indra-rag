# RAG Multimodal - Reto Indra

Chatbot RAG con procesamiento multimodal de documentos PDF usando FastAPI, Gradio, Qdrant y Google Gemini.

## Demo

🔗 **URL**: https://0162b6758c51a8238d.gradio.live

## Stack Tecnológico

- **Backend**: FastAPI
- **Frontend**: Gradio
- **Vector DB**: Qdrant
- **LLM**: Google Gemini
- **Embeddings**: text-embedding-004

## Instalación Rápida

### 1. Requisitos
```bash
Python 3.9+
Docker
API Key de Google Gemini
```

### 2. Setup
```bash
# Clonar repo
git clone https://github.com/tu-usuario/indra-rag.git
cd indra-rag

# Instalar dependencias
pip install -r requirements.txt

# Configurar API Key
echo "GEMINI_API_KEY=tu_api_key" > .env

# Iniciar Qdrant
docker run -p 6333:6333 qdrant/qdrant
```

### 3. Procesar Documento
```bash
# Ejecutar scripts de procesamiento
python src/infrastructure/document/pdf_processor.py
python src/infrastructure/document/text_chunker.py
python src/infrastructure/embeddings/embeddings_generator.py
python scripts/load_to_qdrant.py
```

### 4. Iniciar Sistema
```bash
# Terminal 1 - API
python src/api/main.py

# Terminal 2 - UI
python src/ui/ui_with_images.py
```

## Endpoints API

| Método | Endpoint | Descripción |
|--------|----------|-------------|
| GET | `/` | Health check |
| POST | `/query` | Consulta RAG |
| GET | `/document-info` | Info del documento |
| GET | `/stats` | Estadísticas del sistema |

## Ejemplos de Uso

### Consulta via API
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "¿Quiénes son los autores?", "top_k": 3}'
```

### Preguntas de Prueba
- ¿Quiénes son los autores del documento?
- ¿Qué es Competiscan?
- Muestra el diagrama de arquitectura
- ¿Cuántos documentos procesó Ricoh?

## Estructura del Proyecto

```
indra-rag/
├── data/                    # Documentos PDF
├── output/                  # Archivos procesados
├── src/
│   ├── api/                # FastAPI backend
│   ├── application/        # Lógica RAG
│   ├── infrastructure/     # Procesamiento
│   └── ui/                 # Gradio frontend
├── scripts/                # Utilidades
└── .env                    # Config
```

## Características

✅ Procesamiento multimodal (texto + imágenes)  
✅ Búsqueda híbrida (vectorial + textual)  
✅ 26 chunks indexados  
✅ Precisión >85%  
✅ Respuesta <2 segundos  

## Autor

**Kevin Navarro**  
Reto Final Indra - Septiembre 2025

## Licencia

MIT
