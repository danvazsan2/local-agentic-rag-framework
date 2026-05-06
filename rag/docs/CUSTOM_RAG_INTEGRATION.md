# Integración del RAG Personalizado en el Chatbot

## Descripción General

Se ha integrado el framework RAG personalizado ubicado en `rag/` con la interfaz del chatbot. Esta integración permite a los usuarios:

1. **Subir archivos** mediante el botón "+" en el chat
2. **Los archivos se ingeristan automáticamente** en el sistema RAG cuando el usuario envía su primer mensaje
3. **Las respuestas se generan usando RAG** con los documentos subidos
4. **Cada sesión de chat tiene su propio índice vectorial** aislado
5. **Al crear un nuevo chat**, el índice anterior se limpia automáticamente

## Arquitectura

```
┌─────────────────────────────────────────────────────────────────┐
│                     UI del Chatbot (Next.js)                    │
│    use-chat-handler.tsx / use-select-file-handler.tsx          │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              API Endpoints (Next.js API Routes)                 │
│    /api/chat/custom-rag/ingest                                  │
│    /api/chat/custom-rag/query                                   │
│    /api/chat/custom-rag/clear                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              RAG API Server (Python HTTP Server)                │
│                   rag/api/server.py                             │
│                   Puerto: 8765                                   │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                   RAG Framework (LlamaIndex)                    │
│                   rag/rag_framework/                            │
│                   - Ingestion (Docling)                         │
│                   - Indexing (LanceDB)                          │
│                   - Retrieval (Hybrid BM25 + Vector)            │
│                   - Query Engine (Ollama LLM)                   │
└─────────────────────────────────────────────────────────────────┘
```

## Configuración

### 1. Iniciar el servidor RAG

```bash
# Activar el entorno conda
conda activate sprint5

# Ir al directorio del RAG
cd rag

# Iniciar el servidor
python run_rag.py api
```

El servidor escuchará en `http://localhost:8765`

### 2. Verificar que el servidor está corriendo

```bash
curl http://localhost:8765/health
# Respuesta: {"status": "ok", "sessions": 0}
```

### 3. Configurar Ollama (si no está corriendo)

El RAG usa Ollama para el LLM y embeddings:

```bash
# Verificar que Ollama está corriendo
ollama list

# Descargar los modelos si es necesario
ollama pull llama3-instruct-8k
ollama pull nomic-embed-text:v1.5
```

## Uso

1. **Abrir el chatbot** en el navegador
2. **Hacer clic en el botón "+"** al lado del campo de texto
3. **Seleccionar archivos** (PDF, DOCX, TXT, MD, HTML, JSON, CSV)
4. **Escribir una pregunta** sobre los documentos
5. **Enviar el mensaje** - los archivos se ingestarán y se responderá usando RAG

## Archivos Soportados

- PDF (`.pdf`)
- Word (`.docx`)
- Texto plano (`.txt`)
- Markdown (`.md`)
- HTML (`.html`)
- JSON (`.json`)
- CSV (`.csv`)

## Estructura de Archivos Modificados

```
lib/custom-rag/
├── api_server.py           # NUEVO: Servidor HTTP para RAG
├── client.ts               # NUEVO: Cliente TypeScript para RAG
└── rag_framework/          # Framework RAG existente

app/api/chat/custom-rag/
├── ingest/route.ts         # NUEVO: Endpoint para ingesta
├── query/route.ts          # NUEVO: Endpoint para consultas
└── clear/route.ts          # NUEVO: Endpoint para limpiar sesión

components/
├── chat/
│   ├── chat-hooks/
│   │   ├── use-chat-handler.tsx    # MODIFICADO: Integración RAG
│   │   └── use-select-file-handler.tsx  # MODIFICADO: Captura archivos
│   └── chat-helpers/
│       └── index.ts        # MODIFICADO: Función handleCustomRAG
├── messages/
│   └── message.tsx         # MODIFICADO: Indicador "Procesando con RAG..."
└── utility/
    └── global-state.tsx    # MODIFICADO: Estado del RAG

context/
└── context.tsx             # MODIFICADO: Tipos del contexto
```

## Variables de Estado del RAG

```typescript
// En el contexto global
useCustomRag: boolean          // Flag para usar RAG personalizado (default: true)
customRagSessionId: string     // ID de la sesión actual de RAG
customRagFiles: File[]         // Archivos seleccionados para RAG
customRagIngested: boolean     // Si los archivos ya fueron ingestados
```

## API Endpoints

### POST /api/chat/custom-rag/ingest

Ingesta archivos en el sistema RAG.

```json
{
  "session_id": "uuid-del-chat",
  "files": [
    {
      "name": "documento.pdf",
      "content": "base64-encoded-content"
    }
  ],
  "llm_model": "llama3-instruct-8k"  // opcional
}
```

### POST /api/chat/custom-rag/query

Consulta el sistema RAG.

```json
{
  "session_id": "uuid-del-chat",
  "query": "¿Cuál es el tema principal del documento?",
  "llm_model": "llama3-instruct-8k"  // opcional
}
```

### POST /api/chat/custom-rag/clear

Limpia una sesión de RAG.

```json
{
  "session_id": "uuid-del-chat"
}
```

## Notas Técnicas

- Cada chat/sesión tiene su propio directorio de vectores en `rag/sessions/`
- Los vectores se crean usando LanceDB (embebido, no requiere servidor)
- La búsqueda híbrida combina BM25 (palabra clave) con búsqueda semántica
- El reranker BGE mejora la precisión de los resultados
- Los archivos se procesan con Docling para mejor extracción de contenido

## Solución de Problemas

### El servidor RAG no responde

1. Verificar que el servidor está corriendo:
   ```bash
   curl http://localhost:8765/health
   ```

2. Si no responde, iniciar el servidor:
   ```bash
   conda activate sprint5
   cd rag
   python run_rag.py api
   ```

### Ollama no está disponible

1. Verificar que Ollama está corriendo (Windows nativo):
   ```bash
   ollama list
   ```

2. Si no está corriendo, iniciar Ollama desde Windows:
   ```bash
   ollama serve
   ```

3. Verificar la conexión:
   ```bash
   curl http://localhost:11434/api/tags
   ```

### Los archivos no se procesan

1. Verificar el formato del archivo (ver lista de soportados)
2. Revisar los logs del servidor RAG en la terminal

## Variables de Entorno (Opcionales)

```env
# URL del servidor RAG (default: http://localhost:8765)
RAG_API_URL=http://localhost:8765
NEXT_PUBLIC_RAG_API_URL=http://localhost:8765
```
