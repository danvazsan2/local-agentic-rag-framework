/**
 * TypeScript client for the RAG API server (rag/api/server.py, port 8765).
 * All calls go through the Next.js proxy routes at /api/chat/custom-rag/*.
 */

export interface RagHealthResponse {
  online: boolean
  version?: string
  sessions?: number
}

export interface RagConfig {
  name: string
  path: string
}

export interface RagConfigsResponse {
  configs: RagConfig[]
}

export interface RagSessionResponse {
  success: boolean
  session_id: string
  config?: object
  config_source?: string
  error?: string
}

export interface RagFile {
  name: string
  content: string // base64 encoded
}

export interface IngestResponse {
  success: boolean
  session_id: string
  files_ingested: string[]
  total_files: number
  error?: string
}

export interface QueryResponse {
  success: boolean
  session_id: string
  query: string
  response: string
  model?: string
  error?: string
}

export interface ClearResponse {
  success: boolean
  session_id: string
  message: string
  error?: string
}

export interface ConfigureResponse {
  success: boolean
  session_id: string
  requires_reindex?: boolean
  status?: number
  config?: {
    llm: { provider: string; model: string }
    embedding: { provider: string; model: string }
    prompt_template?: string
    custom_prompt?: string
    vector_store_type?: string
    sql_enabled?: boolean
    sql_path?: string
  }
  error?: string
}

type ApiErrorResponse = {
  success?: boolean
  error?: string
  details?: string
  status?: number
}

async function readJsonSafe<T>(response: Response): Promise<T | null> {
  try {
    return (await response.json()) as T
  } catch {
    return null
  }
}

async function fetchJsonOrError<
  T extends { success?: boolean; error?: string }
>(input: RequestInfo | URL, init?: RequestInit): Promise<T> {
  try {
    const response = await fetch(input, init)
    const data = await readJsonSafe<T & ApiErrorResponse>(response)

    if (!response.ok) {
      const message =
        data?.error || `Request failed with status ${response.status}`

      const errorResponse: ApiErrorResponse = {
        ...(data || {}),
        success: false,
        error: message,
        status: response.status
      }

      return errorResponse as unknown as T
    }

    return data as T
  } catch (error: any) {
    const errorResponse: ApiErrorResponse = {
      success: false,
      error: error?.message || "Network error"
    }

    return errorResponse as unknown as T
  }
}

// ---------------------------------------------------------------------------
// Health & config
// ---------------------------------------------------------------------------

export const getHealth = async (): Promise<RagHealthResponse> => {
  try {
    const res = await fetch("/api/chat/custom-rag/health")
    return res.json()
  } catch {
    return { online: false }
  }
}

export const getConfigs = async (): Promise<RagConfigsResponse> => {
  try {
    const res = await fetch("/api/chat/custom-rag/configs")
    return res.json()
  } catch {
    return { configs: [] }
  }
}

// ---------------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------------

export const createRagSession = async (
  sessionId: string,
  configName?: string | null
): Promise<RagSessionResponse> => {
  return fetchJsonOrError<RagSessionResponse>("/api/chat/custom-rag/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, config_name: configName })
  })
}

export const createRagSessionWithConfig = async (
  sessionId: string,
  opts: {
    presetName?: string
    llmModel?: string
    embeddingModel?: string
    template?: string
    sqlPath?: string
    customPrompt?: string
    vectorStore?: string
    retrievalMode?: string
    dataSourceMode?: string
  }
): Promise<RagSessionResponse> => {
  return fetchJsonOrError<RagSessionResponse>("/api/chat/custom-rag/session", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      config_name: opts.presetName || null,
      llm_model: opts.llmModel || null,
      embedding_model: opts.embeddingModel || null,
      prompt_template:
        opts.template && opts.template !== "default" ? opts.template : null,
      sql_path: opts.sqlPath || null,
      custom_prompt: opts.customPrompt || null,
      vector_store: opts.vectorStore || null,
      retrieval_mode: opts.retrievalMode || null,
      data_source_mode: opts.dataSourceMode || null
    })
  })
}

// ---------------------------------------------------------------------------
// Ingestion
// ---------------------------------------------------------------------------

const toBase64 = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => resolve((reader.result as string).split(",")[1])
    reader.onerror = reject
    reader.readAsDataURL(file)
  })

export const ingestFiles = async (
  sessionId: string,
  files: { name: string; file: File }[]
): Promise<IngestResponse> => {
  const ragFiles: RagFile[] = await Promise.all(
    files.map(async ({ name, file }) => ({
      name,
      content: await toBase64(file)
    }))
  )

  return fetchJsonOrError<IngestResponse>("/api/chat/custom-rag/ingest", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, files: ragFiles })
  })
}

// ---------------------------------------------------------------------------
// Query & clear
// ---------------------------------------------------------------------------

export const queryRAG = async (
  sessionId: string,
  query: string,
  llmModel?: string
): Promise<QueryResponse> => {
  return fetchJsonOrError<QueryResponse>("/api/chat/custom-rag/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, query, llm_model: llmModel })
  })
}

export const configureRAG = async (
  sessionId: string,
  opts: {
    llmModel?: string
    promptTemplate?: string
    customPrompt?: string
    sqlPath?: string
    embeddingModel?: string
    vectorStoreType?: string
  }
): Promise<ConfigureResponse> => {
  return fetchJsonOrError<ConfigureResponse>("/api/chat/custom-rag/configure", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      session_id: sessionId,
      llm_model: opts.llmModel,
      prompt_template: opts.promptTemplate,
      custom_prompt: opts.customPrompt,
      sql_path: opts.sqlPath,
      embedding_model: opts.embeddingModel,
      vector_store_type: opts.vectorStoreType
    })
  })
}

export const getTemplates = async (): Promise<{
  templates: { name: string; description: string }[]
}> => {
  try {
    const r = await fetch("/api/chat/custom-rag/templates", {
      cache: "no-store"
    })
    return (
      (await readJsonSafe<{
        templates: { name: string; description: string }[]
      }>(r)) || {
        templates: []
      }
    )
  } catch {
    return { templates: [] }
  }
}

export const getSessionConfig = async (sessionId: string): Promise<any> => {
  try {
    const r = await fetch(
      `/api/chat/custom-rag/session-config?session_id=${encodeURIComponent(sessionId)}`,
      { cache: "no-store" }
    )
    return readJsonSafe(r)
  } catch {
    return null
  }
}

export const clearRAGSession = async (
  sessionId: string
): Promise<ClearResponse> => {
  return fetchJsonOrError<ClearResponse>("/api/chat/custom-rag/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId })
  })
}
