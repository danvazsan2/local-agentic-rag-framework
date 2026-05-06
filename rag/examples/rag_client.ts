/**
 * RAG API Client Example
 * ======================
 *
 * Example TypeScript client for interacting with the RAG API server.
 * Copy this file to your UI project and adjust as needed.
 *
 * Usage:
 *   import { RAGClient } from './rag_client'
 *
 *   const client = new RAGClient('http://localhost:8765')
 *   await client.ingest(sessionId, files)
 *   const response = await client.query(sessionId, 'What is X?')
 */

export interface RAGFile {
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
  error?: string
}

export interface ClearResponse {
  success: boolean
  session_id: string
  message: string
  error?: string
}

export interface HealthResponse {
  status: string
  version: string
  sessions: number
}

export interface ConfigResponse {
  config: {
    llm: { provider: string; model: string }
    embedding: { provider: string; model: string }
    retrieval?: {
      use_hybrid_search: boolean
      top_k: number
      reranker_enabled: boolean
    }
  }
  version: string
}

/**
 * Convert a File object to base64 string
 */
export const fileToBase64 = (file: File): Promise<string> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      const base64 = result.split(",")[1]
      resolve(base64)
    }
    reader.onerror = error => reject(error)
    reader.readAsDataURL(file)
  })
}

/**
 * RAG API Client
 */
export class RAGClient {
  private baseUrl: string

  constructor(baseUrl: string = "http://localhost:8765") {
    this.baseUrl = baseUrl.replace(/\/$/, "") // Remove trailing slash
  }

  /**
   * Check server health
   */
  async health(): Promise<HealthResponse> {
    const response = await fetch(`${this.baseUrl}/health`)
    return response.json()
  }

  /**
   * Get server configuration
   */
  async config(): Promise<ConfigResponse> {
    const response = await fetch(`${this.baseUrl}/config`)
    return response.json()
  }

  /**
   * Ingest files into a session
   */
  async ingest(
    sessionId: string,
    files: RAGFile[],
    llmModel?: string
  ): Promise<IngestResponse> {
    const response = await fetch(`${this.baseUrl}/ingest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        files,
        llm_model: llmModel
      })
    })
    return response.json()
  }

  /**
   * Query the RAG system
   */
  async query(
    sessionId: string,
    query: string,
    llmModel?: string
  ): Promise<QueryResponse> {
    const response = await fetch(`${this.baseUrl}/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        query,
        llm_model: llmModel
      })
    })
    return response.json()
  }

  /**
   * Clear a session
   */
  async clear(sessionId: string): Promise<ClearResponse> {
    const response = await fetch(`${this.baseUrl}/clear`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId })
    })
    return response.json()
  }

  /**
   * Helper to ingest File objects directly
   */
  async ingestFiles(
    sessionId: string,
    files: File[],
    llmModel?: string
  ): Promise<IngestResponse> {
    const ragFiles: RAGFile[] = await Promise.all(
      files.map(async file => ({
        name: file.name,
        content: await fileToBase64(file)
      }))
    )
    return this.ingest(sessionId, ragFiles, llmModel)
  }
}

// Default export for convenience
export default RAGClient
