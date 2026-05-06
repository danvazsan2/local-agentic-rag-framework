import { NextRequest, NextResponse } from "next/server"

const RAG_API_URL = process.env.RAG_API_URL || "http://localhost:8765"

/**
 * POST /api/chat/custom-rag/query
 *
 * Query the RAG system for a specific session.
 *
 * Request body:
 * {
 *   "session_id": "chat-uuid",
 *   "query": "user question",
 *   "llm_model": "optional-model-name"
 * }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    const { session_id, query, llm_model } = body

    if (!session_id) {
      return NextResponse.json(
        { error: "session_id is required" },
        { status: 400 }
      )
    }

    if (!query) {
      return NextResponse.json({ error: "query is required" }, { status: 400 })
    }

    // Forward to RAG API server
    const response = await fetch(`${RAG_API_URL}/query`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        session_id,
        query,
        llm_model
      })
    })

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status })
    }

    return NextResponse.json(data)
  } catch (error: any) {
    console.error("Error in custom-rag/query:", error)

    // Check if RAG server is running
    if (error.code === "ECONNREFUSED") {
      return NextResponse.json(
        {
          error: "RAG server not available. Please start the RAG API server.",
          details: "Run: python rag/api/server.py"
        },
        { status: 503 }
      )
    }

    return NextResponse.json(
      { error: error.message || "Internal server error" },
      { status: 500 }
    )
  }
}
