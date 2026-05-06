import { NextRequest, NextResponse } from "next/server"

const RAG_API_URL = process.env.RAG_API_URL || "http://localhost:8765"

/**
 * POST /api/chat/custom-rag/clear
 *
 * Clear a RAG session (remove indexed documents).
 *
 * Request body:
 * {
 *   "session_id": "chat-uuid"
 * }
 */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    const { session_id } = body

    if (!session_id) {
      return NextResponse.json(
        { error: "session_id is required" },
        { status: 400 }
      )
    }

    // Forward to RAG API server
    const response = await fetch(`${RAG_API_URL}/clear`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        session_id
      })
    })

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status })
    }

    return NextResponse.json(data)
  } catch (error: any) {
    console.error("Error in custom-rag/clear:", error)

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
