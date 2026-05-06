import { NextRequest, NextResponse } from "next/server"

const RAG_API_URL = process.env.RAG_API_URL || "http://localhost:8765"

/** POST /api/chat/custom-rag/configure — update runtime config for a session */
export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const res = await fetch(`${RAG_API_URL}/configure`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    })
    const data = await res.json()
    return NextResponse.json(data, { status: res.status })
  } catch (error: any) {
    if (error.code === "ECONNREFUSED") {
      return NextResponse.json(
        {
          error: "RAG server not available.",
          details: "Run: python rag/run_rag.py api"
        },
        { status: 503 }
      )
    }
    return NextResponse.json({ error: error.message }, { status: 500 })
  }
}
