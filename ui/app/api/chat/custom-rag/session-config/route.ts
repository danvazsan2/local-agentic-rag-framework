import { NextRequest, NextResponse } from "next/server"

const RAG_API_URL = process.env.RAG_API_URL || "http://localhost:8765"

export const dynamic = "force-dynamic"
export const revalidate = 0

/** GET /api/chat/custom-rag/session-config?session_id=... — get per-session config */
export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const sessionId = searchParams.get("session_id")
    const url = sessionId
      ? `${RAG_API_URL}/session-config?session_id=${encodeURIComponent(sessionId)}`
      : `${RAG_API_URL}/session-config`

    const res = await fetch(url, { cache: "no-store" })
    const data = await res.json()
    return NextResponse.json(data, {
      status: res.status,
      headers: {
        "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate"
      }
    })
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
