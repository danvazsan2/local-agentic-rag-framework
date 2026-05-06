import { NextResponse } from "next/server"

const RAG_API_URL = process.env.RAG_API_URL || "http://localhost:8765"

export async function GET() {
  try {
    const res = await fetch(`${RAG_API_URL}/health`, {
      signal: AbortSignal.timeout(3000)
    })
    const data = await res.json()
    return NextResponse.json({ online: true, ...data })
  } catch {
    return NextResponse.json({ online: false }, { status: 503 })
  }
}
