import { NextResponse } from "next/server"

const RAG_API_URL = process.env.RAG_API_URL || "http://localhost:8765"

export async function GET() {
  try {
    const res = await fetch(`${RAG_API_URL}/configs`, {
      signal: AbortSignal.timeout(3000)
    })
    const data = await res.json()
    return NextResponse.json(data)
  } catch {
    return NextResponse.json({ configs: [] }, { status: 503 })
  }
}
