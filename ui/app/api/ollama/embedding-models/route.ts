import { NextResponse } from "next/server"

const OLLAMA_URL =
  process.env.NEXT_PUBLIC_OLLAMA_URL || "http://localhost:11434"

export const dynamic = "force-dynamic"
export const revalidate = 0

export async function GET() {
  try {
    const res = await fetch(`${OLLAMA_URL}/api/tags`, { cache: "no-store" })
    if (!res.ok) return NextResponse.json({ models: [] })
    const data = await res.json()
    const embeddingFamilies = ["bert", "nomic-bert", "xlm-roberta"]
    const embeds = (data.models || []).filter((m: any) => {
      const n = m.name.toLowerCase()
      const family = (m.details?.family || "").toLowerCase()
      const families = (m.details?.families || []).map((f: string) =>
        f.toLowerCase()
      )
      const allFamilies = [family, ...families]
      const isReranker = n.includes("reranker") || n.includes("rerank")
      return (
        !isReranker &&
        (n.includes("embed") ||
          n.includes("embedding") ||
          allFamilies.some(f => embeddingFamilies.includes(f)))
      )
    })
    return NextResponse.json(
      { models: embeds },
      {
        headers: {
          "Cache-Control":
            "no-store, no-cache, must-revalidate, proxy-revalidate"
        }
      }
    )
  } catch {
    return NextResponse.json({ models: [] })
  }
}
