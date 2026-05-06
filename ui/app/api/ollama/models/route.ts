import { NextResponse } from "next/server"

const OLLAMA_URL =
  process.env.NEXT_PUBLIC_OLLAMA_URL || "http://localhost:11434"

export const dynamic = "force-dynamic"
export const revalidate = 0

export async function GET() {
  try {
    // Conectar directamente a Ollama en Windows nativo
    const response = await fetch(`${OLLAMA_URL}/api/tags`, {
      cache: "no-store"
    })

    if (!response.ok) {
      throw new Error(`Ollama responded with status: ${response.status}`)
    }

    const data = await response.json()

    const embeddingFamilies = ["bert", "nomic-bert", "xlm-roberta"]
    const filteredModels = data.models.filter((model: any) => {
      const name = model.name.toLowerCase()
      const family = (model.details?.family || "").toLowerCase()
      const families = (model.details?.families || []).map((f: string) =>
        f.toLowerCase()
      )
      const allFamilies = [family, ...families]

      const isNotLLM =
        name.includes("embed") ||
        name.includes("embedding") ||
        name.includes("reranker") ||
        name.includes("rerank") ||
        allFamilies.some(f => embeddingFamilies.includes(f))

      return !isNotLLM
    })

    return NextResponse.json(
      { models: filteredModels },
      {
        headers: {
          "Cache-Control":
            "no-store, no-cache, must-revalidate, proxy-revalidate"
        }
      }
    )
  } catch (error) {
    console.error("Error fetching Ollama models:", error)
    return NextResponse.json(
      {
        error:
          "No se pudo conectar a Ollama. Asegúrate de que Ollama está corriendo."
      },
      { status: 502 }
    )
  }
}
