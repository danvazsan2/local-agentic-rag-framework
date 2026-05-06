import { NextRequest, NextResponse } from "next/server"

const OLLAMA_URL =
  process.env.NEXT_PUBLIC_OLLAMA_URL || "http://localhost:11434"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    // Conectar directamente a Ollama en Windows nativo
    const response = await fetch(`${OLLAMA_URL}/api/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(body)
    })

    if (!response.ok) {
      throw new Error(`Ollama responded with status: ${response.status}`)
    }

    const text = await response.text()

    // Ollama devuelve múltiples JSONs separados por líneas (streaming)
    const lines = text
      .trim()
      .split("\n")
      .filter(line => line.trim())

    // Combinar todas las respuestas
    let fullResponse = ""
    for (const line of lines) {
      try {
        const parsed = JSON.parse(line)
        if (parsed.message?.content) {
          fullResponse += parsed.message.content
        }
      } catch (e) {
        console.warn("Error parsing line:", line)
      }
    }

    // Retornar en el formato esperado por el frontend
    const lastLine = lines[lines.length - 1]
    const lastParsed = JSON.parse(lastLine)

    return new NextResponse(
      JSON.stringify({
        ...lastParsed,
        message: {
          ...lastParsed.message,
          content: fullResponse
        }
      }),
      {
        headers: {
          "Content-Type": "application/json"
        }
      }
    )
  } catch (error) {
    console.error("Error fetching Ollama chat:", error)
    return NextResponse.json(
      {
        error:
          "No se pudo conectar a Ollama. Asegúrate de que Ollama está corriendo."
      },
      { status: 502 }
    )
  }
}
