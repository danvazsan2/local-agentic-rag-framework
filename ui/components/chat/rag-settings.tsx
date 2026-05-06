"use client"

import { ChatbotUIContext } from "@/context/context"
import { IconSettings } from "@tabler/icons-react"
import { FC, useContext, useEffect, useState } from "react"
import { LLMID } from "@/types"
import { Button } from "../ui/button"
import { Popover, PopoverContent, PopoverTrigger } from "../ui/popover"

interface RagSettingsProps {}

export const RagSettings: FC<RagSettingsProps> = ({}) => {
  const {
    chatSettings,
    setChatSettings,
    ragEnabled,
    ragIsConfigured,
    pendingRagConfig
  } = useContext(ChatbotUIContext)

  const [open, setOpen] = useState(false)
  const [llmModels, setLlmModels] = useState<string[]>([])
  const [localLlmModel, setLocalLlmModel] = useState<string>("")

  useEffect(() => {
    if (!open || ragEnabled) return

    fetch("/api/ollama/models")
      .then(r => r.json())
      .then(data => setLlmModels((data.models || []).map((m: any) => m.name)))
      .catch(() => setLlmModels([]))

    if (chatSettings?.model) {
      setLocalLlmModel(chatSettings.model)
    }
  }, [open, ragEnabled, chatSettings?.model])

  const handleModelChange = (modelId: string) => {
    setLocalLlmModel(modelId)
    if (chatSettings) {
      setChatSettings({ ...chatSettings, model: modelId as LLMID })
    }
  }

  const selectCls =
    "border-input bg-background text-foreground w-full rounded-md border px-2 py-1.5 text-xs focus:outline-none"
  const labelCls = "text-muted-foreground mb-1 block text-xs font-medium"
  const sectionCls = "space-y-1"

  return (
    <Popover open={open} onOpenChange={setOpen}>
      <PopoverTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          className="size-8"
          aria-label="Ajustes"
        >
          <IconSettings size={20} />
        </Button>
      </PopoverTrigger>

      <PopoverContent
        className="bg-background border-input flex w-[260px] flex-col space-y-3 overflow-y-auto rounded-lg border-2 p-4 dark:border-none"
        style={{ maxHeight: "360px" }}
        align="end"
      >
        {/* NON-RAG MODE: LLM selector for regular chat */}
        {!ragEnabled && (
          <div className={sectionCls}>
            <label className={labelCls}>Modelo LLM (Ollama)</label>
            {llmModels.length === 0 ? (
              <p className="text-muted-foreground text-xs">
                Ollama no disponible
              </p>
            ) : (
              <select
                className={selectCls}
                value={localLlmModel}
                onChange={e => handleModelChange(e.target.value)}
              >
                {llmModels.map(m => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            )}
          </div>
        )}

        {/* RAG MODE: show read-only config */}
        {ragEnabled && !ragIsConfigured && (
          <div className="rounded bg-blue-500/10 p-2 text-xs text-blue-700 dark:text-blue-300">
            Configura el RAG en el panel inferior y pulsa{" "}
            <strong>Activar RAG</strong> antes de comenzar.
          </div>
        )}

        {ragEnabled && ragIsConfigured && (
          <div className={sectionCls}>
            <div className="mb-1 rounded bg-blue-500/10 px-2 py-1.5 text-xs font-medium text-blue-700 dark:text-blue-300">
              Configuración activa (bloqueada)
            </div>
            <div className="text-muted-foreground space-y-0.5">
              {pendingRagConfig.presetName && (
                <p className="text-xs">
                  <span className="font-medium">Base:</span>{" "}
                  {pendingRagConfig.presetName}
                </p>
              )}
              <p className="text-xs">
                <span className="font-medium">LLM:</span>{" "}
                {pendingRagConfig.llmModel || "predeterminado"}
              </p>
              <p className="text-xs">
                <span className="font-medium">Embeddings:</span>{" "}
                {pendingRagConfig.embeddingModel || "predeterminado"}
              </p>
              <p className="text-xs">
                <span className="font-medium">Template:</span>{" "}
                {pendingRagConfig.template || "default"}
              </p>
              {pendingRagConfig.sqlPath && (
                <p className="text-xs">
                  <span className="font-medium">SQL:</span>{" "}
                  {pendingRagConfig.sqlPath}
                </p>
              )}
            </div>
            <p className="text-muted-foreground mt-2 text-xs italic">
              Para cambiar, usa &quot;Limpiar sesión&quot; en el panel RAG.
            </p>
          </div>
        )}
      </PopoverContent>
    </Popover>
  )
}
