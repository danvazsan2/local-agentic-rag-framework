"use client"

import { ChatbotUIContext, RagFile, RagServerStatus } from "@/context/context"
import {
  createRagSessionWithConfig,
  getConfigs,
  getHealth,
  getTemplates,
  ingestFiles
} from "@/lib/custom-rag/client"
import { cn } from "@/lib/utils"
import {
  IconAlertCircle,
  IconCheck,
  IconChevronDown,
  IconChevronRight,
  IconDatabase,
  IconFileTypeCsv,
  IconFileTypeDocx,
  IconFileTypePdf,
  IconFileTypeTxt,
  IconJson,
  IconLoader2,
  IconMarkdown,
  IconRefresh,
  IconSettings,
  IconX
} from "@tabler/icons-react"
import { FC, useCallback, useContext, useEffect, useRef, useState } from "react"
import { toast } from "sonner"
import { v4 as uuidv4 } from "uuid"

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

const ServerDot: FC<{ status: RagServerStatus }> = ({ status }) => {
  const colors: Record<RagServerStatus, string> = {
    online: "bg-green-500",
    offline: "bg-red-500",
    checking: "bg-yellow-500 animate-pulse",
    unknown: "bg-gray-400"
  }
  return (
    <span
      className={cn("inline-block size-2 rounded-full", colors[status])}
      aria-label={status}
    />
  )
}

const FileIcon: FC<{ name: string }> = ({ name }) => {
  const ext = name.split(".").pop()?.toLowerCase()
  const cls = "size-3.5 shrink-0"
  switch (ext) {
    case "pdf":
      return <IconFileTypePdf className={cls} />
    case "docx":
      return <IconFileTypeDocx className={cls} />
    case "csv":
      return <IconFileTypeCsv className={cls} />
    case "txt":
      return <IconFileTypeTxt className={cls} />
    case "md":
      return <IconMarkdown className={cls} />
    case "json":
      return <IconJson className={cls} />
    default:
      return <IconDatabase className={cls} />
  }
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

interface RagPanelProps {
  onDisable: () => void
}

export const RagPanel: FC<RagPanelProps> = ({ onDisable }) => {
  const {
    ragSessionId,
    setRagSessionId,
    ragFiles,
    setRagFiles,
    ragServerStatus,
    setRagServerStatus,
    ragIsConfigured,
    setRagIsConfigured,
    pendingRagConfig,
    setPendingRagConfig
  } = useContext(ChatbotUIContext)

  // Options for config form
  const [llmModels, setLlmModels] = useState<string[]>([])
  const [embeddingModels, setEmbeddingModels] = useState<string[]>([])
  const [presets, setPresets] = useState<{ name: string; path: string }[]>([])
  const [templates, setTemplates] = useState<
    { name: string; description: string }[]
  >([])
  const [loadingOptions, setLoadingOptions] = useState(false)

  // UI state
  const [showCustomPrompt, setShowCustomPrompt] = useState(false)
  const [activating, setActivating] = useState(false)
  const [showConfigDetail, setShowConfigDetail] = useState(false)

  const healthIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null)

  // -------------------------------------------------------------------------
  // Health polling
  // -------------------------------------------------------------------------
  const checkHealth = useCallback(async () => {
    setRagServerStatus("checking")
    const h = await getHealth()
    setRagServerStatus(h.online ? "online" : "offline")
  }, [setRagServerStatus])

  useEffect(() => {
    checkHealth()
    healthIntervalRef.current = setInterval(checkHealth, 30_000)
    return () => {
      if (healthIntervalRef.current) clearInterval(healthIntervalRef.current)
    }
  }, [checkHealth])

  // -------------------------------------------------------------------------
  // Load config options (only in config mode)
  // -------------------------------------------------------------------------
  useEffect(() => {
    if (ragIsConfigured) return

    setLoadingOptions(true)
    Promise.allSettled([
      fetch("/api/ollama/models")
        .then(r => r.json())
        .then(d => setLlmModels((d.models || []).map((m: any) => m.name)))
        .catch(() => {}),
      fetch("/api/ollama/embedding-models")
        .then(r => r.json())
        .then(d => setEmbeddingModels((d.models || []).map((m: any) => m.name)))
        .catch(() => {}),
      getConfigs()
        .then(d => setPresets(d.configs || []))
        .catch(() => {}),
      getTemplates()
        .then(d => setTemplates(d.templates || []))
        .catch(() => {})
    ]).finally(() => setLoadingOptions(false))
  }, [ragIsConfigured])

  // -------------------------------------------------------------------------
  // Activate RAG with selected config
  // -------------------------------------------------------------------------
  const handleActivate = useCallback(async () => {
    setActivating(true)
    try {
      const newId = `session-${uuidv4()}`
      const res = await createRagSessionWithConfig(newId, pendingRagConfig)
      if (res.success || res.session_id) {
        setRagSessionId(newId)
        setRagIsConfigured(true)
        toast.success("RAG activado")
      } else {
        toast.error(res.error || "Error al activar RAG")
      }
    } catch {
      toast.error("No se pudo conectar con el servidor RAG")
    } finally {
      setActivating(false)
    }
  }, [pendingRagConfig, setRagSessionId, setRagIsConfigured])

  // -------------------------------------------------------------------------
  // File actions (file mode)
  // -------------------------------------------------------------------------
  const handleRetry = useCallback(
    async (file: RagFile) => {
      if (!ragSessionId) return
      setRagFiles(prev =>
        prev.map(f =>
          f.id === file.id ? { ...f, status: "ingesting", error: undefined } : f
        )
      )
      try {
        const result = await ingestFiles(ragSessionId, [
          { name: file.name, file: file.file }
        ])
        setRagFiles(prev =>
          prev.map(f =>
            f.id === file.id
              ? {
                  ...f,
                  status: result.success ? "ready" : "error",
                  error: result.error
                }
              : f
          )
        )
      } catch (err: any) {
        setRagFiles(prev =>
          prev.map(f =>
            f.id === file.id ? { ...f, status: "error", error: err.message } : f
          )
        )
      }
    },
    [ragSessionId, setRagFiles]
  )

  const handleRemoveFile = useCallback(
    (id: string) => {
      setRagFiles(prev => prev.filter(f => f.id !== id))
    },
    [setRagFiles]
  )

  // -------------------------------------------------------------------------
  // Derived state
  // -------------------------------------------------------------------------
  const readyCount = ragFiles.filter(f => f.status === "ready").length
  const ingestingCount = ragFiles.filter(f => f.status === "ingesting").length
  const errorCount = ragFiles.filter(f => f.status === "error").length

  const selectCls =
    "border-input bg-background/90 text-foreground w-full rounded border px-2 py-1 text-xs backdrop-blur-sm focus:outline-none focus:ring-1 focus:ring-blue-400/20"
  const labelCls = "text-muted-foreground mb-0.5 block text-xs font-medium"

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <div className="bg-background/85 relative z-20 rounded-xl border-2 border-blue-500/35 px-3 py-2.5 shadow-md backdrop-blur-sm dark:border-blue-400/30">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <IconDatabase
            size={16}
            className="text-blue-500 dark:text-blue-400"
          />
          <span className="text-sm font-semibold text-blue-700 dark:text-blue-300">
            RAG
          </span>
          <ServerDot status={ragServerStatus} />
          {ragServerStatus === "offline" && (
            <span className="text-xs text-red-500">servidor no disponible</span>
          )}
          {ragIsConfigured && ragFiles.length > 0 && (
            <span className="text-muted-foreground text-xs">
              {readyCount}/{ragFiles.length} listos
              {ingestingCount > 0 && ` · ${ingestingCount} indexando`}
              {errorCount > 0 && ` · ${errorCount} error`}
            </span>
          )}
          {ragIsConfigured && (
            <button
              onClick={() => setShowConfigDetail(v => !v)}
              title="Ver configuración activa"
              className="text-muted-foreground hover:text-foreground transition-colors"
            >
              <IconSettings size={13} />
            </button>
          )}
        </div>
        <button
          onClick={onDisable}
          aria-label="Desactivar RAG"
          className="text-muted-foreground hover:text-foreground rounded p-0.5 transition-colors hover:bg-white/10"
        >
          <IconX size={14} />
        </button>
      </div>

      {/* Config detail (file mode) */}
      {ragIsConfigured && showConfigDetail && (
        <div className="mt-2 rounded border border-blue-500/20 bg-blue-500/10 p-2 text-xs">
          <p className="mb-1 font-semibold text-blue-700 dark:text-blue-300">
            Configuración activa
          </p>
          <div className="text-muted-foreground space-y-0.5">
            {pendingRagConfig.presetName && (
              <div>
                <span className="font-medium">Base:</span>{" "}
                {pendingRagConfig.presetName}
              </div>
            )}
            <div>
              <span className="font-medium">LLM:</span>{" "}
              {pendingRagConfig.llmModel || "predeterminado"}
            </div>
            <div>
              <span className="font-medium">Embeddings:</span>{" "}
              {pendingRagConfig.embeddingModel || "predeterminado"}
            </div>
            <div>
              <span className="font-medium">BD vectorial:</span>{" "}
              {pendingRagConfig.vectorStore || "LanceDB"}
            </div>
            <div>
              <span className="font-medium">Recuperación:</span>{" "}
              {pendingRagConfig.retrievalMode || "híbrido"}
            </div>
            <div>
              <span className="font-medium">Fuente:</span>{" "}
              {pendingRagConfig.dataSourceMode || "documentos"}
            </div>
            <div>
              <span className="font-medium">Template:</span>{" "}
              {pendingRagConfig.template || "default"}
            </div>
            {pendingRagConfig.sqlPath && (
              <div>
                <span className="font-medium">SQL:</span>{" "}
                {pendingRagConfig.sqlPath}
              </div>
            )}
          </div>
          <p className="text-muted-foreground mt-1 italic">
            Bloqueado. Crea un nuevo chat para reconfigurar.
          </p>
        </div>
      )}

      {/* ─── CONFIG MODE ─── */}
      {!ragIsConfigured && (
        <div className="mt-2 space-y-2">
          {ragServerStatus === "offline" && (
            <div className="rounded bg-red-500/10 p-2 text-xs text-red-600">
              Servidor no disponible. Inícialo con:{" "}
              <code className="font-mono">python rag/run_rag.py api</code>
            </div>
          )}

          {loadingOptions && (
            <div className="text-muted-foreground flex items-center gap-1.5 text-xs">
              <IconLoader2 size={12} className="animate-spin" />
              Cargando opciones...
            </div>
          )}

          {/* Preset */}
          <div>
            <label className={labelCls}>Configuración base</label>
            <select
              className={selectCls}
              value={pendingRagConfig.presetName}
              onChange={e =>
                setPendingRagConfig(p => ({
                  ...p,
                  presetName: e.target.value
                }))
              }
            >
              <option value="">— predeterminada (servidor) —</option>
              {presets.map(p => (
                <option key={p.path} value={p.name}>
                  {p.name}
                </option>
              ))}
            </select>
          </div>

          {/* LLM + Embedding */}
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className={labelCls}>Modelo LLM</label>
              <select
                className={selectCls}
                value={pendingRagConfig.llmModel}
                onChange={e =>
                  setPendingRagConfig(p => ({ ...p, llmModel: e.target.value }))
                }
              >
                <option value="">— predeterminado —</option>
                {llmModels.map(m => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className={labelCls}>Embeddings</label>
              <select
                className={selectCls}
                value={pendingRagConfig.embeddingModel}
                onChange={e =>
                  setPendingRagConfig(p => ({
                    ...p,
                    embeddingModel: e.target.value
                  }))
                }
              >
                <option value="">— predeterminado —</option>
                {embeddingModels.map(m => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {/* Vector DB + Retrieval mode */}
          <div className="grid grid-cols-2 gap-2">
            <div>
              <label className={labelCls}>Base de datos vectorial</label>
              <select
                className={selectCls}
                value={pendingRagConfig.vectorStore}
                onChange={e =>
                  setPendingRagConfig(p => ({
                    ...p,
                    vectorStore: e.target.value
                  }))
                }
              >
                <option value="">LanceDB (pred.)</option>
                <option value="lancedb">LanceDB</option>
                <option value="chroma">Chroma</option>
                <option value="faiss">FAISS</option>
              </select>
            </div>
            <div>
              <label className={labelCls}>Método de recuperación</label>
              <select
                className={selectCls}
                value={pendingRagConfig.retrievalMode}
                onChange={e =>
                  setPendingRagConfig(p => ({
                    ...p,
                    retrievalMode: e.target.value
                  }))
                }
              >
                <option value="">Híbrido (pred.)</option>
                <option value="hybrid">Híbrido</option>
                <option value="semantic">Semántico</option>
                <option value="bm25">BM25</option>
              </select>
            </div>
          </div>

          {/* Data source */}
          <div>
            <label className={labelCls}>Fuente de datos</label>
            <select
              className={selectCls}
              value={pendingRagConfig.dataSourceMode}
              onChange={e =>
                setPendingRagConfig(p => ({
                  ...p,
                  dataSourceMode: e.target.value
                }))
              }
            >
              <option value="">No estructurado (documentos)</option>
              <option value="documents">No estructurado (documentos)</option>
              <option value="sql">Estructurado (SQL)</option>
              <option value="auto">Auto (documentos + SQL)</option>
            </select>
          </div>

          {/* Template */}
          <div>
            <label className={labelCls}>Template de prompt</label>
            <select
              className={selectCls}
              value={pendingRagConfig.template}
              onChange={e =>
                setPendingRagConfig(p => ({ ...p, template: e.target.value }))
              }
            >
              <option value="default">default</option>
              {templates
                .filter(t => t.name !== "default")
                .map(t => (
                  <option key={t.name} value={t.name} title={t.description}>
                    {t.name}
                  </option>
                ))}
              <option value="custom">custom (personalizado)</option>
            </select>
          </div>

          {/* SQL path */}
          <div>
            <label className={labelCls}>
              SQL database{" "}
              <span className="text-muted-foreground font-normal">
                (opcional)
              </span>
            </label>
            <input
              type="text"
              className={selectCls}
              placeholder="ruta/a/base.sqlite"
              value={pendingRagConfig.sqlPath}
              onChange={e =>
                setPendingRagConfig(p => ({ ...p, sqlPath: e.target.value }))
              }
            />
          </div>

          {/* Custom prompt (collapsible) */}
          {pendingRagConfig.template !== "custom" && !showCustomPrompt && (
            <button
              onClick={() => setShowCustomPrompt(true)}
              className="text-muted-foreground hover:text-foreground flex items-center gap-1 text-xs transition-colors"
            >
              <IconChevronRight size={12} />
              Prompt personalizado
            </button>
          )}

          {(pendingRagConfig.template === "custom" || showCustomPrompt) && (
            <div>
              <div className="mb-0.5 flex items-center justify-between">
                <label className={labelCls + " mb-0"}>
                  Prompt personalizado
                </label>
                {showCustomPrompt && pendingRagConfig.template !== "custom" && (
                  <button
                    onClick={() => setShowCustomPrompt(false)}
                    className="text-muted-foreground hover:text-foreground text-xs"
                  >
                    <IconChevronDown size={12} />
                  </button>
                )}
              </div>
              <p className="text-muted-foreground mb-1 text-xs">
                Usa <code className="font-mono">{"{context_str}"}</code> y{" "}
                <code className="font-mono">{"{query_str}"}</code>
              </p>
              <textarea
                className="border-input bg-background/90 text-foreground placeholder:text-muted-foreground w-full rounded border px-2 py-1 text-xs backdrop-blur-sm focus:outline-none focus:ring-1 focus:ring-blue-400/20"
                rows={3}
                placeholder={
                  "Dado el contexto:\n{context_str}\nResponde: {query_str}"
                }
                value={pendingRagConfig.customPrompt}
                onChange={e =>
                  setPendingRagConfig(p => ({
                    ...p,
                    customPrompt: e.target.value
                  }))
                }
              />
            </div>
          )}

          {/* Activate button */}
          <div className="flex justify-end pt-1">
            <button
              onClick={handleActivate}
              disabled={activating || ragServerStatus === "offline"}
              className="rounded bg-blue-500 px-3 py-1.5 text-xs font-medium text-white transition-colors hover:bg-blue-600 disabled:opacity-50"
            >
              {activating ? (
                <span className="flex items-center gap-1">
                  <IconLoader2 size={12} className="animate-spin" />
                  Activando...
                </span>
              ) : (
                "Activar RAG"
              )}
            </button>
          </div>
        </div>
      )}

      {/* ─── FILE MODE ─── */}
      {ragIsConfigured && (
        <>
          {ragFiles.length > 0 && (
            <div className="mt-2 space-y-0.5">
              {ragFiles.map(f => (
                <div
                  key={f.id}
                  className="flex items-center justify-between rounded px-1 py-0.5 text-xs"
                >
                  <div className="text-muted-foreground flex min-w-0 items-center gap-1.5">
                    <FileIcon name={f.name} />
                    <span className="truncate">{f.name}</span>
                  </div>
                  <div className="ml-2 flex shrink-0 items-center gap-1">
                    {f.status === "ingesting" && (
                      <IconLoader2
                        size={12}
                        className="animate-spin text-yellow-500"
                      />
                    )}
                    {f.status === "ready" && (
                      <IconCheck size={12} className="text-green-500" />
                    )}
                    {f.status === "queued" && (
                      <IconLoader2
                        size={12}
                        className="text-muted-foreground"
                      />
                    )}
                    {f.status === "error" && (
                      <>
                        <span aria-label={f.error ?? "error"}>
                          <IconAlertCircle size={12} className="text-red-500" />
                        </span>
                        <button
                          onClick={() => handleRetry(f)}
                          aria-label="Reintentar"
                          className="text-muted-foreground hover:text-foreground"
                        >
                          <IconRefresh size={12} />
                        </button>
                      </>
                    )}
                    <button
                      onClick={() => handleRemoveFile(f.id)}
                      className="text-muted-foreground hover:text-red-500"
                    >
                      <IconX size={12} />
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}

          {ragFiles.length === 0 && (
            <p className="text-muted-foreground mt-1.5 text-xs">
              Sube documentos con el botón <span className="font-mono">+</span>{" "}
              para indexarlos.
            </p>
          )}
        </>
      )}
    </div>
  )
}
