import { ChatbotUIContext } from "@/context/context"
import {
  createDocXFile,
  createFile,
  createFileWithoutProcessing
} from "@/db/files"
import { ingestFiles } from "@/lib/custom-rag/client"
import { LLM_LIST } from "@/lib/models/llm/llm-list"
import mammoth from "mammoth"
import { useContext, useEffect, useState } from "react"
import { toast } from "sonner"
import { v4 as uuidv4 } from "uuid"

export const ACCEPTED_FILE_TYPES = [
  "text/csv",
  "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  "application/json",
  "text/markdown",
  "application/pdf",
  "text/plain",
  "text/html"
].join(",")

export const useSelectFileHandler = () => {
  const {
    selectedWorkspace,
    profile,
    chatSettings,
    setNewMessageImages,
    setNewMessageFiles,
    setShowFilesDisplay,
    setFiles,
    setUseRetrieval,
    // RAG
    ragEnabled,
    ragSessionId,
    setRagFiles
  } = useContext(ChatbotUIContext)

  const [filesToAccept, setFilesToAccept] = useState(ACCEPTED_FILE_TYPES)

  useEffect(() => {
    handleFilesToAccept()
  }, [chatSettings?.model])

  const handleFilesToAccept = () => {
    const model = chatSettings?.model
    const FULL_MODEL = LLM_LIST.find(llm => llm.modelId === model)
    if (!FULL_MODEL) return
    setFilesToAccept(
      FULL_MODEL.imageInput
        ? `${ACCEPTED_FILE_TYPES},image/*`
        : ACCEPTED_FILE_TYPES
    )
  }

  const handleSelectDeviceFile = async (fileList: FileList) => {
    if (!profile || !selectedWorkspace || !chatSettings) return

    setShowFilesDisplay(true)
    setUseRetrieval(true)

    const filesArray = Array.from(fileList)

    // --- RAG ingestion (per-file tracking) ---
    if (ragEnabled && ragSessionId) {
      const docFiles = filesArray.filter(f => !f.type.includes("image"))

      for (const file of docFiles) {
        const fileId = uuidv4()

        // Register as 'ingesting'
        setRagFiles(prev => {
          if (prev.some(f => f.name === file.name)) return prev
          return [
            ...prev,
            { id: fileId, name: file.name, file, status: "ingesting" }
          ]
        })

        // Ingest in background
        ;(async () => {
          try {
            const result = await ingestFiles(ragSessionId, [
              { name: file.name, file }
            ])
            if (result.success && result.files_ingested?.length > 0) {
              setRagFiles(prev =>
                prev.map(f => (f.id === fileId ? { ...f, status: "ready" } : f))
              )
            } else {
              const errMsg = result.error ?? "No se pudo indexar el archivo"
              setRagFiles(prev =>
                prev.map(f =>
                  f.id === fileId ? { ...f, status: "error", error: errMsg } : f
                )
              )
              toast.error(`RAG: ${errMsg}`)
            }
          } catch (err: any) {
            setRagFiles(prev =>
              prev.map(f =>
                f.id === fileId
                  ? { ...f, status: "error", error: err.message }
                  : f
              )
            )
            toast.error(`RAG: ${err.message}`)
          }
        })()
      }
    }

    // --- Standard file processing (Supabase storage) ---
    for (const file of filesArray) {
      let simplifiedFileType = file.type.split("/")[1]

      const reader = new FileReader()

      if (file.type.includes("image")) {
        reader.readAsDataURL(file)
      } else if (ACCEPTED_FILE_TYPES.split(",").includes(file.type)) {
        if (simplifiedFileType.includes("vnd.adobe.pdf")) {
          simplifiedFileType = "pdf"
        } else if (
          simplifiedFileType.includes(
            "vnd.openxmlformats-officedocument.wordprocessingml.document"
          ) ||
          simplifiedFileType.includes("docx")
        ) {
          simplifiedFileType = "docx"
        }

        setNewMessageFiles(prev => [
          ...prev,
          { id: "loading", name: file.name, type: simplifiedFileType, file }
        ])

        if (
          file.type.includes(
            "vnd.openxmlformats-officedocument.wordprocessingml.document"
          )
        ) {
          const arrayBuffer = await file.arrayBuffer()
          const result = await mammoth.extractRawText({ arrayBuffer })

          const createdFile = ragEnabled
            ? await createFileWithoutProcessing(
                file,
                {
                  user_id: profile.user_id,
                  description: "",
                  file_path: "",
                  name: file.name,
                  size: file.size,
                  tokens: 0,
                  type: simplifiedFileType
                },
                selectedWorkspace.id
              )
            : await createDocXFile(
                result.value,
                file,
                {
                  user_id: profile.user_id,
                  description: "",
                  file_path: "",
                  name: file.name,
                  size: file.size,
                  tokens: 0,
                  type: simplifiedFileType
                },
                selectedWorkspace.id,
                chatSettings.embeddingsProvider
              )

          setFiles(prev => [...prev, createdFile])
          setNewMessageFiles(prev =>
            prev.map(item =>
              item.id === "loading"
                ? {
                    id: createdFile.id,
                    name: createdFile.name,
                    type: createdFile.type,
                    file
                  }
                : item
            )
          )
          continue
        } else {
          file.type.includes("pdf")
            ? reader.readAsArrayBuffer(file)
            : reader.readAsText(file)
        }
      } else {
        throw new Error("Unsupported file type")
      }

      reader.onloadend = async function () {
        try {
          if (file.type.includes("image")) {
            const imageUrl = URL.createObjectURL(file)
            setNewMessageImages(prev => [
              ...prev,
              {
                messageId: "temp",
                path: "",
                base64: reader.result,
                url: imageUrl,
                file
              }
            ])
          } else {
            const createdFile = ragEnabled
              ? await createFileWithoutProcessing(
                  file,
                  {
                    user_id: profile.user_id,
                    description: "",
                    file_path: "",
                    name: file.name,
                    size: file.size,
                    tokens: 0,
                    type: simplifiedFileType
                  },
                  selectedWorkspace.id
                )
              : await createFile(
                  file,
                  {
                    user_id: profile.user_id,
                    description: "",
                    file_path: "",
                    name: file.name,
                    size: file.size,
                    tokens: 0,
                    type: simplifiedFileType
                  },
                  selectedWorkspace.id,
                  chatSettings.embeddingsProvider
                )

            setFiles(prev => [...prev, createdFile])
            setNewMessageFiles(prev =>
              prev.map(item =>
                item.id === "loading"
                  ? {
                      id: createdFile.id,
                      name: createdFile.name,
                      type: createdFile.type,
                      file
                    }
                  : item
              )
            )
          }
        } catch (error: any) {
          toast.error("Failed to upload. " + error?.message, {
            duration: 10000
          })
          setNewMessageImages(prev =>
            prev.filter(img => img.messageId !== "temp")
          )
          setNewMessageFiles(prev => prev.filter(f => f.id !== "loading"))
        }
      }
    }
  }

  return { handleSelectDeviceFile, filesToAccept }
}
