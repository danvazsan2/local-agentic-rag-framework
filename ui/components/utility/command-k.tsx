import { ChatbotUIContext } from "@/context/context"
import useHotkey from "@/lib/hooks/use-hotkey"
import { IconLoader2, IconSend } from "@tabler/icons-react"
import { FC, useContext, useState } from "react"
import { Dialog, DialogContent } from "../ui/dialog"
import { TextareaAutosize } from "../ui/textarea-autosize"

interface CommandKProps {}

export const CommandK: FC<CommandKProps> = ({}) => {
  useHotkey("k", () => setIsOpen(prevState => !prevState))

  const { profile } = useContext(ChatbotUIContext)

  const [isOpen, setIsOpen] = useState(false)
  const [value, setValue] = useState("")
  const [loading, setLoading] = useState(false)
  const [content, setContent] = useState("")

  const handleCommandK = async () => {
    setLoading(true)

    const response = await fetch("/api/command", {
      method: "POST",
      body: JSON.stringify({
        input: value
      })
    })

    const data = await response.json()

    setContent(data.content)
    setLoading(false)
    setValue("")
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleCommandK()
    }
  }

  if (!profile) return null

  return (
    isOpen && (
      <Dialog open={isOpen} onOpenChange={setIsOpen}>
        <DialogContent onKeyDown={handleKeyDown}>
          {profile.openai_api_key ? (
            <div className="space-y-2">
              <div>{content}</div>

              <div>Activar el modo oscuro.</div>
              <div>Buscar mi chat SQL</div>
              <div>Necesito un nuevo asistente</div>
              <div>Iniciar un chat con mi archivo de resoluciones 2024</div>

              <div className="border-input relative flex min-h-[50px] w-full items-center justify-center rounded-xl border-2">
                <TextareaAutosize
                  className="ring-offset-background placeholder:text-muted-foreground focus-visible:ring-ring text-md flex w-full resize-none rounded-md border-none bg-transparent px-3 py-2 pr-14 focus-visible:outline-none disabled:cursor-not-allowed disabled:opacity-50"
                  placeholder="crea un prompt para escribir código SQL"
                  value={value}
                  onValueChange={setValue}
                />
                {loading ? (
                  <IconLoader2
                    className="absolute bottom-[8px] right-3 animate-spin cursor-pointer rounded p-1 hover:opacity-50"
                    size={30}
                  />
                ) : (
                  <IconSend
                    className="bg-primary text-secondary absolute bottom-[8px] right-3 cursor-pointer rounded p-1 hover:opacity-50"
                    onClick={handleCommandK}
                    size={30}
                  />
                )}
              </div>
            </div>
          ) : (
            <div>
              Agrega tu clave API de OpenAI en la configuración para desbloquear
              CMD+K.
            </div>
          )}
        </DialogContent>
      </Dialog>
    )
  )
}
