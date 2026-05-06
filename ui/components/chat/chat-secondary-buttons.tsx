import { useChatHandler } from "@/components/chat/chat-hooks/use-chat-handler"
import { ChatbotUIContext } from "@/context/context"
import { IconInfoCircle, IconMessagePlus } from "@tabler/icons-react"
import { FC, useContext } from "react"
import { useTranslation } from "react-i18next"
import { WithTooltip } from "../ui/with-tooltip"

interface ChatSecondaryButtonsProps {}

export const ChatSecondaryButtons: FC<ChatSecondaryButtonsProps> = ({}) => {
  const { selectedChat } = useContext(ChatbotUIContext)
  const { t } = useTranslation()

  const { handleNewChat } = useChatHandler()

  return (
    <>
      {selectedChat && (
        <>
          <WithTooltip
            delayDuration={200}
            display={
              <div>
                <div className="text-xl font-bold">
                  {t("Información del chat")}
                </div>

                <div className="mx-auto mt-2 max-w-xs space-y-2 sm:max-w-sm md:max-w-md lg:max-w-lg">
                  <div>
                    {t("Modelo")}: {selectedChat.model}
                  </div>
                  <div>
                    {t("Prompt")}: {selectedChat.prompt}
                  </div>

                  <div>
                    {t("Temperatura del modelo:")}: {selectedChat.temperature}
                  </div>
                  <div>
                    {t("Longitud del contexto:")}: {selectedChat.context_length}
                  </div>

                  <div>
                    {t("Contexto del perfil:")}:{" "}
                    {selectedChat.include_profile_context
                      ? t("Habilitado")
                      : t("Deshabilitado")}
                  </div>
                  <div>
                    {" "}
                    {t("Instrucciones del espacio de trabajo:")}:{" "}
                    {selectedChat.include_workspace_instructions
                      ? t("Habilitado")
                      : t("Deshabilitado")}
                  </div>

                  <div>
                    {t("Proveedor de Embeddings:")}:{" "}
                    {selectedChat.embeddings_provider}
                  </div>
                </div>
              </div>
            }
            trigger={
              <div className="mt-1">
                <IconInfoCircle
                  className="cursor-default hover:opacity-50"
                  size={24}
                />
              </div>
            }
          />

          <WithTooltip
            delayDuration={200}
            display={<div>{t("Empezar un nuevo chat")}</div>}
            trigger={
              <div className="mt-1">
                <IconMessagePlus
                  className="cursor-pointer hover:opacity-50"
                  size={24}
                  onClick={handleNewChat}
                />
              </div>
            }
          />
        </>
      )}
    </>
  )
}
