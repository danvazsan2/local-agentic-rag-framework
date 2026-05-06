import { FC } from "react"

interface FinishStepProps {
  displayName: string
}

export const FinishStep: FC<FinishStepProps> = ({ displayName }) => {
  return (
    <div className="space-y-4 text-center sm:text-left">
      <div className="text-base sm:text-lg">
        Bienvenido a TFG Chatbot
        {displayName.length > 0 ? `, ${displayName.split(" ")[0]}` : null}!
      </div>

      <div className="text-muted-foreground text-sm sm:text-base">
        Haz clic en siguiente para comenzar a chatear.
      </div>
    </div>
  )
}
