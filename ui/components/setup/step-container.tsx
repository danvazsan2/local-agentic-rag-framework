import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
  CardTitle
} from "@/components/ui/card"
import { FC, useRef } from "react"

export const SETUP_STEP_COUNT = 2

interface StepContainerProps {
  stepDescription: string
  stepNum: number
  stepTitle: string
  onShouldProceed: (shouldProceed: boolean) => void
  children?: React.ReactNode
  showBackButton?: boolean
  showNextButton?: boolean
}

export const StepContainer: FC<StepContainerProps> = ({
  stepDescription,
  stepNum,
  stepTitle,
  onShouldProceed,
  children,
  showBackButton = false,
  showNextButton = true
}) => {
  const buttonRef = useRef<HTMLButtonElement>(null)

  const handleKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      if (buttonRef.current) {
        buttonRef.current.click()
      }
    }
  }

  return (
    <Card
      className="mx-auto max-h-[calc(100vh-60px)] w-full max-w-[600px] overflow-auto"
      onKeyDown={handleKeyDown}
    >
      <CardHeader className="px-4 sm:px-6">
        <CardTitle className="flex flex-col gap-2 text-xl sm:flex-row sm:justify-between sm:text-2xl">
          <div>{stepTitle}</div>

          <div className="text-muted-foreground text-sm font-normal">
            {stepNum} / {SETUP_STEP_COUNT}
          </div>
        </CardTitle>

        <CardDescription className="text-sm sm:text-base">
          {stepDescription}
        </CardDescription>
      </CardHeader>

      <CardContent className="space-y-4 px-4 sm:px-6">{children}</CardContent>

      <CardFooter className="flex flex-col justify-between gap-4 px-4 sm:flex-row sm:px-6">
        <div className="w-full sm:w-auto">
          {showBackButton && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => onShouldProceed(false)}
              className="w-full sm:w-auto"
            >
              Atrás
            </Button>
          )}
        </div>

        <div className="w-full sm:w-auto">
          {showNextButton && (
            <Button
              ref={buttonRef}
              size="sm"
              onClick={() => onShouldProceed(true)}
              className="w-full sm:w-auto"
            >
              Siguiente
            </Button>
          )}
        </div>
      </CardFooter>
    </Card>
  )
}
