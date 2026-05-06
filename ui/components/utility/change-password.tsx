"use client"

import { supabase } from "@/lib/supabase/browser-client"
import { useRouter } from "next/navigation"
import { FC, useState } from "react"
import { useTranslation } from "react-i18next"
import { Button } from "../ui/button"
import {
  Dialog,
  DialogContent,
  DialogFooter,
  DialogHeader,
  DialogTitle
} from "../ui/dialog"
import { Input } from "../ui/input"
import { toast } from "sonner"

interface ChangePasswordProps {}

export const ChangePassword: FC<ChangePasswordProps> = () => {
  const router = useRouter()
  const { t } = useTranslation()

  const [newPassword, setNewPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")

  const handleResetPassword = async () => {
    if (!newPassword)
      return toast.info(t("Por favor, introduzca su nueva contraseña."))

    await supabase.auth.updateUser({ password: newPassword })

    toast.success(t("Contraseña cambiada con éxito."))

    return router.push("/login")
  }

  return (
    <Dialog open={true}>
      <DialogContent className="h-[240px] w-[400px] p-4">
        <DialogHeader>
          <DialogTitle>{t("Cambiar Contraseña")}</DialogTitle>
        </DialogHeader>

        <Input
          id="password"
          placeholder={t("Nueva Contraseña")}
          type="password"
          value={newPassword}
          onChange={e => setNewPassword(e.target.value)}
        />

        <Input
          id="confirmPassword"
          placeholder={t("Confirmar Nueva Contraseña")}
          type="password"
          value={confirmPassword}
          onChange={e => setConfirmPassword(e.target.value)}
        />

        <DialogFooter>
          <Button onClick={handleResetPassword}>{t("Confirmar Cambio")}</Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
