import { createClient } from "@/lib/supabase/server"
import { cookies } from "next/headers"
import { NextResponse } from "next/server"

export async function GET(request: Request) {
  const requestUrl = new URL(request.url)
  const code = requestUrl.searchParams.get("code")
  const next = requestUrl.searchParams.get("next")

  if (code) {
    const cookieStore = cookies()
    const supabase = createClient(cookieStore)
    const { error } = await supabase.auth.exchangeCodeForSession(code)

    if (!error) {
      // Después del login exitoso, verificar si el usuario tiene un workspace
      const {
        data: { user }
      } = await supabase.auth.getUser()

      if (user) {
        // Verificar si el usuario tiene un perfil
        const { data: profile } = await supabase
          .from("profiles")
          .select("*")
          .eq("user_id", user.id)
          .single()

        // Si no tiene perfil (nuevo usuario de Google), redirigir a setup
        if (!profile) {
          return NextResponse.redirect(requestUrl.origin + "/setup")
        }

        // Si no ha completado el onboarding, redirigir a setup
        if (!profile.has_onboarded) {
          return NextResponse.redirect(requestUrl.origin + "/setup")
        }

        // Buscar el workspace home del usuario
        const { data: homeWorkspace } = await supabase
          .from("workspaces")
          .select("*")
          .eq("user_id", user.id)
          .eq("is_home", true)
          .single()

        if (homeWorkspace) {
          return NextResponse.redirect(
            requestUrl.origin + `/${homeWorkspace.id}/chat`
          )
        } else {
          // Si no tiene workspace home, redirigir a setup para crearlo
          return NextResponse.redirect(requestUrl.origin + "/setup")
        }
      }
    } else {
      // Si hay error en el intercambio de código, redirigir a login con mensaje
      console.error("Auth error:", error)
      return NextResponse.redirect(
        requestUrl.origin +
          "/login?message=" +
          encodeURIComponent(error.message)
      )
    }
  }

  if (next) {
    return NextResponse.redirect(requestUrl.origin + next)
  } else {
    // Si no hay código ni next, redirigir a login
    return NextResponse.redirect(requestUrl.origin + "/login")
  }
}
