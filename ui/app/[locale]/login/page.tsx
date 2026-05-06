import { Brand } from "@/components/ui/brand"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { SubmitButton } from "@/components/ui/submit-button"
import { createClient } from "@/lib/supabase/server"
import { Database } from "@/supabase/types"
import { createServerClient } from "@supabase/ssr"
import { GoogleSVG } from "@/components/icons/google-svg"
import { get } from "@vercel/edge-config"
import { Metadata } from "next"
import { cookies, headers } from "next/headers"
import { redirect } from "next/navigation"

export const metadata: Metadata = {
  title: "Login"
}

export default async function Login({
  searchParams
}: {
  searchParams: { message: string }
}) {
  const cookieStore = cookies()
  const supabase = createServerClient<Database>(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        get(name: string) {
          return cookieStore.get(name)?.value
        }
      }
    }
  )
  const session = (await supabase.auth.getSession()).data.session

  if (session) {
    const { data: homeWorkspace, error } = await supabase
      .from("workspaces")
      .select("*")
      .eq("user_id", session.user.id)
      .eq("is_home", true)
      .single()

    if (!homeWorkspace) {
      throw new Error(error.message)
    }

    return redirect(`/${homeWorkspace.id}/chat`)
  }

  const signIn = async (formData: FormData) => {
    "use server"

    const email = formData.get("email") as string
    const password = formData.get("password") as string
    const cookieStore = cookies()
    const supabase = createClient(cookieStore)

    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password
    })

    if (error) {
      return redirect(`/login?message=${error.message}`)
    }

    const { data: homeWorkspace, error: homeWorkspaceError } = await supabase
      .from("workspaces")
      .select("*")
      .eq("user_id", data.user.id)
      .eq("is_home", true)
      .single()

    if (!homeWorkspace) {
      throw new Error(
        homeWorkspaceError?.message || "Ha ocurrido un error inesperado."
      )
    }

    return redirect(`/${homeWorkspace.id}/chat`)
  }

  const signInWithGoogle = async () => {
    "use server"

    const cookieStore = cookies()
    const supabase = createClient(cookieStore)
    const headersList = headers()

    // Obtener la URL base correcta para Vercel
    const host = headersList.get("host")
    const protocol =
      headersList.get("x-forwarded-proto") ||
      (host?.includes("localhost") ? "http" : "https")
    const origin = `${protocol}://${host}`

    console.log("OAuth redirect URL:", `${origin}/auth/callback`) // Para debug

    const { data, error } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: {
        redirectTo: `${origin}/auth/callback`
      }
    })

    if (error) {
      console.error("OAuth error:", error)
      return redirect(`/login?message=${error.message}`)
    }

    return redirect(data.url)
  }

  const getEnvVarOrEdgeConfigValue = async (name: string) => {
    "use server"
    if (process.env.EDGE_CONFIG) {
      return await get<string>(name)
    }

    return process.env[name]
  }

  const signUp = async (formData: FormData) => {
    "use server"

    const email = formData.get("email") as string
    const password = formData.get("password") as string

    const emailDomainWhitelistPatternsString = await getEnvVarOrEdgeConfigValue(
      "EMAIL_DOMAIN_WHITELIST"
    )
    const emailDomainWhitelist = emailDomainWhitelistPatternsString?.trim()
      ? emailDomainWhitelistPatternsString?.split(",")
      : []
    const emailWhitelistPatternsString =
      await getEnvVarOrEdgeConfigValue("EMAIL_WHITELIST")
    const emailWhitelist = emailWhitelistPatternsString?.trim()
      ? emailWhitelistPatternsString?.split(",")
      : []

    // If there are whitelist patterns, check if the email is allowed to sign up
    if (emailDomainWhitelist.length > 0 || emailWhitelist.length > 0) {
      const domainMatch = emailDomainWhitelist?.includes(email.split("@")[1])
      const emailMatch = emailWhitelist?.includes(email)
      if (!domainMatch && !emailMatch) {
        return redirect(
          `/login?message=Email ${email} no está permitido para registrarse.`
        )
      }
    }

    const cookieStore = cookies()
    const supabase = createClient(cookieStore)

    const { error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        // USE IF YOU WANT TO SEND EMAIL VERIFICATION, ALSO CHANGE TOML FILE
        // emailRedirectTo: `${origin}/auth/callback`
      }
    })

    if (error) {
      console.error(error)
      return redirect(`/login?message=${error.message}`)
    }

    return redirect("/setup")

    // USE IF YOU WANT TO SEND EMAIL VERIFICATION, ALSO CHANGE TOML FILE
    // return redirect("/login?message=Check email to continue sign in process")
  }

  const handleResetPassword = async (formData: FormData) => {
    "use server"

    const headersList = headers()
    const host = headersList.get("host")
    const protocol =
      headersList.get("x-forwarded-proto") ||
      (host?.includes("localhost") ? "http" : "https")
    const origin = `${protocol}://${host}`

    const email = formData.get("email") as string
    const cookieStore = cookies()
    const supabase = createClient(cookieStore)

    const { error } = await supabase.auth.resetPasswordForEmail(email, {
      redirectTo: `${origin}/auth/callback?next=/login/password`
    })

    if (error) {
      return redirect(`/login?message=${error.message}`)
    }

    return redirect(
      "/login?message=Revisa el email para restaurar la contraseña."
    )
  }

  return (
    <div className="flex w-full flex-1 flex-col justify-center gap-2 px-8 sm:max-w-md">
      <div className="animate-in text-foreground flex w-full flex-1 flex-col justify-center gap-2">
        <Brand />

        {/* Botón de Google OAuth */}
        <form action={signInWithGoogle}>
          <Button
            type="submit"
            variant="outline"
            className="group relative mb-4 flex w-full items-center justify-center space-x-3 overflow-hidden border-2 border-gray-200 bg-gradient-to-r from-white via-gray-50 to-white px-4 py-3 text-gray-700 shadow-lg transition-all duration-300 hover:border-blue-300 hover:from-blue-50 hover:via-white hover:to-blue-50 hover:shadow-xl focus:border-blue-400 focus:outline-none focus:ring-2 focus:ring-blue-200"
          >
            <div className="flex items-center space-x-3">
              <GoogleSVG width={24} height={24} />
              <span className="font-medium">Continuar con Google</span>
            </div>
            <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-20"></div>
          </Button>
        </form>

        <div className="relative mb-4">
          <div className="absolute inset-0 flex items-center">
            <span className="w-full border-t" />
          </div>
          <div className="relative flex justify-center text-xs uppercase">
            <span className="bg-background text-muted-foreground px-2">
              O continúa con email
            </span>
          </div>
        </div>

        {/* Formulario de email/password */}
        <form action={signIn}>
          <Label className="text-md" htmlFor="email">
            Email
          </Label>
          <Input
            className="mb-3 rounded-md border bg-inherit px-4 py-2"
            name="email"
            placeholder="you@example.com"
            required
          />

          <Label className="text-md" htmlFor="password">
            Contraseña
          </Label>
          <Input
            className="mb-6 rounded-md border bg-inherit px-4 py-2"
            type="password"
            name="password"
            placeholder="••••••••"
          />

          <SubmitButton className="mb-2 rounded-md bg-blue-700 px-4 py-2 text-white">
            Iniciar sesión
          </SubmitButton>

          <SubmitButton
            formAction={signUp}
            className="border-foreground/20 mb-2 rounded-md border px-4 py-2"
          >
            Registrarse
          </SubmitButton>

          <div className="text-muted-foreground mt-1 flex justify-center text-sm">
            <span className="mr-1">¿Olvidaste tu contraseña?</span>
            <button
              formAction={handleResetPassword}
              className="text-primary ml-1 underline hover:opacity-80"
            >
              Restablecer
            </button>
          </div>
        </form>

        {searchParams?.message && (
          <p className="bg-foreground/10 text-foreground mt-4 p-4 text-center">
            {searchParams.message}
          </p>
        )}
      </div>
    </div>
  )
}
