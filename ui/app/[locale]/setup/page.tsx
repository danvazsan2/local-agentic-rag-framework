"use client"

import { ChatbotUIContext } from "@/context/context"
import { getProfileByUserId, updateProfile } from "@/db/profile"
import {
  getHomeWorkspaceByUserId,
  getWorkspacesByUserId
} from "@/db/workspaces"
import {
  fetchHostedModels,
  fetchOllamaModels,
  fetchOpenRouterModels
} from "@/lib/models/fetch-models"
import { supabase } from "@/lib/supabase/browser-client"
import { TablesUpdate, TablesInsert } from "@/supabase/types"
import { useRouter } from "next/navigation"
import { useContext, useEffect, useState } from "react"
import { APIStep } from "../../../components/setup/api-step"
import { FinishStep } from "../../../components/setup/finish-step"
import { ProfileStep } from "../../../components/setup/profile-step"
import {
  SETUP_STEP_COUNT,
  StepContainer
} from "../../../components/setup/step-container"

export default function SetupPage() {
  const {
    profile,
    setProfile,
    setWorkspaces,
    setSelectedWorkspace,
    setEnvKeyMap,
    setAvailableHostedModels,
    setAvailableLocalModels,
    setAvailableOpenRouterModels
  } = useContext(ChatbotUIContext)

  const router = useRouter()

  const [loading, setLoading] = useState(true)

  const [currentStep, setCurrentStep] = useState(1)

  // Profile Step
  const [displayName, setDisplayName] = useState("")
  const [username, setUsername] = useState(profile?.username || "")
  const [usernameAvailable, setUsernameAvailable] = useState(true)

  // API Step
  const [useAzureOpenai, setUseAzureOpenai] = useState(false)
  const [openaiAPIKey, setOpenaiAPIKey] = useState("")
  const [openaiOrgID, setOpenaiOrgID] = useState("")
  const [azureOpenaiAPIKey, setAzureOpenaiAPIKey] = useState("")
  const [azureOpenaiEndpoint, setAzureOpenaiEndpoint] = useState("")
  const [azureOpenai35TurboID, setAzureOpenai35TurboID] = useState("")
  const [azureOpenai45TurboID, setAzureOpenai45TurboID] = useState("")
  const [azureOpenai45VisionID, setAzureOpenai45VisionID] = useState("")
  const [azureOpenaiEmbeddingsID, setAzureOpenaiEmbeddingsID] = useState("")
  const [anthropicAPIKey, setAnthropicAPIKey] = useState("")
  const [googleGeminiAPIKey, setGoogleGeminiAPIKey] = useState("")
  const [mistralAPIKey, setMistralAPIKey] = useState("")
  const [groqAPIKey, setGroqAPIKey] = useState("")
  const [perplexityAPIKey, setPerplexityAPIKey] = useState("")
  const [openrouterAPIKey, setOpenrouterAPIKey] = useState("")

  useEffect(() => {
    ;(async () => {
      const session = (await supabase.auth.getSession()).data.session

      if (!session) {
        return router.push("/login")
      } else {
        const user = session.user

        const profile = await getProfileByUserId(user.id)

        // Cargar modelos de Ollama
        const localModels = await fetchOllamaModels()
        if (localModels && localModels.length > 0) {
          setAvailableLocalModels(localModels)
        }

        if (!profile) {
          // Nuevo usuario de Google OAuth - usar datos del usuario para prellenar
          setDisplayName(user.user_metadata?.full_name || "")
          setUsername(
            user.user_metadata?.preferred_username ||
              user.email?.split("@")[0] ||
              ""
          )
          setLoading(false)
          return
        }

        setProfile(profile)
        setUsername(profile.username)
        setDisplayName(
          profile.display_name || user.user_metadata?.full_name || ""
        )

        if (!profile.has_onboarded) {
          setLoading(false)
        } else {
          const data = await fetchHostedModels(profile)

          if (!data) return

          setEnvKeyMap(data.envKeyMap)
          setAvailableHostedModels(data.hostedModels)

          if (profile["openrouter_api_key"] || data.envKeyMap["openrouter"]) {
            const openRouterModels = await fetchOpenRouterModels()
            if (!openRouterModels) return
            setAvailableOpenRouterModels(openRouterModels)
          }

          const homeWorkspaceId = await getHomeWorkspaceByUserId(
            session.user.id
          )
          return router.push(`/${homeWorkspaceId}/chat`)
        }
      }
    })()
  }, [])

  const handleShouldProceed = (proceed: boolean) => {
    if (proceed) {
      if (currentStep === SETUP_STEP_COUNT) {
        handleSaveSetupSetting()
      } else {
        setCurrentStep(currentStep + 1)
      }
    } else {
      setCurrentStep(currentStep - 1)
    }
  }

  const handleSaveSetupSetting = async () => {
    const session = (await supabase.auth.getSession()).data.session
    if (!session) {
      return router.push("/login")
    }

    const user = session.user
    let profile = await getProfileByUserId(user.id)

    let updateProfilePayload:
      | TablesUpdate<"profiles">
      | TablesInsert<"profiles">

    if (!profile) {
      // Crear nuevo perfil para usuario de Google OAuth
      updateProfilePayload = {
        user_id: user.id,
        has_onboarded: true,
        display_name: displayName,
        username,
        openai_api_key: openaiAPIKey || null,
        openai_organization_id: openaiOrgID || null,
        anthropic_api_key: anthropicAPIKey || null,
        google_gemini_api_key: googleGeminiAPIKey || null,
        mistral_api_key: mistralAPIKey || null,
        groq_api_key: groqAPIKey || null,
        perplexity_api_key: perplexityAPIKey || null,
        openrouter_api_key: openrouterAPIKey || null,
        use_azure_openai: useAzureOpenai,
        azure_openai_api_key: azureOpenaiAPIKey || null,
        azure_openai_endpoint: azureOpenaiEndpoint || null,
        azure_openai_35_turbo_id: azureOpenai35TurboID || null,
        azure_openai_45_turbo_id: azureOpenai45TurboID || null,
        azure_openai_45_vision_id: azureOpenai45VisionID || null,
        azure_openai_embeddings_id: azureOpenaiEmbeddingsID || null,
        // Valores por defecto para campos requeridos
        bio: "",
        profile_context: "",
        image_url: user.user_metadata?.avatar_url || "",
        image_path: ""
      }

      // Crear el perfil usando supabase directamente
      const { data: newProfile, error } = await supabase
        .from("profiles")
        .insert(updateProfilePayload as TablesInsert<"profiles">)
        .select()
        .single()

      if (error) {
        console.error("Error creating profile:", error)
        return
      }

      profile = newProfile
    } else {
      // Actualizar perfil existente
      updateProfilePayload = {
        ...profile,
        has_onboarded: true,
        display_name: displayName,
        username,
        openai_api_key: openaiAPIKey,
        openai_organization_id: openaiOrgID,
        anthropic_api_key: anthropicAPIKey,
        google_gemini_api_key: googleGeminiAPIKey,
        mistral_api_key: mistralAPIKey,
        groq_api_key: groqAPIKey,
        perplexity_api_key: perplexityAPIKey,
        openrouter_api_key: openrouterAPIKey,
        use_azure_openai: useAzureOpenai,
        azure_openai_api_key: azureOpenaiAPIKey,
        azure_openai_endpoint: azureOpenaiEndpoint,
        azure_openai_35_turbo_id: azureOpenai35TurboID,
        azure_openai_45_turbo_id: azureOpenai45TurboID,
        azure_openai_45_vision_id: azureOpenai45VisionID,
        azure_openai_embeddings_id: azureOpenaiEmbeddingsID
      }

      const updatedProfile = await updateProfile(
        profile.id,
        updateProfilePayload
      )
      profile = updatedProfile
    }

    setProfile(profile)

    // Initialize hosted models data after profile creation
    const hostedModelRes = await fetchHostedModels(profile)
    if (hostedModelRes) {
      setEnvKeyMap(hostedModelRes.envKeyMap)
      setAvailableHostedModels(hostedModelRes.hostedModels)

      if (
        profile["openrouter_api_key"] ||
        hostedModelRes.envKeyMap["openrouter"]
      ) {
        const openRouterModels = await fetchOpenRouterModels()
        if (openRouterModels) {
          setAvailableOpenRouterModels(openRouterModels)
        }
      }
    }

    const workspaces = await getWorkspacesByUserId(profile.user_id)
    const homeWorkspace = workspaces.find(w => w.is_home)

    // There will always be a home workspace
    setSelectedWorkspace(homeWorkspace!)
    setWorkspaces(workspaces)

    return router.push(`/${homeWorkspace?.id}/chat`)
  }

  const renderStep = (stepNum: number) => {
    switch (stepNum) {
      // Profile Step
      case 1:
        return (
          <StepContainer
            stepDescription="Vamos a crear tu perfil."
            stepNum={currentStep}
            stepTitle="Bienvenido a TFG Chatbot"
            onShouldProceed={handleShouldProceed}
            showNextButton={!!(username && usernameAvailable)}
            showBackButton={false}
          >
            <ProfileStep
              username={username}
              usernameAvailable={usernameAvailable}
              displayName={displayName}
              onUsernameAvailableChange={setUsernameAvailable}
              onUsernameChange={setUsername}
              onDisplayNameChange={setDisplayName}
            />
          </StepContainer>
        )

      // Finish Step (formerly step 3, now step 2 - API step is hidden)
      case 2:
        return (
          <StepContainer
            stepDescription="¡Ya estás listo!"
            stepNum={currentStep}
            stepTitle="Configuración completa"
            onShouldProceed={handleShouldProceed}
            showNextButton={true}
            showBackButton={true}
          >
            <FinishStep displayName={displayName} />
          </StepContainer>
        )
      default:
        return null
    }
  }

  if (loading) {
    return null
  }

  return (
    <div className="flex h-full items-center justify-center p-4 sm:p-6 lg:p-8">
      {renderStep(currentStep)}
    </div>
  )
}
