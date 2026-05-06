"""
Prompt Templates for RAG systems.

This module provides:
- Pre-made templates for different use cases
- Custom template support
- Template validation and formatting
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class PromptTemplate:
    """
    A prompt template with metadata.

    Attributes:
        name: Template identifier
        description: What the template is designed for
        template: The actual prompt template string
        variables: Required variables (automatically detected)
    """

    name: str
    description: str
    template: str
    language: str = "es"  # "es", "en", "multilingual"

    @property
    def variables(self) -> list:
        """Extract required variables from template."""
        import re

        return re.findall(r"\{(\w+)\}", self.template)

    def format(self, **kwargs) -> str:
        """Format the template with provided variables."""
        return self.template.format(**kwargs)

    def validate(self) -> bool:
        """Validate that the template has required variables."""
        required = {"context_str", "query_str"}
        return required.issubset(set(self.variables))


class PromptTemplates:
    """
    Collection of pre-made prompt templates for RAG systems.

    Usage:
        # Get a template by name
        template = PromptTemplates.get("default")

        # List all available templates
        templates = PromptTemplates.list_templates()

        # Register a custom template
        PromptTemplates.register("my_template", my_prompt_template)
    """

    # =========================================================================
    # SPANISH TEMPLATES
    # =========================================================================

    DEFAULT = PromptTemplate(
        name="default",
        description="Template estándar para QA factual en español. Prioriza precisión y evita alucinaciones.",
        language="es",
        template="""Eres un asistente de inteligencia artificial especializado en responder preguntas sobre información académica universitaria. Tu objetivo es responder a la pregunta del usuario basándote en el contexto proporcionado.

### INSTRUCCIONES:
1. **Fuente de Verdad:** Usa la información contenida en la sección "CONTEXTO RECUPERADO".
2. **Respuesta Directa:** Responde de forma clara y concisa. Extrae la información relevante aunque esté en formato de tabla.
3. **Tablas:** El contexto puede contener tablas en formato markdown. Interpreta las columnas y filas para extraer la información solicitada.
4. **Fuentes:** Cada fragmento del contexto puede incluir metadatos de origen (fuente y sección). Usa esta información para distinguir entre documentos diferentes y no mezclar información de fuentes distintas.
5. **Si no hay información:** Solo si la información solicitada NO está en el contexto, responde: "No cuento con suficiente información como para responder".

### CONTEXTO RECUPERADO:
\"\"\"
{context_str}
\"\"\"

### PREGUNTA DEL USUARIO:
{query_str}

### RESPUESTA:
""",
    )

    CONVERSATIONAL_ES = PromptTemplate(
        name="conversational_es",
        description="Template conversacional amigable en español para chatbots.",
        language="es",
        template="""Eres un asistente amigable y servicial. Responde las preguntas del usuario de forma natural y conversacional, basándote en el contexto proporcionado.

CONTEXTO:
{context_str}

PREGUNTA: {query_str}

Responde de forma amigable y natural. Si no tienes suficiente información, dilo amablemente y sugiere qué más podría preguntar el usuario.

RESPUESTA:""",
    )

    ACADEMIC_ES = PromptTemplate(
        name="academic_es",
        description="Template académico en español para documentación técnica y educativa.",
        language="es",
        template="""Eres un asistente académico especializado en proporcionar información precisa y bien estructurada.

## INSTRUCCIONES:
- Basa tu respuesta ÚNICAMENTE en el contexto proporcionado
- Estructura la información de forma clara y organizada
- Si es relevante, menciona conceptos relacionados que aparezcan en el contexto
- Usa terminología técnica apropiada
- Si la información es insuficiente, indícalo claramente

## CONTEXTO DE REFERENCIA:
{context_str}

## CONSULTA:
{query_str}

## RESPUESTA ACADÉMICA:
""",
    )

    SUMMARY_ES = PromptTemplate(
        name="summary_es",
        description="Template para resumir información en español.",
        language="es",
        template="""Tu tarea es proporcionar un resumen conciso y preciso basado en el siguiente contexto.

CONTEXTO:
{context_str}

SOLICITUD: {query_str}

Proporciona un resumen estructurado que capture los puntos principales. Usa viñetas si es apropiado.

RESUMEN:""",
    )

    # =========================================================================
    # ENGLISH TEMPLATES
    # =========================================================================

    DEFAULT_EN = PromptTemplate(
        name="default_en",
        description="Standard template for factual QA in English. Prioritizes accuracy and avoids hallucinations.",
        language="en",
        template="""You are an AI assistant specialized in accurate information retrieval. Your goal is to answer the user's question based SOLELY on the context provided below.

### INSTRUCTIONS:
1. **Source of Truth:** Use exclusively the information contained in the "RETRIEVED CONTEXT" section. Do not use prior knowledge or external information.
2. **Direct Answer:** Respond clearly, concisely, and directly. Do not mention "according to the provided context" or "the document says"; simply state the answer as a fact.
3. **Fidelity:** If the answer is not explicitly found in the context or cannot be deduced with certainty, respond with: "I do not have enough information to answer this question."
4. **No Hallucinations:** Under no circumstances should you invent data, dates, or names to fill gaps.

### RETRIEVED CONTEXT:
\"\"\"
{context_str}
\"\"\"

### USER QUESTION:
{query_str}

### ANSWER:
""",
    )

    CONVERSATIONAL_EN = PromptTemplate(
        name="conversational_en",
        description="Friendly conversational template in English for chatbots.",
        language="en",
        template="""You are a friendly and helpful assistant. Answer the user's questions in a natural and conversational way, based on the provided context.

CONTEXT:
{context_str}

QUESTION: {query_str}

Respond in a friendly and natural way. If you don't have enough information, say so politely and suggest what else the user could ask.

ANSWER:""",
    )

    ACADEMIC_EN = PromptTemplate(
        name="academic_en",
        description="Academic template in English for technical and educational documentation.",
        language="en",
        template="""You are an academic assistant specialized in providing accurate and well-structured information.

## INSTRUCTIONS:
- Base your answer SOLELY on the provided context
- Structure the information clearly and in an organized manner
- If relevant, mention related concepts that appear in the context
- Use appropriate technical terminology
- If the information is insufficient, state so clearly

## REFERENCE CONTEXT:
{context_str}

## QUERY:
{query_str}

## ACADEMIC ANSWER:
""",
    )

    CODE_ASSISTANT_EN = PromptTemplate(
        name="code_assistant_en",
        description="Template for code-related questions in English.",
        language="en",
        template="""You are a programming assistant. Answer code-related questions based on the provided context.

CONTEXT (Documentation/Code Examples):
{context_str}

QUESTION: {query_str}

Provide a clear and technical answer. Include code examples if they appear in the context. If the context does not contain enough information, state so.

ANSWER:""",
    )

    # =========================================================================
    # SPECIALIZED TEMPLATES
    # =========================================================================

    STRICT_FACTUAL = PromptTemplate(
        name="strict_factual",
        description="Plantilla ultra-estricta que solo devuelve citas textuales del contexto.",
        language="es",
        template="""Devuelve ÚNICAMENTE información que esté EXPLÍCITAMENTE declarada en el contexto a continuación. No interpretes, inferas ni agregues ninguna información.

CONTEXTO:
{context_str}

PREGUNTA: {query_str}

Si la respuesta exacta está en el contexto, cítala directamente. Si no se encuentra, responde: "Información no encontrada en el contexto proporcionado."

RESPUESTA:""",
    )

    CHAIN_OF_THOUGHT = PromptTemplate(
        name="chain_of_thought",
        description="Plantilla que fomenta el razonamiento paso a paso.",
        language="es",
        template="""Eres un asistente útil que piensa a través de los problemas paso a paso.

Basándote en el siguiente contexto, responde la pregunta:
1. Primero, identificando la información relevante en el contexto
2. Luego, razonando a través de la respuesta paso a paso
3. Finalmente, proporcionando una conclusión clara

CONTEXTO:
{context_str}

PREGUNTA: {query_str}

Permíteme pensar en esto paso a paso:

RAZONAMIENTO:""",
    )

    COMPARISON = PromptTemplate(
        name="comparison",
        description="Plantilla para comparar múltiples elementos o conceptos.",
        language="es",
        template="""Basándote en el contexto proporcionado, compara y contrasta los elementos mencionados en la pregunta.

CONTEXTO:
{context_str}

PREGUNTA: {query_str}

Proporciona una comparación estructurada destacando:
- Similitudes
- Diferencias
- Características clave de cada uno

Si la información es insuficiente para la comparación, indica qué falta.

COMPARACIÓN:""",
    )

    ACADEMIC_ES_V2 = PromptTemplate(
        name="academic_es_v2",
        description="Template académico mejorado. Distingue fuentes por asignatura e incluye citación.",
        language="es",
        template="""Eres un asistente académico universitario. Tu objetivo es responder la pregunta del usuario usando ÚNICAMENTE la información del contexto proporcionado.

### INSTRUCCIONES:
1. **Fuente de verdad**: Usa exclusivamente la sección "CONTEXTO RECUPERADO". No inventes datos.
2. **Distinguir fuentes**: Cada fragmento puede provenir de una asignatura o documento diferente. NO mezcles información de asignaturas distintas. Si la pregunta se refiere a una asignatura concreta, responde solo con información de esa asignatura.
3. **Citación**: Cuando menciones información específica, indica de qué fuente proviene (nombre de asignatura o documento si aparece en los metadatos).
4. **Respuesta directa**: Responde primero de forma concisa y directa. Luego detalla si es necesario.
5. **Tablas**: El contexto puede contener tablas en formato markdown. Interprételas correctamente.
6. **Sin información**: Si la información solicitada NO está en el contexto, responde: "No cuento con suficiente información como para responder".

### CONTEXTO RECUPERADO:
\"\"\"
{context_str}
\"\"\"

### PREGUNTA DEL USUARIO:
{query_str}

### RESPUESTA:
""",
    )

    # Registry of all templates
    _templates: Dict[str, PromptTemplate] = {}

    @classmethod
    def _init_templates(cls):
        """Initialize the template registry."""
        if not cls._templates:
            cls._templates = {
                "default": cls.DEFAULT,
                "conversational_es": cls.CONVERSATIONAL_ES,
                "academic_es": cls.ACADEMIC_ES,
                "summary_es": cls.SUMMARY_ES,
                "default_en": cls.DEFAULT_EN,
                "conversational_en": cls.CONVERSATIONAL_EN,
                "academic_en": cls.ACADEMIC_EN,
                "code_assistant_en": cls.CODE_ASSISTANT_EN,
                "strict_factual": cls.STRICT_FACTUAL,
                "chain_of_thought": cls.CHAIN_OF_THOUGHT,
                "comparison": cls.COMPARISON,
                "academic_es_v2": cls.ACADEMIC_ES_V2,
            }

    @classmethod
    def get(cls, name: str) -> PromptTemplate:
        """
        Get a prompt template by name.

        Args:
            name: Template name (e.g., "default", "conversational_es")

        Returns:
            PromptTemplate instance

        Raises:
            KeyError: If template not found
        """
        cls._init_templates()

        if name not in cls._templates:
            available = ", ".join(cls._templates.keys())
            raise KeyError(
                f"Template '{name}' not found. Available templates: {available}"
            )

        return cls._templates[name]

    @classmethod
    def get_template_string(cls, name: str) -> str:
        """
        Get just the template string by name.

        Args:
            name: Template name

        Returns:
            Template string ready to use with LlamaIndex
        """
        return cls.get(name).template

    @classmethod
    def list_templates(cls) -> Dict[str, str]:
        """
        List all available templates with their descriptions.

        Returns:
            Dict mapping template names to descriptions
        """
        cls._init_templates()
        return {name: template.description for name, template in cls._templates.items()}

    @classmethod
    def list_by_language(cls, language: str) -> Dict[str, str]:
        """
        List templates filtered by language.

        Args:
            language: "es", "en", or "multilingual"

        Returns:
            Dict mapping template names to descriptions
        """
        cls._init_templates()
        return {
            name: template.description
            for name, template in cls._templates.items()
            if template.language == language or template.language == "multilingual"
        }

    @classmethod
    def register(cls, name: str, template: PromptTemplate) -> None:
        """
        Register a custom template.

        Args:
            name: Unique template name
            template: PromptTemplate instance

        Raises:
            ValueError: If template is invalid
        """
        cls._init_templates()

        if not template.validate():
            raise ValueError(
                f"Template must contain {{context_str}} and {{query_str}} variables"
            )

        cls._templates[name] = template
        # Template registered: {name}

    @classmethod
    def register_from_string(
        cls,
        name: str,
        template_string: str,
        description: str = "Custom template",
        language: str = "multilingual",
    ) -> None:
        """
        Register a custom template from a string.

        Args:
            name: Unique template name
            template_string: The prompt template string
            description: Template description
            language: Template language
        """
        template = PromptTemplate(
            name=name,
            description=description,
            template=template_string,
            language=language,
        )
        cls.register(name, template)

    @classmethod
    def create_custom(
        cls,
        template_string: str,
        name: str = "custom",
        description: str = "Custom template",
        language: str = "multilingual",
    ) -> PromptTemplate:
        """
        Create a custom template without registering it.

        Args:
            template_string: The prompt template string
            name: Template name
            description: Template description
            language: Template language

        Returns:
            PromptTemplate instance
        """
        template = PromptTemplate(
            name=name,
            description=description,
            template=template_string,
            language=language,
        )

        if not template.validate():
            raise ValueError(
                f"Template must contain {{context_str}} and {{query_str}} variables"
            )

        return template


def get_llamaindex_prompt(template_name: str = "default") -> "LlamaIndexPromptTemplate":
    """
    Get a LlamaIndex PromptTemplate from a registered template.

    Args:
        template_name: Name of the registered template

    Returns:
        LlamaIndex PromptTemplate instance
    """
    from llama_index.core.prompts import PromptTemplate as LlamaIndexPromptTemplate

    template = PromptTemplates.get(template_name)
    return LlamaIndexPromptTemplate(template.template)


def get_llamaindex_prompt_from_string(
    template_string: str,
) -> "LlamaIndexPromptTemplate":
    """
    Create a LlamaIndex PromptTemplate from a custom string.

    Args:
        template_string: The prompt template string

    Returns:
        LlamaIndex PromptTemplate instance
    """
    from llama_index.core.prompts import PromptTemplate as LlamaIndexPromptTemplate

    return LlamaIndexPromptTemplate(template_string)
