"""
SQL Agent Module.

Translates natural language questions into SQL queries using LLM,
executes them safely, and formats results for synthesis.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
import logging
from datetime import datetime

from rag_framework.config.models import RAGConfig, SQLConfig
from rag_framework.exceptions import (
    SQLGenerationError,
    SQLQueryError,
    SQLSecurityError,
    SQLConnectionError,
)
from rag_framework.sql.schema import SchemaLoader, TableInfo
from rag_framework.sql.validator import QueryValidator
from rag_framework.sql.executor import QueryExecutor, QueryResult

logger = logging.getLogger(__name__)


@dataclass
class SQLQueryResult:
    """
    Complete result of SQL agent query processing.

    Contains both the execution results and metadata needed
    for response synthesis.
    """

    success: bool
    query: str = ""  # Generated SQL query
    result: Optional[QueryResult] = None
    formatted_result: str = ""  # For LLM synthesis
    error: Optional[str] = None

    # Metadata
    generation_attempts: int = 0
    total_time_ms: float = 0.0
    query_relaxed: bool = False  # True if LIKE conditions were relaxed after 0 rows

    def get_context_for_llm(self) -> str:
        """
        Get formatted context string for LLM synthesis.

        Returns a string that can be included in the LLM prompt
        to provide database query results as context.
        """
        if not self.success:
            return f"Database query failed: {self.error}"

        return f"""
DATABASE QUERY RESULTS
======================
Query: {self.query}

Results:
{self.formatted_result}
"""


class SQLAgent:
    """
    Agent that translates natural language to SQL and executes queries.

    The agent uses an LLM to generate SQL from natural language,
    validates the generated SQL for safety, executes it, and
    formats results for downstream synthesis.

    Key features:
    - Schema-aware SQL generation
    - Multi-attempt generation with error feedback
    - Strict security validation
    - Token-efficient result formatting
    """

    # Prompt template for SQL generation
    # CHANGED: Enriched with schema descriptions, stricter rules, analysis step,
    # CAST guidance, and faithfulness constraint
    SQL_GENERATION_PROMPT = """Eres un experto en SQL. Genera una consulta SQL para responder a la pregunta del usuario basándote en el esquema de base de datos proporcionado.

ESQUEMA DE BASE DE DATOS:
{schema}

REGLAS ESTRICTAS:
1. Solo usa sentencias SELECT (obligatorio)
2. Solo usa tablas y columnas del esquema anterior (verifica nombres exactos)
3. Usa JOINs apropiados cuando sea necesario para unir información relacionada
4. Usa alias descriptivos para mejorar la legibilidad
5. Para agregaciones (COUNT, SUM, AVG), usa nombres de columna claros (ej: COUNT(*) AS total)
6. Si la pregunta implica filtros temporales o condicionales, usa WHERE apropiado
7. Si una columna es de tipo TEXT pero almacena valores numéricos (mira los valores de ejemplo), usa CAST(columna AS REAL) o CAST(columna AS INTEGER) para comparaciones numéricas (>, <, >=, <=, BETWEEN, ORDER BY numérico)
8. Responde EXACTAMENTE lo que se pregunta. NO añadas filtros, condiciones ni restricciones que el usuario no haya mencionado explícitamente. Si la pregunta es general, no limites por categorías, cursos u otros atributos salvo que se pidan
9. Si la pregunta pide una distribución o conteo por categorías, usa GROUP BY correctamente y asegúrate de que los alias de columna reflejen fielmente cada categoría
10. NO añadas cláusula LIMIT a las consultas. Devuelve siempre todos los resultados.

CONTEXTO DEL ESQUEMA:
{schema_descriptions}

PREGUNTA DEL USUARIO: {question}

ANÁLISIS ANTES DE GENERAR:
- ¿Qué tabla(s) contienen la información solicitada?
- ¿Se necesitan JOINs? ¿Cuáles son las relaciones FK?
- ¿Es una agregación (COUNT, SUM) o un listado?
- ¿Hay filtros EXPLÍCITOS en la pregunta? (no inventes filtros que el usuario no pidió)
- ¿Hay columnas TEXT con datos numéricos que necesiten CAST?

Responde con SOLO la consulta SQL, sin explicaciones ni formato markdown.
SQL:"""

    # Prompt for fixing SQL errors
    SQL_FIX_PROMPT = """La consulta SQL anterior tuvo un error. Corrígela.

CONSULTA ORIGINAL:
{query}

ERROR:
{error}

ESQUEMA DE BASE DE DATOS:
{schema}

CONTEXTO DEL ESQUEMA:
{schema_descriptions}

RECUERDA:
- Solo usa tablas y columnas del esquema (verifica nombres exactos)
- Si una columna es TEXT pero contiene números, usa CAST(col AS REAL) para comparaciones numéricas
- Verifica que los alias de JOIN sean correctos y que las columnas pertenezcan a la tabla correcta

Proporciona SOLO la consulta SQL corregida.
SQL:"""

    # Prompt for relaxing an over-specific query that returned 0 rows
    SQL_ZERO_ROWS_PROMPT = """Una consulta SQL se ejecutó correctamente pero devolvió 0 filas.
Esto probablemente se debe a que los filtros de texto (cláusulas LIKE o comparaciones =) son demasiado específicos y no coinciden exactamente con los valores almacenados en la base de datos.

CONSULTA ORIGINAL (devolvió 0 filas):
{query}

PREGUNTA DEL USUARIO:
{question}

ESQUEMA DE BASE DE DATOS:
{schema}

ESTRATEGIAS PARA RELAJAR LA CONSULTA:
1. Si hay cláusulas LIKE con texto específico (ej: LIKE '%fundamentos de la programación%'), prueba con términos más cortos o palabras clave principales (ej: LIKE '%fundamentos%' o LIKE '%programaci%').
2. Si hay comparaciones exactas (col = 'texto'), cámbialas a LIKE '%texto%' o usa LOWER() para ignorar mayúsculas/minúsculas (ej: LOWER(col) LIKE LOWER('%texto%')).
3. Si hay filtros combinados con AND, elimina los menos importantes y conserva solo el filtro principal.
4. Mantén la lógica de la consulta (JOINs, GROUP BY, ORDER BY) intacta.
5. No cambies lo que se pide, solo relaja los filtros de coincidencia de texto.

Responde con SOLO la consulta SQL relajada, sin explicaciones ni formato markdown.
SQL:"""

    def __init__(
        self,
        config: RAGConfig,
        llm: Optional[Any] = None,
    ):
        """
        Initialize SQL agent.

        Args:
            config: RAG configuration
            llm: LLM instance (loaded from config if not provided)
        """
        self.config = config
        self.sql_config = config.sql
        self._llm = llm

        # Initialize components (lazy loaded)
        self._schema_loader: Optional[SchemaLoader] = None
        self._executor: Optional[QueryExecutor] = None
        self._validator: Optional[QueryValidator] = None

        # Cache schema string
        self._schema_string: Optional[str] = None

        logger.info("SQLAgent initialized")

    # =========================================================================
    # Properties (Lazy Loading)
    # =========================================================================

    @property
    def llm(self):
        """Lazy-load LLM."""
        if self._llm is None:
            from rag_framework.providers.llm import LLMFactory

            self._llm = LLMFactory.get_llm(self.config.llm)
        return self._llm

    @property
    def schema_loader(self) -> SchemaLoader:
        """Lazy-load schema loader."""
        if self._schema_loader is None:
            self._schema_loader = SchemaLoader(self.sql_config)
        return self._schema_loader

    @property
    def validator(self) -> QueryValidator:
        """Lazy-load validator with schema whitelist."""
        if self._validator is None:
            # Get allowed tables/columns from schema
            allowed_tables = set(self.schema_loader.get_table_names())
            allowed_columns = set(self.schema_loader.get_all_column_names())

            self._validator = QueryValidator(
                self.sql_config.security,
                allowed_tables=allowed_tables,
                allowed_columns=allowed_columns,
            )
        return self._validator

    @property
    def executor(self) -> QueryExecutor:
        """Lazy-load query executor."""
        if self._executor is None:
            self._executor = QueryExecutor(
                engine=self.schema_loader.engine,
                config=self.sql_config,
                validator=self.validator,
            )
        return self._executor

    @property
    def schema_string(self) -> str:
        """Get cached schema string."""
        if self._schema_string is None:
            self._schema_string = self.schema_loader.get_compact_schema(
                include_samples=self.sql_config.schema.include_sample_values
            )
        return self._schema_string

    # =========================================================================
    # Main Interface
    # =========================================================================

    def query(self, question: str, trace=None) -> SQLQueryResult:
        """
        Process a natural language question and return database results.

        This is the main entry point for the SQL agent. It:
        1. Generates SQL from the question using LLM
        2. Validates the SQL for safety
        3. Executes the query
        4. Formats results for synthesis

        Args:
            question: Natural language question
            trace: Optional QueryTrace for per-attempt instrumentation

        Returns:
            SQLQueryResult with query results and metadata
        """
        import time as _time
        start_time = datetime.now()
        query_relaxed = False
        sql_attempts: List[Dict[str, Any]] = []

        logger.info(f"SQLAgent processing question: {question[:100]}...")

        try:
            # Generate SQL with retry logic
            sql_query, attempts = self._generate_sql_with_retry(
                question, attempt_log=sql_attempts
            )

            logger.info(f"Generated SQL: {sql_query}")

            # Execute query
            t_exec = _time.perf_counter()
            result = self.executor.execute(sql_query)
            exec_ms = round((_time.perf_counter() - t_exec) * 1000, 2)

            if sql_attempts:
                sql_attempts[-1]["execution_ms"] = exec_ms
                sql_attempts[-1]["error_type"] = None if result.success else "execution_error"

            if not result.success:
                # Try to fix and retry once
                t_fix_gen = _time.perf_counter()
                fixed_query = self._try_fix_query(sql_query, result.error, question)
                fix_gen_ms = round((_time.perf_counter() - t_fix_gen) * 1000, 2)
                if fixed_query and fixed_query != sql_query:
                    logger.info(f"Retrying with fixed query: {fixed_query}")
                    t_exec = _time.perf_counter()
                    result = self.executor.execute(fixed_query)
                    fix_exec_ms = round((_time.perf_counter() - t_exec) * 1000, 2)
                    sql_query = fixed_query
                    attempts += 1
                    sql_attempts.append({
                        "attempt": attempts,
                        "sql_generation_ms": fix_gen_ms,
                        "validation_ms": 0,
                        "execution_ms": fix_exec_ms,
                        "error_type": None if result.success else "fix_execution_error",
                    })

            total_time = (datetime.now() - start_time).total_seconds() * 1000

            if result.success:
                # If 0 rows returned, the query may be over-specific (LIKE too strict)
                if result.row_count == 0:
                    t_relax_gen = _time.perf_counter()
                    relaxed_query = self._try_relax_query(question, sql_query)
                    relax_gen_ms = round((_time.perf_counter() - t_relax_gen) * 1000, 2)
                    if relaxed_query and relaxed_query != sql_query:
                        logger.info(
                            f"0 rows returned — retrying with relaxed query: {relaxed_query}"
                        )
                        t_exec = _time.perf_counter()
                        relaxed_result = self.executor.execute(relaxed_query)
                        relax_exec_ms = round((_time.perf_counter() - t_exec) * 1000, 2)
                        attempts += 1
                        sql_attempts.append({
                            "attempt": attempts,
                            "sql_generation_ms": relax_gen_ms,
                            "validation_ms": 0,
                            "execution_ms": relax_exec_ms,
                            "error_type": "zero_rows_relaxed",
                        })
                        if relaxed_result.success and relaxed_result.row_count > 0:
                            result = relaxed_result
                            sql_query = relaxed_query
                            query_relaxed = True
                            logger.info(
                                f"Relaxed query returned {result.row_count} rows"
                            )
                        else:
                            logger.info(
                                "Relaxed query also returned 0 rows — keeping original result"
                            )

                if trace is not None:
                    trace.sql = {
                        "attempts": sql_attempts,
                        "total_attempts": attempts,
                        "total_ms": round(total_time, 2),
                        "success": True,
                        "query_relaxed": query_relaxed,
                    }

                formatted = result.to_natural_language()
                return SQLQueryResult(
                    success=True,
                    query=sql_query,
                    result=result,
                    formatted_result=formatted,
                    generation_attempts=attempts,
                    total_time_ms=total_time,
                    query_relaxed=query_relaxed,
                )
            else:
                if trace is not None:
                    trace.sql = {
                        "attempts": sql_attempts,
                        "total_attempts": attempts,
                        "total_ms": round(total_time, 2),
                        "success": False,
                        "query_relaxed": False,
                    }
                return SQLQueryResult(
                    success=False,
                    query=sql_query,
                    error=result.error,
                    generation_attempts=attempts,
                    total_time_ms=total_time,
                )

        except SQLSecurityError as e:
            logger.error(f"Security error in SQL generation: {e}")
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            if trace is not None:
                trace.sql = {"attempts": sql_attempts, "total_attempts": len(sql_attempts),
                             "total_ms": round(total_time, 2), "success": False, "query_relaxed": False}
            return SQLQueryResult(
                success=False,
                error=f"Security validation failed: {str(e)}",
                total_time_ms=total_time,
            )

        except Exception as e:
            logger.error(f"SQL agent error: {e}", exc_info=True)
            total_time = (datetime.now() - start_time).total_seconds() * 1000
            if trace is not None:
                trace.sql = {"attempts": sql_attempts, "total_attempts": len(sql_attempts),
                             "total_ms": round(total_time, 2), "success": False, "query_relaxed": False}
            return SQLQueryResult(
                success=False,
                error=str(e),
                total_time_ms=total_time,
            )

    # =========================================================================
    # SQL Generation
    # =========================================================================

    def _generate_sql(self, question: str) -> str:
        """
        Generate SQL from natural language using LLM.

        Args:
            question: Natural language question

        Returns:
            Generated SQL query string
        """
        # CHANGED: Build schema descriptions for richer context
        schema_descriptions = self._build_schema_descriptions()

        # Build prompt
        prompt = self.SQL_GENERATION_PROMPT.format(
            schema=self.schema_string,
            schema_descriptions=schema_descriptions,
            question=question,
        )

        # Add example queries if configured (few-shot learning)
        if self.sql_config.example_queries:
            examples = "\n\nEJEMPLOS DE REFERENCIA:\n"
            for ex in self.sql_config.example_queries:
                examples += f"Pregunta: {ex['question']}\nSQL: {ex['sql']}\n\n"
            prompt = prompt.replace(
                "PREGUNTA DEL USUARIO:",
                examples + "PREGUNTA DEL USUARIO:",
            )

        # Call LLM
        response = self.llm.complete(prompt)
        sql = str(response).strip()

        # Clean up response
        sql = self._clean_sql_response(sql)

        return sql

    def _build_schema_descriptions(self) -> str:
        """
        Build human-readable schema descriptions from SchemaLoader metadata.

        Extracts table descriptions, column details, and foreign key
        relationships to provide richer context to the LLM.

        Returns:
            Formatted string with table/column descriptions
        """
        if not self._schema_loader:
            return "No additional schema context available."

        tables = self.schema_loader.load_schema()
        if not tables:
            return "No additional schema context available."

        parts = []
        for table_name, table_info in tables.items():
            desc = f"- Table '{table_name}'"
            if table_info.description:
                desc += f": {table_info.description}"

            col_details = []
            fk_details = []
            for col in table_info.columns:
                col_desc = f"  · {col.name} ({col.data_type})"
                if col.primary_key:
                    col_desc += " [PK]"
                if col.description:
                    col_desc += f" - {col.description}"
                col_details.append(col_desc)

                if col.foreign_key:
                    fk_details.append(f"  FK: {col.name} → {col.foreign_key}")

            desc += "\n" + "\n".join(col_details)
            if fk_details:
                desc += "\n" + "\n".join(fk_details)

            parts.append(desc)

        return "\n".join(parts) if parts else "No additional schema context available."

    def _generate_sql_with_retry(
        self,
        question: str,
        attempt_log: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[str, int]:
        """
        Generate SQL with retry on validation failures.

        Args:
            question: Natural language question
            attempt_log: Optional list to append per-attempt timing dicts into

        Returns:
            Tuple of (sql_query, attempt_count)
        """
        import time as _time
        max_retries = self.sql_config.max_retries
        last_error = None

        for attempt in range(max_retries):
            t_gen = _time.perf_counter()
            sql = self._generate_sql(question)
            gen_ms = round((_time.perf_counter() - t_gen) * 1000, 2)

            # Validate
            t_val = _time.perf_counter()
            validation = self.validator.validate(sql)
            val_ms = round((_time.perf_counter() - t_val) * 1000, 2)

            if validation.is_valid:
                if attempt_log is not None:
                    attempt_log.append({
                        "attempt": attempt + 1,
                        "sql_generation_ms": gen_ms,
                        "validation_ms": val_ms,
                        "execution_ms": None,
                        "error_type": None,
                    })
                return validation.query, attempt + 1

            last_error = "; ".join(validation.errors)
            if attempt_log is not None:
                attempt_log.append({
                    "attempt": attempt + 1,
                    "sql_generation_ms": gen_ms,
                    "validation_ms": val_ms,
                    "execution_ms": None,
                    "error_type": "validation_error",
                })
            logger.warning(
                f"SQL validation failed (attempt {attempt + 1}): {last_error}"
            )

            # Try to fix the query
            if attempt < max_retries - 1:
                t_fix = _time.perf_counter()
                sql = self._try_fix_query(sql, last_error, question)
                fix_ms = round((_time.perf_counter() - t_fix) * 1000, 2)
                if sql:
                    t_val2 = _time.perf_counter()
                    validation = self.validator.validate(sql)
                    val2_ms = round((_time.perf_counter() - t_val2) * 1000, 2)
                    if validation.is_valid:
                        if attempt_log is not None:
                            attempt_log.append({
                                "attempt": attempt + 2,
                                "sql_generation_ms": fix_ms,
                                "validation_ms": val2_ms,
                                "execution_ms": None,
                                "error_type": None,
                            })
                        return validation.query, attempt + 2

        raise SQLGenerationError(
            f"Failed to generate valid SQL after {max_retries} attempts. "
            f"Last error: {last_error}"
        )

    def _try_fix_query(
        self,
        query: str,
        error: str,
        original_question: str,
    ) -> Optional[str]:
        """
        Try to fix a failed SQL query using LLM.

        Args:
            query: The failed query
            error: Error message
            original_question: Original user question

        Returns:
            Fixed query or None if fix failed
        """
        try:
            schema_descriptions = self._build_schema_descriptions()
            prompt = self.SQL_FIX_PROMPT.format(
                query=query,
                error=error,
                schema=self.schema_string,
                schema_descriptions=schema_descriptions,
            )

            response = self.llm.complete(prompt)
            fixed_sql = str(response).strip()
            fixed_sql = self._clean_sql_response(fixed_sql)

            return fixed_sql

        except Exception as e:
            logger.warning(f"Failed to fix query: {e}")
            return None

    def _try_relax_query(
        self,
        question: str,
        zero_rows_query: str,
    ) -> Optional[str]:
        """
        Ask the LLM to regenerate a looser version of a query that returned 0 rows.

        Relaxation strategies include simplifying LIKE terms, using LOWER() for
        case-insensitive matching, and removing over-restrictive AND conditions.

        Args:
            question: Original natural language question
            zero_rows_query: The SQL that executed successfully but returned no rows

        Returns:
            Relaxed SQL query (validated) or None if generation/validation failed
        """
        try:
            prompt = self.SQL_ZERO_ROWS_PROMPT.format(
                query=zero_rows_query,
                question=question,
                schema=self.schema_string,
            )

            response = self.llm.complete(prompt)
            relaxed_sql = self._clean_sql_response(str(response).strip())

            # Validate before returning
            validation = self.validator.validate(relaxed_sql)
            if validation.is_valid:
                return validation.query

            logger.warning(
                f"Relaxed SQL failed validation: {'; '.join(validation.errors)}"
            )
            return None

        except Exception as e:
            logger.warning(f"Failed to relax query: {e}")
            return None

    def _clean_sql_response(self, sql: str) -> str:
        """
        Clean LLM response to extract pure SQL.

        Removes markdown formatting, explanations, multiple statements, etc.
        SQLite (and safe practice in general) only allows one statement at a time.
        """
        # Remove markdown code blocks
        if "```sql" in sql.lower():
            sql = sql.split("```sql")[-1].split("```")[0]
        elif "```" in sql:
            sql = sql.split("```")[1].split("```")[0]

        # Remove common prefixes
        prefixes_to_remove = ["SQL:", "Query:", "SELECT"]
        for prefix in prefixes_to_remove[:-1]:
            if sql.upper().startswith(prefix.upper()):
                sql = sql[len(prefix) :].strip()

        # Ensure it starts with SELECT
        if not sql.upper().startswith("SELECT"):
            # Try to find SELECT in the response
            idx = sql.upper().find("SELECT")
            if idx != -1:
                sql = sql[idx:]

        sql = sql.strip()

        # ── Keep only the first SQL statement ──────────────────────
        # LLMs sometimes return multiple statements separated by ";".
        # SQLite raises "You can only execute one statement at a time."
        # We split on ";" that are NOT inside quoted strings and keep
        # only the first non-empty SELECT statement.
        if ";" in sql:
            # Simple split that respects single-quoted string literals
            statements = self._split_statements(sql)
            for stmt in statements:
                stmt = stmt.strip()
                if stmt and stmt.upper().startswith("SELECT"):
                    sql = stmt
                    break

        return sql.strip()

    @staticmethod
    def _split_statements(sql: str) -> List[str]:
        """Split SQL on semicolons that are outside of quoted strings."""
        statements: List[str] = []
        current: List[str] = []
        in_single_quote = False

        for char in sql:
            if char == "'" and not in_single_quote:
                in_single_quote = True
                current.append(char)
            elif char == "'" and in_single_quote:
                in_single_quote = False
                current.append(char)
            elif char == ";" and not in_single_quote:
                statements.append("".join(current))
                current = []
            else:
                current.append(char)

        # Remaining text after last semicolon
        remaining = "".join(current).strip()
        if remaining:
            statements.append(remaining)

        return statements

    # =========================================================================
    # Schema Access
    # =========================================================================

    def get_schema_for_router(self) -> Dict[str, Any]:
        """Get schema info formatted for the query router."""
        return self.schema_loader.get_schema_for_router()

    def refresh_schema(self) -> None:
        """Force refresh of schema cache."""
        self._schema_string = None
        self.schema_loader.load_schema(force_reload=True)
        logger.info("Schema cache refreshed")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def close(self) -> None:
        """Clean up resources."""
        if self._schema_loader:
            self._schema_loader.close()
        logger.info("SQLAgent resources released")
