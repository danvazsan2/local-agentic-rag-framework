# Inventario del Corpus — Evaluación TFG RAG

**Grado en Ingeniería Informática – E.T.S. Ingeniería Informática, Universidad de Sevilla**  
Curso académico activo en la BD: 2025-26

---

## 1. Base de datos: `data/proyectos_docentes.db`

### Tablas relevantes para el dataset

| Tabla | Filas | Descripción |
|-------|-------|-------------|
| ASG_ASIGNATURA_1 | 57 | Catálogo completo de asignaturas (57 total = 53 docentes + 4 movilidad) |
| ASG_CPSGACPRF_1 | 429 | Asignaciones grupo-profesor (con nombres de grupo) |
| ASG_PROYECTODATO | 3 404 | Contenidos de proyectos docentes por grupo/año |
| ASG_PROGRAMADATO | 13 875 | Datos del programa maestra (objetivos, contenidos, metodología) |
| ASG_PROGRAMACABECERA_FOTO | 497 | Cabeceras de programas con versiones históricas |
| ASG_PROYECTOPROFESOR | 112 | Relación proyecto-profesor |
| ASG_PERMISO_ROL_USUARIO | 65 | Tabla de profesores con nombre, apellidos, uvus |
| ASG_BIBLIOGRAFIA | 578 | Bibliografía por asignatura |
| ASG_PROGRAMAANYO_FOTO | 53 | Un registro por asignatura activa en 2025-26 |

### Tipos de asignatura (TAS_CODALF)

| Código | Nombre | Count |
|--------|--------|-------|
| T | Troncal / Formación Básica | 9 |
| B | Obligatoria | 9 |
| O | Optativa | 37 |
| X | Prácticas Externas Optativas | 1 |
| P | TFG / Proyecto Fin de Carrera | 1 |

**Respuestas SQL verificadas:**
- Optativas: **37**
- Obligatorias: **9**
- Troncales: **9**
- Anuales: **7** (FP, ADDA, Prácticas Externas, y 4 de movilidad)
- Cuatrimestrales: **50**
- Asignaturas con 12 créditos: **3** (FP 2060001, ADDA 2060010, TFG 2060053)
- Departamentos distintos: **9**
- Créditos 1.er curso: **60** | Créditos 2.º curso: **60**

---

## 2. Corpus documental (PDFs)

**Total PDFs:** ~195 ficheros en `documents/`

| Tipo | Patrón | Count |
|------|--------|-------|
| Programa maestra | `Programa_CODNUM_VERSION.pdf` | 55 |
| Proyecto docente | `Proyecto_CODNUM_2025-26_GRUPO.pdf` | ~140 |

**Asignaturas con múltiples proyectos docentes (varios grupos):**
- 2060001 FP: grupos 1, 2, 3
- 2060003 Cálculo: grupos 1, 2, 3, 4
- 2060005 IMD: grupos 1, 2, 3, 4
- 2060010 ADDA: grupos 1TI, 2TI, 3TI (sufijo TI = Tecnologías Informáticas)
- 2060054/2060055 IS I/II: grupos 1TI, 2TI, 3TI

**Sin PDF de Proyecto 2025-26** (solo Programa disponible):
- 2060031 Prácticas Externas
- 2060053 TFG

**PDF obsoleto** (solo 2015-16):
- 2060036 (ya no aparece en catálogo activo)

---

## 3. Catálogo completo de asignaturas docentes

| Cód | Nombre | Curso | Tipo | Cr | Sem | Departamento |
|-----|--------|-------|------|----|-----|--------------|
| 2060001 | Fundamentos de Programación | 1 | Troncal | 12 | Anual | Lenguajes y Sistemas Informáticos |
| 2060002 | Administración de Empresas | 1 | Troncal | 6 | C2 | Organización Industrial y Gestión Emp.I |
| 2060003 | Cálculo Infinitesimal y Numérico | 1 | Troncal | 6 | C1 | Matemática Aplicada I |
| 2060004 | Circuitos Electrónicos Digitales | 1 | Troncal | 6 | C1 | Tecnología Electrónica |
| 2060005 | Introducción a la Matemática Discreta | 1 | Troncal | 6 | C1 | Matemática Aplicada I |
| 2060006 | Álgebra Lineal y Numérica | 1 | Troncal | 6 | C2 | Matemática Aplicada I |
| 2060007 | Estadística | 1 | Troncal | 6 | C2 | Estadística e Investigación Operativa |
| 2060008 | Estructura de Computadores | 1 | Troncal | 6 | C2 | Tecnología Electrónica |
| 2060009 | Fundamentos Físicos de la Informática | 1 | Troncal | 6 | C1 | Física Aplicada I |
| 2060010 | Análisis y Diseño de Datos y Algoritmos | 2 | Obligatoria | 12 | Anual | Lenguajes y Sistemas Informáticos |
| 2060012 | Lógica Informática | 2 | Optativa | 6 | C1 | Ciencias de la Computación e IA |
| 2060013 | Matemática Discreta | 2 | Obligatoria | 6 | C1 | Matemática Aplicada I |
| 2060014 | Redes de Computadores | 2 | Obligatoria | 6 | C1 | Tecnología Electrónica |
| 2060015 | Arquitectura de Computadores | 2 | Obligatoria | 6 | C2 | Arquitectura y Tecnol. de Computadores |
| 2060016 | Arquitectura de Redes | 2 | Optativa | 6 | C2 | Tecnología Electrónica |
| 2060017 | Sistemas Operativos | 2 | Obligatoria | 6 | C2 | Lenguajes y Sistemas Informáticos |
| 2060054 | Introducción a la IS y SI I | 2 | Obligatoria | 6 | C1 | Lenguajes y Sistemas Informáticos |
| 2060055 | Introducción a la IS y SI II | 2 | Obligatoria | 6 | C2 | Lenguajes y Sistemas Informáticos |
| 2060018 | Configuración, Implementación y Mantenimiento de SI | 3 | Optativa | 6 | C1 | Arquitectura y Tecnol. de Computadores |
| 2060019 | Gestión de Sistemas de Información | 3 | Optativa | 6 | C1 | Lenguajes y Sistemas Informáticos |
| 2060020 | Gestión y Estrategia Empresarial | 3 | Optativa | 6 | C1 | Organización Industrial y Gestión Emp.I |
| 2060021 | Inteligencia Artificial | 3 | Obligatoria | 6 | C1 | Ciencias de la Computación e IA |
| 2060022 | Procesadores de Lenguajes | 3 | Optativa | 6 | C1 | Lenguajes y Sistemas Informáticos |
| 2060023 | Programación Declarativa | 3 | Optativa | 6 | C1 | Ciencias de la Computación e IA |
| 2060024 | Tecnologías Avanzadas de la Información | 3 | Optativa | 6 | C1 | Tecnología Electrónica |
| 2060025 | Ampliación de Inteligencia Artificial | 3 | Optativa | 6 | C2 | Ciencias de la Computación e IA |
| 2060026 | Arquitectura de Sistemas Distribuidos | 3 | Optativa | 6 | C2 | Arquitectura y Tecnol. de Computadores |
| 2060027 | Matemática Aplicada a Sistemas de Información | 3 | Optativa | 6 | C2 | Matemática Aplicada I |
| 2060028 | Sistemas de Información Empresariales | 3 | Optativa | 6 | C2 | Lenguajes y Sistemas Informáticos |
| 2060029 | Sistemas Inteligentes | 3 | Optativa | 6 | C2 | Ciencias de la Computación e IA |
| 2060030 | Sistemas Orientados a Servicios | 3 | Optativa | 6 | C2 | Lenguajes y Sistemas Informáticos |
| 2060031 | Prácticas Externas | 4 | Ext. Opt. | 6 | Anual | Lenguajes y Sistemas Informáticos |
| 2060032 | Acceso Inteligente a la Información | 4 | Optativa | 6 | C2 | Lenguajes y Sistemas Informáticos |
| 2060033 | Administración de Sistemas de Información | 4 | Optativa | 6 | C1 | Tecnología Electrónica |
| 2060034 | Gestión de Procesos y Servicios | 4 | Optativa | 6 | C1 | Lenguajes y Sistemas Informáticos |
| 2060035 | Infraestructura de Sistemas de Información | 4 | Optativa | 6 | C1 | Tecnología Electrónica |
| 2060037 | Interacción Persona-ordenador | 4 | Optativa | 6 | C1 | Lenguajes y Sistemas Informáticos |
| 2060038 | Matemática Aplicada a Tecnologías de la Información | 4 | Optativa | 6 | C1 | Matemática Aplicada I |
| 2060039 | Matemáticas para la Computación | 4 | Optativa | 6 | C1 | Matemática Aplicada I |
| 2060040 | Planificación y Gestión de Proyectos Informáticos | 4 | Obligatoria | 6 | C1 | Matemática Aplicada I |
| 2060041 | Procesamiento de Imágenes Digitales | 4 | Optativa | 6 | C1 | Matemática Aplicada I |
| 2060042 | Seguridad en Sistemas Informáticos y en Internet | 4 | Optativa | 6 | C1 | Lenguajes y Sistemas Informáticos |
| 2060043 | Teledetección | 4 | Optativa | 6 | C1 | Tecnología Electrónica |
| 2060044 | Aplicaciones de Soft Computing | 4 | Optativa | 6 | C2 | Electrónica y Electromagnetismo |
| 2060045 | Computación Móvil | 4 | Optativa | 6 | C2 | Arquitectura y Tecnol. de Computadores |
| 2060046 | Criptografía | 4 | Optativa | 6 | C2 | Matemática Aplicada I |
| 2060047 | Estadística Computacional | 4 | Optativa | 6 | C2 | Estadística e Investigación Operativa |
| 2060048 | Gestión de la Producción | 4 | Optativa | 6 | C2 | Organización Industrial y Gestión Emp.I |
| 2060049 | Inteligencia Empresarial | 4 | Optativa | 6 | C2 | Lenguajes y Sistemas Informáticos |
| 2060050 | Modelado y Análisis de Requisitos en SI | 4 | Optativa | 6 | C2 | Lenguajes y Sistemas Informáticos |
| 2060051 | Modelos de Computación y Complejidad | 4 | Optativa | 6 | C2 | Ciencias de la Computación e IA |
| 2060052 | Tecnología, Informática y Sociedad | 4 | Optativa | 6 | C2 | Tecnología Electrónica |
| 2060053 | Trabajo Fin de Grado | 4 | TFG | 12 | C2 | Lenguajes y Sistemas Informáticos |

---

## 4. Clusters de solapamiento semántico

Estas agrupaciones son las que hacen difícil el retrieval sin metafiltro o QueryPreprocessor. Críticas para las consultas adversariales.

### Cluster IA (6 asignaturas)
`2060021` Inteligencia Artificial · `2060025` Ampliación de Inteligencia Artificial · `2060029` Sistemas Inteligentes · `2060023` Programación Declarativa · `2060032` Acceso Inteligente a la Información · `2060044` Aplicaciones de Soft Computing

Todas comparten vocabulario: agentes, búsqueda, conocimiento, razonamiento, aprendizaje, redes neuronales. Una query como "¿cómo se evalúa la asignatura de IA avanzada?" podría recuperar chunks de las 6 sin QueryPreprocessor.

### Cluster Redes/Arquitectura (3 asignaturas)
`2060014` Redes de Computadores · `2060016` Arquitectura de Redes · `2060026` Arquitectura de Sistemas Distribuidos

Solapamiento en: protocolos, TCP/IP, capas, routing, switches. "La de redes" es radicalmente ambiguo.

### Cluster Programación (4 asignaturas)
`2060001` Fundamentos de Programación · `2060010` Análisis y Diseño de Datos y Algoritmos · `2060013` Matemática Discreta · `2060022` Procesadores de Lenguajes

Comparten: algoritmos, estructuras de datos, lenguajes, código, compilación.

### Cluster Matemáticas (6 asignaturas)
`2060005` Introducción a la Matemática Discreta · `2060006` Álgebra Lineal y Numérica · `2060007` Estadística · `2060013` Matemática Discreta · `2060027` Matemática Aplicada a SI · `2060038` Matemática Aplicada a TI · `2060039` Matemáticas para la Computación · `2060047` Estadística Computacional

Especialmente: IMD (2060005) y MD (2060013) tienen nombres casi idénticos y son del mismo departamento.

### Cluster SI/Gestión (5 asignaturas)
`2060019` Gestión de Sistemas de Información · `2060028` Sistemas de Información Empresariales · `2060049` Inteligencia Empresarial · `2060050` Modelado y Análisis de Requisitos en SI · `2060054` IS I · `2060055` IS II

Comparten: sistemas de información, bases de datos, modelado, requisitos, empresas.

### Cluster Sistemas (4 asignaturas)
`2060017` Sistemas Operativos · `2060033` Administración de Sistemas de Información · `2060035` Infraestructura de Sistemas de Información · `2060018` Configuración, Implementación y Mantenimiento de SI

"La asignatura de sistemas" es indesambiguable sin código.

---

## 5. Profesorado (para verificar negativas engañosas)

Apellidos presentes: Aguilar, Aguirre, Alonso, Arahal, Arroyo, Bravo, Cisneros, Contreras, Díaz (×2), Fajardo, Fariñas, Fernández de Córdoba, Gallo, García, Gómez (×2), Grandes, Hidalgo, Holguín, López (×2), Lucio-Villegas, Luis, Luque, Melendo, Milán (×2), Montes, Muñoz, Pérez, Rodríguez (×4), Sánchez (×2), Santos, Seco, Surián, Valdecantos, Vinagre

**Apellidos NO presentes** (seguros para negativas engañosas): González, Martínez, Jiménez, Moreno, Torres, Serrano, Flores (hay Santos Flores pero el apellido principal es Santos).

---

## 6. Estadísticas del corpus de profesores por asignatura

| Asignatura | Profs distintos | Grupos |
|-----------|-----------------|--------|
| 2060001 FP | 12 | 9 |
| 2060010 ADDA | 12 | 9 |
| 2060007 Estadística | 10 | 6 |
| 2060016 Arq. Redes | 7 | 13 |
| 2060004 Circuitos | 7 | 13 |
| 2060015 Arq. Computadores | 8 | 16 |
| 2060014 Redes | 8 | 10 |
| 2060009 Fund. Físicos | 6 | 22 |
| 2060055 IS II | 6 | 6 |
| 2060021 IA | 4 | 3 |
| 2060017 SO | 4 | 4 |

---

## 7. Notas de cobertura para el diseño del dataset

- **r-queries fáciles**: usar asignaturas de 1.º-2.º con PDFs Programa + Proyecto completos.
- **r-queries difíciles**: usar optativas de 4.º que solo tienen 1 grupo (menos contexto en retrieval).
- **SQL s01–s10**: todas las respuestas verificadas arriba; no usar `expected_answer_contains` con el número exacto de forma muy literal para evitar fragilidad.
- **Negativas engañosas**: los nombres "Programación Funcional", "Visión por Computador" y "Seguridad Avanzada" NO existen en el catálogo. El apellido "González" no está en el profesorado.
- **Abstención pura a01–a04**: fuera del dominio universitario por completo.
