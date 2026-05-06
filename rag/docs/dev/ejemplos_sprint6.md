# Ejemplos para mostrar sprint6

## Ejemplo uso de la API

- Terminal 1: arrancar el servidor
```bash
python run_rag.py api
```

- Terminal 2: ejecutar la demo
```bash
python demos/demo_api_rest.py
```

## Ejemplo uso del orquestador de datos estructurados y no estructurados

```bash
python demos/demo_router_orchestration.py
```

----------------------------------

## Activar datasette

```bash
datasette ./data/example.db
```

----------------------------------

## PROPUESTAS FUTURAS

- **Buffer de historia** con ventana deslizante (últimos N turnos): Podríamos utilizar el patrón Condense Question y generar un resumen acumulativo de la conversación.

```python
Usuario: "¿Qué productos hay?"          → SQL: SELECT * FROM productos
Usuario: "¿Cuál es el más caro?"         → Sin memoria: FALLA (no sabe de qué habla)
                                          → Con memoria: reformula a "¿Cuál es el producto más caro?"
```

- Utilizar framework **RAGAS** (Retrieval-Augmented Generation Assessment): Para poder obtener métricas de calidad de la respuesta. Esto sólo tiene sentido para hacer pruebas y poder tener unas métricas sobre el rendimiento del RAG y evaluar la calidad de este. No podemos dar métricas de calidad de respuestas que estemos dando a usuarios.

- **Streaming de respuestas**: En vez de esperar 10-30 segundos para que de la respuesta completa el LLM antes de mostrarnosla, ir viendo token a token que se va generando.

- **Citación de fuentes**: Ver si es posible que el LLM responda de la siguiente forma

```txt
Respuesta: Los plazos de matrícula son del 1 al 15 de septiembre [1],
y el coste es de 1.200€ por curso [2].

[1] normativa_academica.pdf, p.23
[2] tabla: tasas, SQL: SELECT importe FROM tasas WHERE concepto='matrícula'
```

