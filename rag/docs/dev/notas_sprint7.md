# NOTAS SPRINT 7


## ALGUNAS COSAS PARA COPIAR Y PEGAR

```bash
python demos/demo_proyectos_docentes.py --no-pause
```
```txt
quién es el coordinador de Introducción a la Ingenería del Software y los Sistemas de la Información I?
```
```txt
Dame el nombre de todas las asignaturas optativas del plan de estudios. Damelo en este formato 1. nombre1\n2. nombre2\n...
```

## PROPUESTAS A FUTURO

- Crear una batería de preguntas con sus respuestas (que estén verificadas) y lanzar el proyecto, y hacer la comparativa "Naive" vs "RAG básico" vs "RAG completo" y obtener métricas de mejora.
    - De aquí podríamos generar una gráfica interesante y sacar en claro algunas conclusiones importantes
    - Esto lo podríamos incluso hacer más detallado, añadiendo "capa a capa" todo nuestro RAG, para ver qué elementos del RAG son más importantes. Por ejemplo:
        - Empezamos con RAG solo vectorial -> obtenemos %
        - Ahora RAG vectorial + BM25 (búsqueda léxica) -> obtenemos %
        - Ahora lo anterior + reranker -> obtenemos %
        - ...
- Documentar los modelos más comúnes (por ejemplo: para el LLM qwen3:8b, llama3-instruct; para el embedding y reranker lo mismo) y proponer modelos para cuando tenemos más o menos VRAM:
    - Por ejemplo, en nuestro caso estamos usando 8 GB de VRAM y estamos usando español.
    - Podríamos documentar paquetes de (llm, embedding, reranker y base de datos) para personas que cuenten con menos VRAM o más VRAM; para usarlo con documentos en inglés;...
- Vamos a utilizar una versión web? O vamos a proponer no utilizar ventana gráfica?
- Para empezar a redactar formalmente el TFG usamos alguna plantilla? 


Algo relacionado con las menciones de teoría

Estudiar limit en SQL

