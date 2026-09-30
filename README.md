# Bookery AI

Proyecto final de Machine Learning: recomendación de libros mediante una interfaz conversacional sencilla. La pregunta principal es: **¿la recuperación semántica mediante embeddings mejora el ranking de libros frente a BM25 para consultas conversacionales con preferencias explícitas?** Este módulo responde a la función de recomendaciones planteada para Todo Libros en el proyecto de grado; la integración con inventario y ventas queda fuera del alcance de esta entrega académica.

Bookery AI usa el **mismo catálogo** para ambos métodos. El backend devuelve exclusivamente títulos existentes en ese catálogo. Opcionalmente, un LLM redacta una explicación breve para cada libro recuperado por el método semántico. El frontend abre con el método semántico evaluado y permite alternar a BM25 o a Semántico + Jev para explorar.

## Arquitectura

`Open Library (ediciones en español) + anotaciones propias → books.json → BM25 / embeddings normalizados → FastAPI → LLM opcional → Nuxt 4`

BM25 usa `k1=1.5`, `b=0.75` y tokenización Unicode simple. La búsqueda semántica usa [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), revisión `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, un modelo **preentrenado** de 384 dimensiones. Se reutilizan sus pesos mediante `sentence-transformers==3.3.1`; aquí se desarrollaron la preparación de datos, el índice exacto NumPy, los filtros, la API y la evaluación. No se entrenó ni ajustó el modelo. Los vectores se normalizan y se ordenan por producto punto, equivalente al coseno. El artefacto guarda modelo, revisión, dimensión y huella del catálogo.

## Dataset

El catálogo de demostración contiene **215 libros con edición en español** y 28 categorías principales, entre ellas fantasía, ciencia ficción, misterio, romance, terror, clásicos, poesía, filosofía, historia, ciencia, tecnología, cocina, arte, cómic y viajes. Partió de 53 fichas revisadas y sumó 161 obras seleccionadas de la [API de búsqueda de Open Library](https://openlibrary.org/dev/docs/api/search) según su señal `readinglog_count` por tema. La importación exigió una edición marcada `spa`, eliminó duplicados por ID de obra y revisó manualmente las categorías; 160 de esas 161 obras tienen un recuento positivo de lecturas. Esta señal mide actividad en Open Library y **no equivale a ventas, popularidad universal ni existencias en Todo Libros**. Los registros añadidos están congelados en [anotaciones](data/curated/popular_annotations_es.json), [ediciones](data/curated/popular_editions_es.json) y [decisiones de revisión](data/curated/popular_review_es.json). Las 54 fichas de base incluyen cuatro obras verificadas en páginas de Editorial Planeta, entre ellas [Yo también puedo programar](https://www.planetadelibros.com/libro-yo-tambien-puedo-programar/214123).

`python -m ml.build_curated_catalog` une las anotaciones revisadas con las ediciones y genera `data/processed/books.json` de forma determinista. Se guardan `id`, `title`, `author`, `description`, `genres`, `language=es`, año, ID de edición o ISBN, URL y fuente. `searchable_text` concatena título, autor, géneros y descripción. Una ficha incompleta o duplicada produce un error. Las 161 fichas nuevas contienen resúmenes de metadatos y etiquetas temáticas en español, **sin sinopsis argumental verificada**; por eso funcionan mejor para búsquedas por título, autor y categoría que por detalles de trama. Las 54 fichas de base sí incluyen sinopsis breves redactadas para el proyecto.

Para actualizar la selección con nuevos datos públicos: ejecute `python -m ml.expand_popular_catalog` (requiere internet), revise los candidatos en `data/raw/`, actualice `data/curated/popular_review_es.json` por ID y género, y ejecute `python -m ml.review_popular_catalog`. Después regenere catálogo e índice semántico. Las instantáneas revisadas ya están en Git, por lo que la instalación normal **no requiere** volver a consultar Open Library. Los antiguos importadores `ml.import_openlibrary` y `ml.prepare_dataset` sirven para explorar otras fuentes; sus salidas sin revisión no sustituyen el catálogo revisado.

Los metadatos de Open Library pueden tener derechos preexistentes; consulte su [información de licencias](https://openlibrary.org/developers/licensing). Las sinopsis de las fichas de base son originales; las fichas ampliadas explicitan que no contienen una sinopsis verificada. La selección y redacción manual pueden favorecer ciertos temas y títulos; los resultados solo generalizan a estos 215 registros. El catálogo no registra precio ni disponibilidad de Todo Libros.

## Instalación y ejecución

Requisitos: Python 3.11 o 3.12, Node.js 22 y pnpm 11. En macOS, compruebe la versión con `python3.12 --version`; el Python 3.9 incluido en algunos equipos no instala `numpy==2.2.1`. Desde la raíz:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ml.build_curated_catalog
python -m ml.semantic_retriever
uvicorn backend.app.main:app --reload
```

La primera generación de embeddings descarga el modelo preentrenado. Cada vez que cambie una ficha del catálogo, regenere el JSON y los embeddings antes de iniciar el backend. El backend sirve `http://localhost:8000/health`. En otra terminal:

```bash
cd frontend
pnpm install
pnpm dev
```

Abra `http://localhost:3000`. Para una prueba rápida de la API:

```bash
curl -X POST http://localhost:8000/api/recommend -H 'Content-Type: application/json' -d '{"query":"Quiero una novela sobre tecnología y control social","limit":5}'
```

Para explorar el catálogo en la interfaz, pruebe «Quiero leer algo parecido a Harry Potter», «Busco un asesinato misterioso en un tren» y «Prefiero una historia familiar latinoamericana con elementos sobrenaturales». La primera consulta busca alternativas y excluye el libro citado. La tarjeta enlaza a la edición española registrada en Open Library.

`POST /api/search/bm25` acepta el mismo cuerpo. Ambos admiten `language`, `genre`, `author` y `limit` de 1 a 20. El frontend usa `language=es`. En consultas como «algo parecido a Harry Potter», los métodos excluyen el libro mencionado y usan su ficha para encontrar alternativas. También detectan consultas por autor, como «libros parecidos a los de Marian Rojas Estapé»: excluyen las obras del autor, amplían la búsqueda con sus temas registrados y priorizan otros libros con géneros compartidos. En estas consultas, el sistema explica la coincidencia usando solo las categorías del catálogo; no envía la respuesta al LLM. El indicador de la interfaz dice «Similitud por categorías del catálogo». En consultas normales el LLM opcional sigue redactando las explicaciones. `GET /api/options` devuelve idiomas y géneros disponibles. Una consulta vacía o un límite inválido devuelve HTTP 422. Si faltan catálogo o embeddings, el inicio del backend falla de forma explícita.

Las preguntas por un título específico («¿tienen el libro X?», «¿de qué trata X?», «¿quién escribió X?») consultan la ficha del catálogo sin llamar al LLM. Una ficha encontrada **no confirma existencias en la librería**. Un título ausente se informa como desconocido. «¿Tienen algún libro similar a X?» se interpreta como petición de similitud; si X no tiene ficha de referencia, se responde que no hay evidencia suficiente y no se muestran coincidencias accidentales. «Libros de [autor registrado]» lista sus obras del catálogo. Las peticiones generales, como «quiero ciencia ficción», siguen usando el método de recuperación seleccionado. Estas reglas cubren las formas documentadas; otras redacciones ambiguas pueden requerir aclaración o una ficha adicional.

### Selección semántica + Jev

El modo **Semántico + Jev** recupera primero hasta cinco libros del catálogo y pide a Jev elegir el más indicado entre esos IDs, con una opción `ninguno`. La interfaz destaca el elegido y muestra los demás como alternativas. Jev no genera títulos ni explicaciones: las fichas siguen viniendo del catálogo y el LLM opcional continúa redactando motivos. Si Jev no está configurado, responde con error, elige `ninguno` o tiene baja confianza, se conserva el orden semántico y la interfaz lo indica.

Para activarlo, configure `TYPESAFE_API_KEY` en `.env` (y, si desea, `BOOKERY_JEV_MODEL`; el valor predeterminado es `jev-latest`). La ruta es `POST /api/recommend/jev` y acepta el mismo cuerpo que `/api/recommend`. La clave se usa solo desde el backend. La selección de Jev cambia el orden final y debe evaluarse por separado de BM25 y el recuperador semántico para sacar conclusiones académicas.


### Activar explicaciones con LLM

La aplicación funciona sin clave: presenta explicaciones básicas. Para usar un proveedor compatible con [Chat Completions y JSON mode](https://developers.openai.com/api/docs/guides/structured-outputs), copie `.env.example` a `.env`, configure modelo y clave y reinicie el backend. El backend **sí carga `.env` automáticamente**. También puede exportar las variables antes de iniciarlo:

```bash
export BOOKERY_LLM_BASE_URL=https://api.openai.com/v1
export BOOKERY_LLM_MODEL=gpt-4o-mini
export BOOKERY_LLM_API_KEY=su_clave
uvicorn backend.app.main:app --reload
```

Para Docker, copie `.env.example` a `.env`, edite `BOOKERY_LLM_MODEL` y `BOOKERY_LLM_API_KEY`, y ejecute `docker compose up --build`. No incluya `.env` ni claves en Git. También puede usar un servidor local compatible, por ejemplo [Ollama](https://registry.ollama.com/blog/openai-compatibility): `BOOKERY_LLM_BASE_URL=http://localhost:11434/v1`, `BOOKERY_LLM_MODEL=<modelo_instalado>` y clave vacía. Si Ollama está en el host y el backend en Docker, use una URL accesible desde el contenedor, como `http://host.docker.internal:11434/v1` en macOS.

El LLM recibe la consulta y los datos de **los libros ya recuperados que tienen sinopsis revisada**; redacta solo el campo `reason`. Las 161 fichas ampliadas, que contienen únicamente metadatos temáticos, usan un motivo determinista que indica expresamente que la trama no está verificada. La API valida que las explicaciones correspondan exactamente a los ID enviados y conserva títulos y orden del recuperador. Si el servicio falla, tarda demasiado o devuelve JSON inválido, usa explicaciones básicas. `explanation_mode` indica `llm`, `basic`, `reference` o `catalog` en la respuesta; el frontend lo muestra. Una explicación generada todavía puede contener inferencias inexactas sobre un libro: revise manualmente las afirmaciones antes de una demostración o publicación.

También se puede usar Docker tras preparar los datos en el host:

```bash
docker compose up --build
```

## Evaluación

### Calificación humana desde la web

Con el backend y el frontend iniciados, abra `http://localhost:3000` y pulse **Evaluar**. Seleccione la primera consulta, lea cada ficha y asigne una nota a **todos** los libros: `0` no encaja, `1` poco, `2` bien, `3` muy bien. Los candidatos son la unión sin duplicados de los cinco primeros resultados de BM25 y semántica; aparecen ordenados por título, sin indicar su procedencia. Pulse **Guardar notas de esta consulta** y continúe con las demás. Puede cerrar y volver a abrir la página: las notas guardadas permanecen en `data/evaluation/reviewed_queries.json`. Una nota `0` también debe guardarse; «Sin calificar» significa que aún falta revisar ese libro.

Al completar las nueve consultas, pulse **Calcular comparación final**. El backend comprueba que todas las recomendaciones actuales tengan nota, calcula ambas métricas y escribe `results/evaluation_reviewed.csv`. Ese archivo y `reviewed_queries.json` son los entregables de la revisión humana. Si cambia el catálogo o el modelo, vuelva a revisar los candidatos afectados. La evaluación no modifica las recomendaciones normales ni las etiquetas piloto originales. En Docker, los directorios de evaluación y resultados se montan con escritura para conservar las notas en el host.

`data/evaluation/queries.json` conserva las nueve consultas y etiquetas piloto propuestas por el asistente antes de la revisión. La evaluación final utiliza `data/evaluation/reviewed_queries.json`: una persona calificó manualmente 71 pares consulta-libro, reuniendo los cinco primeros candidatos de cada método, eliminando duplicados y ocultando el método de origen. Para reproducir la tabla final, con catálogo e índice preparados, ejecute:

```bash
python -m ml.evaluate --cases data/evaluation/reviewed_queries.json --output results/evaluation_reviewed.csv
```

El script ejecuta **las mismas consultas, filtros y catálogo** contra BM25 y búsqueda semántica. Reporta medias de:

- **nDCG@5:** `DCG@5 / IDCG@5`, donde `DCG@5 = Σ(2^relevancia − 1)/log₂(posición + 1)` con posiciones desde 1. Es la métrica principal y premia el orden de libros muy relevantes.
- **Precision@5:** cantidad de libros con relevancia ≥ 2 en las primeras cinco posiciones, dividida por 5.
- **Cumplimiento:** proporción de resultados que cumplen filtros explícitos en consultas que los usan; si no hay resultados por filtros, vale 1 de manera vacía y debe interpretarse junto al número de resultados.
- **Tiempo:** duración media de `search()` en milisegundos, medida tras cargar modelos e índices. Depende del hardware y no incluye HTTP ni arranque.

La evaluación revisada sobre los 215 libros y nueve consultas produjo `results/evaluation_reviewed.csv`: BM25 nDCG@5 `0.8456` y Precision@5 `0.4444`; semántica nDCG@5 `0.8998` y Precision@5 `0.4222`. El método semántico ordenó mejor los libros relevantes según la métrica principal; BM25 obtuvo un libro adicional con relevancia ≥ 2 entre las 45 posiciones evaluadas. El cumplimiento de filtros fue `1.0` para ambos. Las latencias medias de cada ejecución figuran en el CSV; dependen del hardware y no incluyen HTTP, LLM ni arranque. Estas cifras describen solo las nueve consultas y el catálogo actual; no miden la calidad de las explicaciones del LLM ni el modo Semántico + Jev. El archivo `results/evaluation.csv` contiene la comparación piloto anterior y no debe presentarse como resultado final.

Los juicios con nota 0 permiten analizar errores concretos: «Frankenstein» apareció como candidato para una consulta sobre un imperio galáctico; «Orgullo y prejuicio» para una distopía que controla emociones; y «Cómo ganar amigos e influir sobre las personas» para un romance fingido. Comparten palabras o temas parciales, pero no cumplen la petición completa. Las causas propuestas son hipótesis basadas en las fichas, no explicaciones causales verificadas del modelo.

## Pruebas

```bash
python -m pytest -q
cd frontend && pnpm build
```

Las pruebas cubren consulta válida, caso difícil de divergencia léxica, entrada vacía, filtros, métricas y contrato HTTP de la ruta API → recuperador → respuesta. El caso semántico difícil no presupone que el modelo mejore: esa afirmación corresponde a la evaluación etiquetada.

La prueba automatizada de navegador requiere backend y frontend activos según las instrucciones anteriores, además de Chrome/Chromium instalado. Después:

```bash
cd frontend
pnpm install
BOOKERY_CHROME_PATH='/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' pnpm test:e2e
```

En Linux, cambie `BOOKERY_CHROME_PATH` por la ruta de Chromium. La prueba valida entrada vacía y el recorrido navegador → Nuxt → FastAPI → recuperador → tarjeta.

## Limitaciones y defensa

El catálogo de demostración no es el inventario de Todo Libros y contiene 215 libros seleccionados. La cobertura creció, pero los resúmenes de metadatos de las fichas nuevas no describen la trama: consultas narrativas detalladas pueden fallar. Las consultas por Marian Rojas Estapé pueden devolver títulos relacionados por categorías editoriales amplias como «Psicología» o «Crecimiento personal», no una evaluación experta de equivalencia de contenido. El sistema puede omitir libros no registrados. El modelo de embeddings fue preentrenado para similitud de frases, no para este catálogo concreto; entradas largas pueden truncarse. BM25 y embeddings comparten campos, pero sus scores no son comparables directamente. La evaluación humana abarca solo nueve consultas; algunas fichas tienen descripciones temáticas breves, lo que limita los juicios de relevancia. No hay memoria conversacional; el formulario procesa una consulta por vez. Para una exposición de cinco minutos, consulte [el guion de presentación](docs/presentation-outline.md) y [el esquema del informe](docs/report-outline.md).

## Entregables finales

- [Informe técnico PDF](output/pdf/informe-tecnico-bookery-ai.pdf), con tabla, capturas, pruebas, análisis de errores y referencias.
- [Presentación PPTX de cinco diapositivas](output/presentation/bookery-ai-presentacion-final-v4.pptx), con notas para una exposición de cinco minutos.
- [Calificaciones humanas](data/evaluation/reviewed_queries.json) y [resultados revisados](results/evaluation_reviewed.csv).

Los scripts `tools/build_report.py` y `tools/build_presentation.mjs` generan los documentos desde el contenido del repositorio. Antes de la demostración, inicie el backend y el frontend con los comandos de instalación anteriores.
