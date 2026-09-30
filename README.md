# Bookery AI

Proyecto final de Machine Learning: recomendación de libros mediante una interfaz conversacional sencilla. La pregunta principal es: **¿la recuperación semántica mediante embeddings mejora el ranking de libros frente a BM25 para consultas conversacionales con preferencias explícitas?** Este módulo responde a la función de recomendaciones planteada para Todo Libros en el proyecto de grado; la integración con inventario y ventas queda fuera del alcance de esta entrega académica.

Bookery AI usa el **mismo catálogo** para ambos métodos. El backend devuelve exclusivamente títulos existentes en ese catálogo. Opcionalmente, un LLM redacta una explicación breve para cada libro recuperado por el método semántico. El frontend permite alternar el método para la demostración.

## Arquitectura

`Open Library (ediciones en español) + anotaciones propias → books.json → BM25 / embeddings normalizados → FastAPI → LLM opcional → Nuxt 4`

BM25 usa `k1=1.5`, `b=0.75` y tokenización Unicode simple. La búsqueda semántica usa [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), revisión `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, un modelo **preentrenado** de 384 dimensiones. Se reutilizan sus pesos mediante `sentence-transformers==3.3.1`; aquí se desarrollaron la preparación de datos, el índice exacto NumPy, los filtros, la API y la evaluación. No se entrenó ni ajustó el modelo. Los vectores se normalizan y se ordenan por producto punto, equivalente al coseno. El artefacto guarda modelo, revisión, dimensión y huella del catálogo.

## Dataset

El catálogo de demostración contiene **53 libros en español**: 42 novelas de varios géneros, ocho títulos de psicología y crecimiento personal de otros autores, y tres obras de Marian Rojas Estapé para usarlas como referentes. Para los primeros 50, los identificadores de obra, autores, años y ediciones españolas provienen de consultas a la [API de búsqueda de Open Library](https://openlibrary.org/dev/docs/api/search) realizadas el 30 de septiembre de 2026. [La instantánea de procedencia](data/curated/openlibrary_editions_es.json) registra los IDs y títulos de sus ediciones `spa`. Las tres fichas de Marian Rojas Estapé se verificaron con las páginas de [Cómo hacer que te pasen cosas buenas](https://www.planetadelibros.com/libro-como-hacer-que-te-pasen-cosas-buenas-edicion-especial/302877), [Encuentra tu persona vitamina](https://www.planetadelibros.com/libro-encuentra-tu-persona-vitamina/329152) y [Recupera tu mente, reconquista tu vida](https://www.planetadelibros.com/libro-recupera-tu-mente-reconquista-tu-vida/383546) de Editorial Planeta. [Las anotaciones](data/curated/annotations_es.json) contienen géneros controlados y sinopsis breves **redactadas para este proyecto**; no son descripciones oficiales ni pruebas de existencias en la librería. El catálogo es una selección intencional y pequeña, con sesgo hacia títulos conocidos.

`python -m ml.build_curated_catalog` une las anotaciones con los metadatos de Open Library y las tres fichas editoriales, y genera `data/processed/books.json` de forma determinista. Se guardan `id`, `title`, `author`, `description`, `genres`, `language=es`, año, ID de edición o ISBN, URL y fuente. `searchable_text` concatena título, autor, géneros y sinopsis. Una ficha incompleta o duplicada produce un error. Los antiguos importadores `ml.import_openlibrary` y `ml.prepare_dataset` sirven para explorar otras fuentes; sus salidas sin revisión no sustituyen el catálogo curado en español.

Los metadatos de Open Library pueden tener derechos preexistentes; consulte su [información de licencias](https://openlibrary.org/developers/licensing). Las sinopsis de este repositorio son originales. La selección y redacción manual pueden favorecer ciertos temas y títulos; los resultados solo generalizan a estos 53 registros. El catálogo no registra precio ni disponibilidad de Todo Libros.

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

### Activar explicaciones con LLM

La aplicación funciona sin clave: presenta explicaciones básicas. Para usar un proveedor compatible con [Chat Completions y JSON mode](https://developers.openai.com/api/docs/guides/structured-outputs), copie `.env.example` a `.env`, configure modelo y clave y reinicie el backend. El backend **sí carga `.env` automáticamente**. También puede exportar las variables antes de iniciarlo:

```bash
export BOOKERY_LLM_BASE_URL=https://api.openai.com/v1
export BOOKERY_LLM_MODEL=gpt-4o-mini
export BOOKERY_LLM_API_KEY=su_clave
uvicorn backend.app.main:app --reload
```

Para Docker, copie `.env.example` a `.env`, edite `BOOKERY_LLM_MODEL` y `BOOKERY_LLM_API_KEY`, y ejecute `docker compose up --build`. No incluya `.env` ni claves en Git. También puede usar un servidor local compatible, por ejemplo [Ollama](https://registry.ollama.com/blog/openai-compatibility): `BOOKERY_LLM_BASE_URL=http://localhost:11434/v1`, `BOOKERY_LLM_MODEL=<modelo_instalado>` y clave vacía. Si Ollama está en el host y el backend en Docker, use una URL accesible desde el contenedor, como `http://host.docker.internal:11434/v1` en macOS.

El LLM recibe la consulta y los metadatos de **los libros ya recuperados**; redacta solo el campo `reason`. La API valida que las explicaciones correspondan exactamente a los ID recuperados y conserva títulos y orden del recuperador. Si el servicio falla, tarda demasiado o devuelve JSON inválido, usa explicaciones básicas. `explanation_mode` indica `llm`, `basic`, `reference` o `catalog` en la respuesta; el frontend lo muestra. Una explicación generada todavía puede contener inferencias inexactas sobre un libro: revise manualmente las afirmaciones antes de una demostración o publicación.

También se puede usar Docker tras preparar los datos en el host:

```bash
docker compose up --build
```

## Evaluación

`data/evaluation/queries.json` contiene nueve consultas piloto en español con relevancias graduadas 0–3, redactadas y etiquetadas a partir de las fichas **antes de medir los rankings**. Incluye consultas por título y por autor. Son etiquetas preliminares propuestas por el asistente, no juicios humanos independientes; por ello, las cifras son exploratorias. Antes de la defensa, una persona debe revisar las etiquetas y juzgar candidatos de ambos métodos sin conocer su procedencia. Congele entonces ese conjunto y no lo use para ajustar parámetros. Los libros no anotados se tratan como relevancia 0, lo que puede subestimar ambos métodos.

```bash
python -m ml.evaluate --cases data/evaluation/queries.json --output results/evaluation.csv
```

El script ejecuta **las mismas consultas, filtros y catálogo** contra BM25 y búsqueda semántica. Reporta medias de:

- **nDCG@5:** `DCG@5 / IDCG@5`, donde `DCG@5 = Σ(2^relevancia − 1)/log₂(posición + 1)` con posiciones desde 1. Es la métrica principal y premia el orden de libros muy relevantes.
- **Precision@5:** cantidad de libros con relevancia ≥ 2 en las primeras cinco posiciones, dividida por 5.
- **Cumplimiento:** proporción de resultados que cumplen filtros explícitos en consultas que los usan; si no hay resultados por filtros, vale 1 de manera vacía y debe interpretarse junto al número de resultados.
- **Tiempo:** duración media de `search()` en milisegundos, medida tras cargar modelos e índices. Depende del hardware y no incluye HTTP ni arranque.

La ejecución piloto sobre los 53 libros produjo `results/evaluation.csv`: BM25 nDCG@5 `0.8735` y Precision@5 `0.2667`; semántica nDCG@5 `0.8357` y Precision@5 `0.3333`. El cumplimiento de filtros fue `1.0` para ambos. Son mediciones de nueve consultas preliminares, **no evidencia final de superioridad** por el tamaño y la procedencia de las etiquetas. La latencia se guarda en el CSV y debe medirse de nuevo en el hardware de la defensa. Estas métricas miden el ranking, no la calidad de las explicaciones del LLM. Las consultas por referencias de autor usan explicaciones basadas directamente en temas anotados. Los ID devueltos siempre proceden del catálogo.

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

El catálogo de demostración no es el inventario de Todo Libros y contiene solo 53 libros seleccionados. Las consultas por Marian Rojas Estapé pueden devolver títulos relacionados por categorías editoriales amplias como «Psicología» o «Crecimiento personal», no una evaluación experta de equivalencia de contenido. El sistema puede omitir libros no registrados. El modelo de embeddings fue preentrenado para similitud de frases, no para este catálogo concreto; entradas largas pueden truncarse. BM25 y embeddings comparten campos, pero sus scores no son comparables directamente. La calidad solo puede concluirse tras la evaluación independiente. No hay memoria conversacional; el formulario procesa una consulta por vez. Para una exposición de cinco minutos, consulte [el guion de presentación](docs/presentation-outline.md) y [el esquema del informe](docs/report-outline.md).
