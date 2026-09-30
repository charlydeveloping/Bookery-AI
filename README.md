# Bookery AI

Proyecto final de Machine Learning: recomendación de libros mediante una interfaz conversacional sencilla. La pregunta principal es: **¿la recuperación semántica mediante embeddings mejora el ranking de libros frente a BM25 para consultas conversacionales con preferencias explícitas?** Este módulo responde a la función de recomendaciones planteada para Todo Libros en el proyecto de grado; la integración con inventario y ventas queda fuera del alcance de esta entrega académica.

Bookery AI usa el **mismo catálogo** para ambos métodos. El backend devuelve exclusivamente títulos existentes en ese catálogo. Opcionalmente, un LLM redacta una explicación breve para cada libro recuperado por el método semántico. El frontend permite alternar el método para la demostración.

## Arquitectura

`Open Library / Google Books API → limpieza → books.json → BM25 / embeddings normalizados → FastAPI → LLM opcional → Nuxt 4`

BM25 usa `k1=1.5`, `b=0.75` y tokenización Unicode simple. La búsqueda semántica usa [`sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2), revisión `e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, un modelo **preentrenado** de 384 dimensiones. Se reutilizan sus pesos mediante `sentence-transformers==3.3.1`; aquí se desarrollaron la preparación de datos, el índice exacto NumPy, los filtros, la API y la evaluación. No se entrenó ni ajustó el modelo. Los vectores se normalizan y se ordenan por producto punto, equivalente al coseno. El artefacto guarda modelo, revisión, dimensión y huella del catálogo.

## Dataset

La fuente predeterminada es [Open Library](https://openlibrary.org/developers/api). `python -m ml.import_openlibrary` consulta seis temas, descarga las fichas de obras y conserva solo las que tienen descripción real. Se guardan `id`, `title`, `author`, `description`, `genres` (materias de la API), `language`, fecha de publicación, páginas y URL de origen. `searchable_text` concatena título, autor, materias y descripción. Se eliminan entradas sin campos requeridos o descripción de al menos 80 caracteres; se normaliza Unicode y se colapsan IDs repetidos. La descarga verificada el 29 de septiembre de 2026 con `--limit-per-query 10` conservó **51 obras**. Se incluye esa instantánea procesada; una descarga nueva puede cambiar los resultados.

Como alternativa está la [Google Books Volumes API](https://developers.google.com/books/docs/v1/using#WorkingVolumes), mediante `python -m ml.prepare_dataset`. Consulta seis materias y guarda la respuesta original; elimina HTML de las descripciones. Si la API devuelve 429, espere y vuelva a intentar o use Open Library.

Los metadatos y descripciones pueden tener derechos preexistentes; consulte la [información de licencias de Open Library](https://openlibrary.org/developers/licensing) y los [términos de Google Books](https://developers.google.com/books/terms) según la fuente elegida. El importador respeta un intervalo mayor a un segundo entre peticiones, conforme a los [límites de la API de Open Library](https://openlibrary.org/developers/api). Las categorías pueden ser amplias o inconsistentes y las descripciones pueden sesgar el ranking. Los resultados solo generalizan al catálogo descargado.

## Instalación y ejecución

Requisitos: Python 3.11, Node.js 22 y pnpm 11. Desde la raíz:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m ml.import_openlibrary
python -m ml.semantic_retriever
uvicorn backend.app.main:app --reload
```

La primera generación de embeddings descarga el modelo preentrenado. El backend sirve `http://localhost:8000/health`. En otra terminal:

```bash
cd frontend
pnpm install
pnpm dev
```

Abra `http://localhost:3000`. Para una prueba rápida de la API:

```bash
curl -X POST http://localhost:8000/api/recommend -H 'Content-Type: application/json' -d '{"query":"Quiero una novela sobre tecnología y control social","limit":5}'
```

`POST /api/search/bm25` acepta el mismo cuerpo. Ambos admiten `language`, `genre`, `author` y `limit` de 1 a 20. Los filtros se aplican de modo determinista a los dos métodos. `GET /api/options` devuelve idiomas y géneros disponibles. Una consulta vacía o un límite inválido devuelve HTTP 422. Si faltan catálogo o embeddings, el inicio del backend falla de forma explícita.

### Activar explicaciones con LLM

La aplicación funciona sin clave: presenta explicaciones básicas. Para usar un proveedor compatible con [Chat Completions y JSON mode](https://developers.openai.com/api/docs/guides/structured-outputs), configure antes de iniciar el backend:

```bash
export BOOKERY_LLM_BASE_URL=https://api.openai.com/v1
export BOOKERY_LLM_MODEL=gpt-4o-mini
export BOOKERY_LLM_API_KEY=su_clave
uvicorn backend.app.main:app --reload
```

Para Docker, copie `.env.example` a `.env`, edite `BOOKERY_LLM_MODEL` y `BOOKERY_LLM_API_KEY`, y ejecute `docker compose up --build`. No incluya `.env` ni claves en Git. También puede usar un servidor local compatible, por ejemplo [Ollama](https://registry.ollama.com/blog/openai-compatibility): `BOOKERY_LLM_BASE_URL=http://localhost:11434/v1`, `BOOKERY_LLM_MODEL=<modelo_instalado>` y clave vacía. Si Ollama está en el host y el backend en Docker, use una URL accesible desde el contenedor, como `http://host.docker.internal:11434/v1` en macOS.

El LLM recibe la consulta y los metadatos de **los libros ya recuperados**; redacta solo el campo `reason`. La API valida que las explicaciones correspondan exactamente a los ID recuperados y conserva títulos y orden del recuperador. Si el servicio falla, tarda demasiado o devuelve JSON inválido, usa explicaciones básicas. `explanation_mode` indica `llm` o `basic` en la respuesta; el frontend lo muestra. Una explicación generada todavía puede contener inferencias inexactas sobre un libro: revise manualmente las afirmaciones antes de una demostración o publicación.

También se puede usar Docker tras preparar los datos en el host:

```bash
docker compose up --build
```

## Evaluación

`data/evaluation/queries.json` contiene cinco consultas piloto con relevancias graduadas 0–3, redactadas a partir de las descripciones del catálogo **antes de medir los rankings**. Son etiquetas preliminares propuestas por el asistente, no juicios humanos independientes; por ello, cualquier cifra que produzcan es exploratoria y no una conclusión final del experimento. Antes de la defensa, una persona debe revisar las etiquetas, ampliar el conjunto y juzgar los candidatos de ambos métodos sin conocer su procedencia. Congele entonces ese conjunto y no lo use para ajustar parámetros. Los libros no anotados se tratan como relevancia 0, lo que puede subestimar ambos métodos.

```bash
python -m ml.evaluate --cases data/evaluation/queries.json --output results/evaluation.csv
```

El script ejecuta **las mismas consultas, filtros y catálogo** contra BM25 y búsqueda semántica. Reporta medias de:

- **nDCG@5:** `DCG@5 / IDCG@5`, donde `DCG@5 = Σ(2^relevancia − 1)/log₂(posición + 1)` con posiciones desde 1. Es la métrica principal y premia el orden de libros muy relevantes.
- **Precision@5:** cantidad de libros con relevancia ≥ 2 en las primeras cinco posiciones, dividida por 5.
- **Cumplimiento:** proporción de resultados que cumplen filtros explícitos en consultas que los usan; si no hay resultados por filtros, vale 1 de manera vacía y debe interpretarse junto al número de resultados.
- **Tiempo:** duración media de `search()` en milisegundos, medida tras cargar modelos e índices. Depende del hardware y no incluye HTTP ni arranque.

La ejecución piloto sobre la instantánea de 51 libros produjo `results/evaluation.csv`: BM25 nDCG@5 `0.2457` y Precision@5 `0.08`; semántica nDCG@5 `0.4668` y Precision@5 `0.16`. El cumplimiento de filtro de idioma fue `1.0` para ambos. Son mediciones reales de las cinco consultas preliminares, **no evidencia final de superioridad** por el tamaño y la procedencia de las etiquetas. La latencia se guarda en el CSV y debe medirse de nuevo en el hardware de la defensa. Estas métricas miden el **ranking antes del LLM**. La calidad de las explicaciones requiere una revisión humana aparte, por ejemplo contar afirmaciones sin respaldo en las fichas. Los ID devueltos siempre proceden del catálogo.

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

En Linux, cambie `BOOKERY_CHROME_PATH` por la ruta de Chromium. La prueba valida entrada vacía y el recorrido navegador → Nuxt → FastAPI → recuperador → tarjeta. Se ejecutó correctamente con la instantánea de 51 obras.

## Limitaciones y defensa

El catálogo público de demostración no es el inventario de Todo Libros, no cubre todos los libros y no garantiza descripciones de calidad. Una consulta como «algo parecido a Harry Potter» puede recibir resultados débiles si faltan referencias y novelas afines en el catálogo; el LLM no corrige esa ausencia. El modelo de embeddings fue preentrenado para similitud de frases, no para este catálogo concreto; entradas largas pueden truncarse. BM25 y embeddings comparten campos, pero sus scores no son comparables directamente. La calidad solo puede concluirse tras la evaluación independiente. No hay memoria conversacional; el formulario procesa una consulta por vez. Para una exposición de cinco minutos, consulte [el guion de presentación](docs/presentation-outline.md) y [el esquema del informe](docs/report-outline.md).
