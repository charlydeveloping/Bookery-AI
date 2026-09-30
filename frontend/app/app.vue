<script setup lang="ts">
type Book = { id: string; title: string; author: string; genres: string[]; description: string; score: number; reason: string; source_url?: string }
type EvaluationBook = Omit<Book, 'score' | 'reason'> & { relevance: number | null }
type EvaluationOverview = { total: number; completed: number; cases: { id: number; query: string; completed: boolean }[] }
type EvaluationCase = { id: number; query: string; books: EvaluationBook[] }
const config = useRuntimeConfig()
const view = ref<'search' | 'evaluation'>('search')
const query = ref('')
const genre = ref('')
const method = ref<'semantic_jev' | 'semantic' | 'bm25'>('semantic')
const books = ref<Book[]>([])
const options = ref<{ genres: string[] }>({ genres: [] })
const loading = ref(false)
const error = ref('')
const searched = ref(false)
const elapsed = ref<number | null>(null)
const answer = ref('')
const explanationMode = ref<'llm' | 'basic' | 'reference' | 'catalog'>('basic')
const selectionMode = ref<'jev' | 'semantic'>('semantic')
const evaluationOverview = ref<EvaluationOverview | null>(null)
const evaluationCase = ref<EvaluationCase | null>(null)
const evaluationRatings = ref<Record<string, number | null>>({})
const evaluationLoading = ref(false)
const evaluationError = ref('')
const evaluationMessage = ref('')
const evaluationResults = ref<{ method: string; ndcg_at_5: number; precision_at_5: number }[] | null>(null)

async function showEvaluation() {
  view.value = 'evaluation'
  evaluationLoading.value = true
  evaluationError.value = ''
  try {
    evaluationOverview.value = await $fetch<EvaluationOverview>(`${config.public.apiBase}/api/evaluation/cases`)
    await selectEvaluationCase(evaluationOverview.value.cases.find(item => !item.completed)?.id || 1)
  } catch { evaluationError.value = 'No se pudo abrir la evaluación. Comprueba que el backend esté activo.' }
  finally { evaluationLoading.value = false }
}

async function selectEvaluationCase(id: number) {
  evaluationLoading.value = true
  evaluationError.value = ''
  evaluationMessage.value = ''
  try {
    const selected = await $fetch<EvaluationCase>(`${config.public.apiBase}/api/evaluation/cases/${id}`)
    evaluationCase.value = selected
    evaluationRatings.value = Object.fromEntries(selected.books.map(book => [book.id, book.relevance]))
  } catch { evaluationError.value = 'No se pudieron cargar los libros de esta consulta.' }
  finally { evaluationLoading.value = false }
}

async function saveEvaluation() {
  if (!evaluationCase.value) return
  if (evaluationCase.value.books.some(book => evaluationRatings.value[book.id] == null)) {
    evaluationError.value = 'Pon una nota a todos los libros antes de guardar.'
    return
  }
  evaluationLoading.value = true
  evaluationError.value = ''
  evaluationMessage.value = ''
  try {
    evaluationOverview.value = await $fetch<EvaluationOverview>(`${config.public.apiBase}/api/evaluation/cases/${evaluationCase.value.id}/judgments`, {
      method: 'POST', body: { judgments: evaluationCase.value.books.map(book => ({ book_id: book.id, relevance: evaluationRatings.value[book.id] })) }
    })
    evaluationMessage.value = `Calificaciones guardadas: ${evaluationOverview.value.completed} de ${evaluationOverview.value.total} consultas.`
  } catch { evaluationError.value = 'No se pudieron guardar las notas. Inténtalo de nuevo.' }
  finally { evaluationLoading.value = false }
}

async function calculateEvaluation() {
  evaluationLoading.value = true
  evaluationError.value = ''
  try {
    const response = await $fetch<{ results: { method: string; ndcg_at_5: number; precision_at_5: number }[] }>(`${config.public.apiBase}/api/evaluation/calculate`, { method: 'POST' })
    evaluationResults.value = response.results
    evaluationMessage.value = 'Comparación calculada y guardada en results/evaluation_reviewed.csv.'
  } catch { evaluationError.value = 'No se pudo calcular la comparación. Revisa que todas las consultas estén calificadas.' }
  finally { evaluationLoading.value = false }
}

onMounted(async () => {
  try { options.value = await $fetch(`${config.public.apiBase}/api/options`) } catch { /* Search still works without filters. */ }
})

async function search() {
  error.value = ''
  if (!query.value.trim()) { error.value = 'Escribe qué te gustaría leer.'; return }
  loading.value = true
  searched.value = false
  try {
    const path = method.value === 'semantic_jev' ? '/api/recommend/jev' : method.value === 'semantic' ? '/api/recommend' : '/api/search/bm25'
    const response = await $fetch<{ recommendations: Book[]; response_time_ms: number; answer: string; explanation_mode: 'llm' | 'basic' | 'reference' | 'catalog'; selection_mode?: 'jev' | 'semantic' }>(`${config.public.apiBase}${path}`, {
      method: 'POST', body: { query: query.value, language: 'es', genre: genre.value || null, limit: 5 }
    })
    books.value = response.recommendations
    elapsed.value = response.response_time_ms
    answer.value = response.answer
    explanationMode.value = response.explanation_mode
    selectionMode.value = response.selection_mode || 'semantic'
    searched.value = true
  } catch { error.value = 'No se pudieron obtener recomendaciones. Comprueba que el backend esté activo.' }
  finally { loading.value = false }
}
</script>

<template>
  <main class="shell">
    <header class="top"><span class="brand">✦ Bookery <b>AI</b></span><nav class="top-nav" aria-label="Secciones"><button :class="{ active: view === 'search' }" @click="view = 'search'">Recomendar</button><button :class="{ active: view === 'evaluation' }" @click="showEvaluation">Evaluar</button></nav><span class="tag">PROYECTO DE MACHINE LEARNING</span></header>
    <template v-if="view === 'search'"><section class="hero">
      <p class="eyebrow">TU PRÓXIMA LECTURA EMPIEZA AQUÍ</p>
      <h1>Recomendador <em>inteligente</em><br>de libros</h1>
      <p class="intro">Cuéntanos qué historias te interesan. Buscaremos libros en español de nuestro catálogo académico y te mostraremos los más cercanos a tu idea.</p>
      <form @submit.prevent="search" class="searchbox">
        <label for="query">¿Qué te gustaría leer?</label>
        <textarea id="query" v-model="query" rows="3" placeholder="Quiero una novela de ciencia ficción sobre inteligencia artificial que sea fácil de leer."></textarea>
        <div class="controls">
          <select v-model="genre" aria-label="Género"><option value="">Cualquier género</option><option v-for="item in options.genres" :key="item" :value="item">{{ item }}</option></select>
          <select v-model="method" aria-label="Método"><option value="semantic_jev">Semántico + Jev</option><option value="semantic">Búsqueda semántica</option><option value="bm25">BM25</option></select>
          <button :disabled="loading" type="submit">{{ loading ? 'Buscando…' : 'Recomendar libros →' }}</button>
        </div>
      </form>
      <p v-if="error" class="error" role="alert">{{ error }}</p>
    </section>
    <section v-if="searched" class="results"><div class="section-title"><h2>{{ explanationMode === 'catalog' ? 'Consulta del catálogo' : 'Libros para ti' }}</h2><span>{{ books.length }} resultados · {{ elapsed }} ms</span></div><div class="answer" role="status"><span class="answer-label">BOOKERY AI · {{ explanationMode === 'llm' ? 'EXPLICACIONES CON IA' : explanationMode === 'reference' ? 'SIMILITUD POR CATEGORÍAS DEL CATÁLOGO' : explanationMode === 'catalog' ? 'DATOS DEL CATÁLOGO ACADÉMICO' : 'MODO BÁSICO' }}</span><p>{{ answer }}</p></div><template v-if="method === 'semantic_jev' && explanationMode !== 'catalog' && books.length"><h3 class="group-heading">{{ selectionMode === 'jev' ? 'Libro indicado para ti' : 'Primera opción según búsqueda semántica' }}</h3><article class="card featured"><div class="card-top"><span class="number">01</span><span class="genre">{{ books[0]?.genres[0] }}</span></div><h3>{{ books[0]?.title }}</h3><p class="author">{{ books[0]?.author }}</p><p class="description">{{ books[0]?.description }}</p><div class="reason"><span>POR QUÉ ESTE LIBRO</span><p>{{ books[0]?.reason }}</p></div><a v-if="books[0]?.source_url" :href="books[0]?.source_url" target="_blank" rel="noopener noreferrer">Ver fuente ↗</a></article><h3 v-if="books.length > 1" class="group-heading">Otras opciones</h3></template><div class="grid"><article v-for="(book, index) in method === 'semantic_jev' && explanationMode !== 'catalog' ? books.slice(1) : books" :key="book.id" class="card"><div class="card-top"><span class="number">0{{ index + (method === 'semantic_jev' && explanationMode !== 'catalog' ? 2 : 1) }}</span><span class="genre">{{ book.genres[0] }}</span></div><h3>{{ book.title }}</h3><p class="author">{{ book.author }}</p><p class="description">{{ book.description }}</p><div class="reason"><span>{{ explanationMode === 'catalog' ? 'FICHA DEL CATÁLOGO' : 'POR QUÉ ESTE LIBRO' }}</span><p>{{ book.reason }}</p></div><a v-if="book.source_url" :href="book.source_url" target="_blank" rel="noopener noreferrer">Ver fuente ↗</a></article></div></section>
    <section class="about"><div><p class="eyebrow">EL MÉTODO</p><h2>¿Cómo funciona?</h2></div><p>La búsqueda semántica recupera libros del catálogo. Si configuras un modelo de lenguaje, este explica por qué encajan los libros encontrados. BM25 permite comparar los resultados mediante coincidencias de palabras. Los filtros de idioma y género se aplican antes de mostrar resultados.</p></section>
    </template>
    <section v-else class="evaluation">
      <p class="eyebrow">EVALUACIÓN MANUAL</p>
      <h1>Califica las recomendaciones</h1>
      <p class="intro">Elige una consulta y compara cada libro con lo que pide. Los libros están mezclados: no verás qué método los recomendó. Pon una nota de 0 a 3 a todos y pulsa Guardar.</p>
      <div class="evaluation-scale"><strong>0</strong> No encaja <strong>1</strong> Poco <strong>2</strong> Bien <strong>3</strong> Muy bien</div>
      <p v-if="evaluationOverview" class="evaluation-progress">{{ evaluationOverview.completed }} de {{ evaluationOverview.total }} consultas guardadas</p>
      <label class="evaluation-picker">Consulta
        <select :value="evaluationCase?.id" :disabled="evaluationLoading" @change="selectEvaluationCase(Number(($event.target as HTMLSelectElement).value))">
          <option v-for="item in evaluationOverview?.cases || []" :key="item.id" :value="item.id">{{ item.id }}. {{ item.query }} {{ item.completed ? '✓' : '' }}</option>
        </select>
      </label>
      <p v-if="evaluationError" class="error" role="alert">{{ evaluationError }}</p>
      <p v-if="evaluationMessage" class="evaluation-success" role="status">{{ evaluationMessage }}</p>
      <p v-if="evaluationLoading">Cargando…</p>
      <form v-if="evaluationCase" @submit.prevent="saveEvaluation">
        <h2>{{ evaluationCase.query }}</h2>
        <div class="evaluation-list"><article v-for="book in evaluationCase.books" :key="book.id" class="evaluation-book">
          <div><h3>{{ book.title }}</h3><p class="author">{{ book.author }} · {{ book.genres.join(', ') }}</p><p>{{ book.description }}</p><a v-if="book.source_url" :href="book.source_url" target="_blank" rel="noopener noreferrer">Ver fuente ↗</a></div>
          <label :for="`rating-${book.id}`">Tu nota
            <select :id="`rating-${book.id}`" v-model.number="evaluationRatings[book.id]" :disabled="evaluationLoading"><option :value="null">Sin calificar</option><option v-for="score in [0, 1, 2, 3]" :key="score" :value="score">{{ score }}</option></select>
          </label>
        </article></div>
        <button class="evaluation-action" type="submit" :disabled="evaluationLoading">Guardar notas de esta consulta</button>
      </form>
      <button v-if="evaluationOverview?.completed === evaluationOverview?.total && evaluationOverview?.total" class="evaluation-action secondary" :disabled="evaluationLoading" @click="calculateEvaluation">Calcular comparación final</button>
      <table v-if="evaluationResults" class="evaluation-table"><thead><tr><th>Método</th><th>nDCG@5</th><th>Precisión@5</th></tr></thead><tbody><tr v-for="row in evaluationResults" :key="row.method"><td>{{ row.method === 'Semantic' ? 'Semántico' : row.method }}</td><td>{{ row.ndcg_at_5 }}</td><td>{{ row.precision_at_5 }}</td></tr></tbody></table>
    </section>
    <footer>Bookery AI · Catálogo académico de demostración en español; no representa el inventario actual de Todo Libros. Los resultados dependen de las descripciones disponibles.</footer>
  </main>
</template>

<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500&display=swap');
:root{font-family:'DM Sans',sans-serif;color:#24342d;background:#f7f6f0}*{box-sizing:border-box}body{margin:0}.shell{max-width:1180px;margin:auto;padding:0 32px}.top{display:flex;justify-content:space-between;align-items:center;padding:30px 0;border-bottom:1px solid #dedfd5}.brand{font-family:'Playfair Display',serif;font-size:25px;font-weight:700;color:#1f4b37}.brand b{color:#bd734f}.tag,.eyebrow{font-size:11px;letter-spacing:.17em;font-weight:700;color:#a36b4e}.hero{padding:90px 0 72px;max-width:900px}h1,h2,h3{font-family:'Playfair Display',serif}h1{font-size:clamp(48px,7vw,88px);line-height:1.1;letter-spacing:-.045em;margin:18px 0 26px;font-weight:600}h1 em{color:#a86f50;font-weight:500}.intro{font-size:18px;line-height:1.7;color:#637168;max-width:700px;margin-bottom:44px}.searchbox{background:white;padding:26px;border:1px solid #e0e3da;border-radius:16px;box-shadow:0 16px 35px #21382b0b}.searchbox label{font-weight:700;font-size:14px}.searchbox textarea{display:block;width:100%;resize:vertical;border:0;outline:0;font:inherit;font-size:17px;line-height:1.6;margin:16px 0 24px;min-height:92px;color:#24342d}.searchbox textarea::placeholder{color:#9ca99f}.controls{display:flex;gap:10px;flex-wrap:wrap}.controls select{background:#f4f5f0;border:1px solid #e0e4dc;border-radius:8px;padding:12px;color:#45594c;font:inherit;min-width:145px}.controls button{background:#214d38;color:white;border:0;border-radius:8px;padding:13px 20px;font:inherit;font-weight:700;cursor:pointer;margin-left:auto}.controls button:disabled{opacity:.6}.error{color:#a33131;margin-top:15px}.results{padding-bottom:90px}.section-title{display:flex;align-items:baseline;justify-content:space-between}.section-title h2,.about h2{font-size:36px}.section-title span{color:#7a887e;font-size:13px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}.card{background:white;border:1px solid #e0e3da;border-radius:12px;padding:25px;display:flex;flex-direction:column;min-height:350px}.card-top{display:flex;justify-content:space-between;gap:12px}.number{font-family:'Playfair Display',serif;color:#c0cbbf;font-size:26px}.genre{color:#a66e51;font-size:12px;text-align:right}.card h3{font-size:25px;line-height:1.2;margin:24px 0 7px}.author{color:#829087;font-size:14px;margin:0 0 15px}.description{font-size:14px;line-height:1.6;color:#53635a;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden}.reason{border-top:1px solid #e8e9e2;margin-top:auto;padding-top:16px}.reason span{font-size:10px;font-weight:700;letter-spacing:.13em;color:#a36b4e}.reason p{font-size:13px;line-height:1.5}.card a{font-size:13px;color:#1f4b37;font-weight:700;text-decoration:none}.about{display:grid;grid-template-columns:1fr 1.5fr;gap:50px;border-top:1px solid #dedfd5;padding:56px 0}.about p:last-child{line-height:1.8;color:#607065}footer{border-top:1px solid #dedfd5;padding:25px 0 50px;color:#87948b;font-size:12px}@media(max-width:700px){.shell{padding:0 20px}.top{padding:22px 0}.tag{display:none}.hero{padding:55px 0}.controls button{width:100%;margin:0}.controls select{flex:1}.about{grid-template-columns:1fr;gap:0}}
.answer{background:#edf3ec;border-left:4px solid #407655;border-radius:8px;padding:18px 22px;margin:0 0 22px}.answer-label{font-size:11px;font-weight:700;letter-spacing:.12em;color:#316547}.answer p{margin:8px 0 0;line-height:1.6}
.group-heading{font-size:27px;margin:30px 0 16px}.featured{min-height:0;margin-bottom:28px;border-color:#8baa91;background:#f4f8f1}.featured h3{font-size:33px;margin-top:15px}.featured .description{display:block;max-width:760px}.featured .reason{margin-top:18px}
.top-nav{display:flex;gap:8px}.top-nav button{border:1px solid #d7ddd2;background:white;color:#34533f;border-radius:8px;padding:9px 14px;font:inherit;cursor:pointer}.top-nav button.active{background:#214d38;color:white}.evaluation{padding:55px 0 90px}.evaluation h1{font-size:clamp(42px,6vw,66px);margin:15px 0}.evaluation h2{font-size:30px;margin:35px 0 18px}.evaluation-scale{display:flex;gap:10px;flex-wrap:wrap;background:#edf3ec;border-radius:10px;padding:16px;margin:25px 0}.evaluation-scale strong:not(:first-child){margin-left:14px}.evaluation-progress{font-weight:700;color:#214d38}.evaluation-picker{display:block;font-weight:700;margin:20px 0}.evaluation-picker select{display:block;width:100%;margin-top:8px;padding:12px;border:1px solid #d7ddd2;border-radius:8px;background:white;font:inherit}.evaluation-list{display:grid;gap:12px}.evaluation-book{display:flex;justify-content:space-between;gap:25px;background:white;border:1px solid #e0e3da;border-radius:12px;padding:22px}.evaluation-book h3{font-size:22px;margin:0 0 7px}.evaluation-book p{line-height:1.5}.evaluation-book a{color:#1f4b37}.evaluation-book label{min-width:120px;font-weight:700}.evaluation-book select{display:block;width:100%;margin-top:10px;padding:10px;border:1px solid #d7ddd2;border-radius:8px;font:inherit}.evaluation-action{background:#214d38;color:white;border:0;border-radius:8px;padding:13px 20px;font:inherit;font-weight:700;cursor:pointer;margin-top:20px}.evaluation-action:disabled{opacity:.6}.evaluation-action.secondary{background:#a86f50}.evaluation-success{color:#1f673c;font-weight:700}.evaluation-table{border-collapse:collapse;margin-top:22px;width:100%;background:white}.evaluation-table th,.evaluation-table td{padding:14px;border:1px solid #e0e3da;text-align:left}@media(max-width:700px){.evaluation-book{display:block}.evaluation-book label{display:block;margin-top:18px}.top-nav{margin-left:auto}.top-nav button{padding:8px}.evaluation{padding-top:35px}}
</style>
