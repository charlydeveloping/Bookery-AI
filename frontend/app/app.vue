<script setup lang="ts">
type Book = { id: string; title: string; author: string; genres: string[]; description: string; score: number; reason: string; source_url?: string }
const config = useRuntimeConfig()
const query = ref('')
const genre = ref('')
const method = ref<'semantic' | 'bm25'>('semantic')
const books = ref<Book[]>([])
const options = ref<{ genres: string[] }>({ genres: [] })
const loading = ref(false)
const error = ref('')
const searched = ref(false)
const elapsed = ref<number | null>(null)
const answer = ref('')
const explanationMode = ref<'llm' | 'basic' | 'reference'>('basic')

onMounted(async () => {
  try { options.value = await $fetch(`${config.public.apiBase}/api/options`) } catch { /* Search still works without filters. */ }
})

async function search() {
  error.value = ''
  if (!query.value.trim()) { error.value = 'Escribe qué te gustaría leer.'; return }
  loading.value = true
  searched.value = false
  try {
    const path = method.value === 'semantic' ? '/api/recommend' : '/api/search/bm25'
    const response = await $fetch<{ recommendations: Book[]; response_time_ms: number; answer: string; explanation_mode: 'llm' | 'basic' | 'reference' }>(`${config.public.apiBase}${path}`, {
      method: 'POST', body: { query: query.value, language: 'es', genre: genre.value || null, limit: 5 }
    })
    books.value = response.recommendations
    elapsed.value = response.response_time_ms
    answer.value = response.answer
    explanationMode.value = response.explanation_mode
    searched.value = true
  } catch { error.value = 'No se pudieron obtener recomendaciones. Comprueba que el backend esté activo.' }
  finally { loading.value = false }
}
</script>

<template>
  <main class="shell">
    <header class="top"><span class="brand">✦ Bookery <b>AI</b></span><span class="tag">PROYECTO DE MACHINE LEARNING</span></header>
    <section class="hero">
      <p class="eyebrow">TU PRÓXIMA LECTURA EMPIEZA AQUÍ</p>
      <h1>Recomendador <em>inteligente</em><br>de libros</h1>
      <p class="intro">Cuéntanos qué historias te interesan. Buscaremos libros en español de nuestro catálogo académico y te mostraremos los más cercanos a tu idea.</p>
      <form @submit.prevent="search" class="searchbox">
        <label for="query">¿Qué te gustaría leer?</label>
        <textarea id="query" v-model="query" rows="3" placeholder="Quiero una novela de ciencia ficción sobre inteligencia artificial que sea fácil de leer."></textarea>
        <div class="controls">
          <select v-model="genre" aria-label="Género"><option value="">Cualquier género</option><option v-for="item in options.genres" :key="item" :value="item">{{ item }}</option></select>
          <select v-model="method" aria-label="Método"><option value="semantic">Búsqueda semántica</option><option value="bm25">BM25</option></select>
          <button :disabled="loading" type="submit">{{ loading ? 'Buscando…' : 'Recomendar libros →' }}</button>
        </div>
      </form>
      <p v-if="error" class="error" role="alert">{{ error }}</p>
    </section>
    <section v-if="searched" class="results"><div class="section-title"><h2>Libros para ti</h2><span>{{ books.length }} resultados · {{ elapsed }} ms</span></div><div class="answer" role="status"><span class="answer-label">BOOKERY AI · {{ explanationMode === 'llm' ? 'EXPLICACIONES CON IA' : explanationMode === 'reference' ? 'SIMILITUD POR TEMAS DEL CATÁLOGO' : 'MODO BÁSICO' }}</span><p>{{ answer }}</p></div><div class="grid"><article v-for="(book, index) in books" :key="book.id" class="card"><div class="card-top"><span class="number">0{{ index + 1 }}</span><span class="genre">{{ book.genres[0] }}</span></div><h3>{{ book.title }}</h3><p class="author">{{ book.author }}</p><p class="description">{{ book.description }}</p><div class="reason"><span>POR QUÉ ESTE LIBRO</span><p>{{ book.reason }}</p></div><a v-if="book.source_url" :href="book.source_url" target="_blank" rel="noopener noreferrer">Ver fuente ↗</a></article></div></section>
    <section class="about"><div><p class="eyebrow">EL MÉTODO</p><h2>¿Cómo funciona?</h2></div><p>La búsqueda semántica recupera libros del catálogo. Si configuras un modelo de lenguaje, este explica por qué encajan los libros encontrados. BM25 permite comparar los resultados mediante coincidencias de palabras. Los filtros de idioma y género se aplican antes de mostrar resultados.</p></section>
    <footer>Bookery AI · Catálogo académico de demostración en español; no representa el inventario actual de Todo Libros. Los resultados dependen de las descripciones disponibles.</footer>
  </main>
</template>

<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:ital,wght@0,500;0,600;0,700;1,500&display=swap');
:root{font-family:'DM Sans',sans-serif;color:#24342d;background:#f7f6f0}*{box-sizing:border-box}body{margin:0}.shell{max-width:1180px;margin:auto;padding:0 32px}.top{display:flex;justify-content:space-between;align-items:center;padding:30px 0;border-bottom:1px solid #dedfd5}.brand{font-family:'Playfair Display',serif;font-size:25px;font-weight:700;color:#1f4b37}.brand b{color:#bd734f}.tag,.eyebrow{font-size:11px;letter-spacing:.17em;font-weight:700;color:#a36b4e}.hero{padding:90px 0 72px;max-width:900px}h1,h2,h3{font-family:'Playfair Display',serif}h1{font-size:clamp(48px,7vw,88px);line-height:1.1;letter-spacing:-.045em;margin:18px 0 26px;font-weight:600}h1 em{color:#a86f50;font-weight:500}.intro{font-size:18px;line-height:1.7;color:#637168;max-width:700px;margin-bottom:44px}.searchbox{background:white;padding:26px;border:1px solid #e0e3da;border-radius:16px;box-shadow:0 16px 35px #21382b0b}.searchbox label{font-weight:700;font-size:14px}.searchbox textarea{display:block;width:100%;resize:vertical;border:0;outline:0;font:inherit;font-size:17px;line-height:1.6;margin:16px 0 24px;min-height:92px;color:#24342d}.searchbox textarea::placeholder{color:#9ca99f}.controls{display:flex;gap:10px;flex-wrap:wrap}.controls select{background:#f4f5f0;border:1px solid #e0e4dc;border-radius:8px;padding:12px;color:#45594c;font:inherit;min-width:145px}.controls button{background:#214d38;color:white;border:0;border-radius:8px;padding:13px 20px;font:inherit;font-weight:700;cursor:pointer;margin-left:auto}.controls button:disabled{opacity:.6}.error{color:#a33131;margin-top:15px}.results{padding-bottom:90px}.section-title{display:flex;align-items:baseline;justify-content:space-between}.section-title h2,.about h2{font-size:36px}.section-title span{color:#7a887e;font-size:13px}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}.card{background:white;border:1px solid #e0e3da;border-radius:12px;padding:25px;display:flex;flex-direction:column;min-height:350px}.card-top{display:flex;justify-content:space-between;gap:12px}.number{font-family:'Playfair Display',serif;color:#c0cbbf;font-size:26px}.genre{color:#a66e51;font-size:12px;text-align:right}.card h3{font-size:25px;line-height:1.2;margin:24px 0 7px}.author{color:#829087;font-size:14px;margin:0 0 15px}.description{font-size:14px;line-height:1.6;color:#53635a;display:-webkit-box;-webkit-line-clamp:4;-webkit-box-orient:vertical;overflow:hidden}.reason{border-top:1px solid #e8e9e2;margin-top:auto;padding-top:16px}.reason span{font-size:10px;font-weight:700;letter-spacing:.13em;color:#a36b4e}.reason p{font-size:13px;line-height:1.5}.card a{font-size:13px;color:#1f4b37;font-weight:700;text-decoration:none}.about{display:grid;grid-template-columns:1fr 1.5fr;gap:50px;border-top:1px solid #dedfd5;padding:56px 0}.about p:last-child{line-height:1.8;color:#607065}footer{border-top:1px solid #dedfd5;padding:25px 0 50px;color:#87948b;font-size:12px}@media(max-width:700px){.shell{padding:0 20px}.top{padding:22px 0}.tag{display:none}.hero{padding:55px 0}.controls button{width:100%;margin:0}.controls select{flex:1}.about{grid-template-columns:1fr;gap:0}}
.answer{background:#edf3ec;border-left:4px solid #407655;border-radius:8px;padding:18px 22px;margin:0 0 22px}.answer-label{font-size:11px;font-weight:700;letter-spacing:.12em;color:#316547}.answer p{margin:8px 0 0;line-height:1.6}
</style>
