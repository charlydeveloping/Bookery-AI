export default defineNuxtConfig({
  compatibilityDate: '2026-01-01',
  runtimeConfig: { public: { apiBase: process.env.NUXT_PUBLIC_API_BASE || 'http://localhost:8000' } },
  app: { head: { title: 'Bookery AI | Recomendador inteligente de libros', meta: [{ name: 'description', content: 'Descubre libros reales mediante búsqueda semántica.' }] } }
})
