import { chromium } from 'playwright-core'

const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.BOOKERY_CHROME_PATH || undefined
})

try {
  const page = await browser.newPage()
  await page.goto(process.env.BOOKERY_FRONTEND_URL || 'http://localhost:3000')
  await page.getByRole('button', { name: /Recomendar libros/ }).click()
  await page.getByRole('alert').getByText('Escribe qué te gustaría leer.').waitFor()
  await page.getByLabel('¿Qué te gustaría leer?').fill('Quiero ciencia ficción sobre un imperio galáctico')
  await page.getByRole('button', { name: /Recomendar libros/ }).click()
  await page.getByRole('heading', { name: 'Libros para ti' }).waitFor()
  const cards = page.locator('.card')
  if (await cards.count() < 1) throw new Error('No recommendation cards rendered')
  const title = await cards.first().locator('h3').innerText()
  if (!title) throw new Error('First recommendation has no title')
  console.log(`E2E passed: browser → Nuxt → API → retriever → ${title}`)
} finally {
  await browser.close()
}
