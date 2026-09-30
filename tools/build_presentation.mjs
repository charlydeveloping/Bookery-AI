"use strict";

import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = path.resolve(path.dirname(new URL(import.meta.url).pathname), "..");
const SKILL_DIR = "/Users/carlos/.codex/plugins/cache/openai-primary-runtime/presentations/26.909.12148/skills/presentations";
const RUNTIME_PYTHON = "/Users/carlos/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3";
const FINAL_PPTX = path.join(workspaceDir, "output/presentation/bookery-ai-presentacion-final-v3.pptx");
const TMP_DIR = path.join(workspaceDir, ".codex-finalizer");
const { resolvePresentationFont, finalizePresentation } = await import(
  pathToFileURL(path.join(SKILL_DIR, "container_tools/artifact_tool_utils.mjs")).href,
);

const family = resolvePresentationFont({ fontFamily: "Arial" });
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const green = "#214d38";
const clay = "#a86f50";
const ink = "#24342d";
const paper = "#f7f6f0";
const mist = "#edf3ec";

function text(slide, value, x, y, w, h, size = 27, color = ink, bold = false) {
  const box = slide.shapes.add({ geometry: "textbox", position: { left: x, top: y, width: w, height: h },
    fill: "none", line: { fill: "none", width: 0 } });
  box.text = value;
  box.text.style = { typeface: family, fontSize: size, color, bold, autoFit: "none" };
  return box;
}

function rect(slide, x, y, w, h, fill) {
  return slide.shapes.add({ geometry: "rect", position: { left: x, top: y, width: w, height: h },
    fill, line: { fill: "none", width: 0 } });
}

function base(number, title, subtitle = "") {
  const slide = presentation.slides.add();
  slide.background.fill = paper;
  rect(slide, 0, 0, 1280, 15, green);
  text(slide, title, 74, 46, 1120, 85, 43, green, true);
  if (subtitle) text(slide, subtitle, 76, 130, 1080, 50, 22, ink);
  text(slide, `BOOKERY AI  ·  ${number}/5`, 76, 663, 460, 25, 15, clay, true);
  return slide;
}

// Slide 1: problem and product.
{
  const slide = base(1, "Recomendar libros con una consulta", "Proyecto final de Machine Learning");
  text(slide, "El usuario escribe qué quiere leer.\nBookery AI devuelve cinco libros del catálogo académico en español.",
    78, 225, 980, 180, 33, ink);
  rect(slide, 78, 455, 8, 95, clay);
  text(slide, "215 libros · 28 categorías\nModelo semántico y línea base BM25", 112, 460, 910, 95, 27, green, true);
  slide.speakerNotes.textFrame.setText(
    "Tiempo sugerido: 45 s. Problema: una consulta conversacional requiere ordenar libros relevantes. Objetivo: comparar semántica con BM25. Datos del catálogo y arquitectura descritos en README.md.");
}

// Slide 2: method and validation.
{
  const slide = base(2, "Dos métodos, las mismas consultas", "BM25 es la referencia; el modelo semántico reutiliza pesos preentrenados.");
  rect(slide, 76, 240, 530, 8, clay);
  text(slide, "BM25", 76, 268, 510, 58, 35, green, true);
  text(slide, "Ordena por coincidencia de palabras\nen título, autor, géneros y descripción.", 76, 337, 520, 105, 25, ink);
  rect(slide, 680, 240, 530, 8, clay);
  text(slide, "Semántico", 680, 268, 510, 58, 35, green, true);
  text(slide, "Compara embeddings normalizados\ncon similitud coseno.", 680, 337, 520, 105, 25, ink);
  text(slide, "Validación: 9 consultas · 71 pares consulta-libro · notas humanas de 0 a 3",
    76, 516, 1120, 70, 25, green, true);
  slide.speakerNotes.textFrame.setText(
    "Tiempo sugerido: 55 s. Modelo: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, revisión e8f8c211226b894fcb81acc59f3b34ba3efd5f42. No se entrenó en este proyecto. Los candidatos de los dos métodos se mezclaron sin revelar procedencia y se calificaron de 0 a 3. Fuente: README.md y data/evaluation/reviewed_queries.json.");
}

// Slide 3: the main result.
{
  const slide = base(3, "El semántico ordenó mejor", "Resultado de 9 consultas calificadas manualmente");
  text(slide, "nDCG@5", 76, 230, 280, 46, 28, green, true);
  text(slide, "Semántico", 76, 305, 240, 40, 22, ink);
  rect(slide, 330, 307, 740 * 0.8998, 32, green);
  text(slide, "0,8998", 1020, 304, 170, 42, 25, green, true);
  text(slide, "BM25", 76, 369, 240, 40, 22, ink);
  rect(slide, 330, 371, 740 * 0.8456, 32, clay);
  text(slide, "0,8456", 1020, 368, 170, 42, 25, clay, true);
  rect(slide, 76, 467, 1124, 2, "#d8dfd6");
  text(slide, "Precision@5: BM25 0,4444 · Semántico 0,4222", 76, 502, 1080, 45, 26, ink);
  text(slide, "BM25 incluyó un libro útil más entre 45 posiciones evaluadas.", 76, 558, 1080, 40, 21, ink);
  slide.speakerNotes.textFrame.setText(
    "Tiempo sugerido: 45 s. nDCG@5 es la métrica principal y evalúa el orden de los cinco primeros libros. El semántico obtuvo 0,8998 y BM25 0,8456. En Precision@5, BM25 obtuvo 20 libros con nota 2 o 3 entre 45 posiciones, y el semántico 19. Fuente: results/evaluation_reviewed.csv. Muestra pequeña; no afirmar generalización.");
}

// Slide 4: demonstration.
{
  const slide = base(4, "Demostración del sistema", "Frontend → API FastAPI → recuperador → resultados del catálogo");
  const bytes = await fs.readFile(path.join(workspaceDir, "docs/captura-frontend.png"));
  slide.images.add({ blob: bytes, contentType: "image/png", alt: "Consulta en la interfaz de Bookery AI",
    fit: "contain",
    position: { left: 60, top: 215, width: 690, height: 385 } });
  text(slide, "Caso válido", 790, 231, 420, 45, 29, green, true);
  text(slide, "“Imperio galáctico” → Fundación", 790, 289, 420, 85, 25, ink);
  text(slide, "Límite observado", 790, 402, 420, 45, 29, clay, true);
  text(slide, "“Vigilancia digital” también trajo libros poco relacionados.", 790, 460, 420, 115, 24, ink);
  slide.speakerNotes.textFrame.setText(
    "Tiempo sugerido: 120 s. Demostrar en vivo una consulta válida y la entrada vacía. Si hay tiempo, mostrar vigilancia digital como caso difícil. La captura es respaldo, no reemplaza la aplicación. Fuente: prueba local del frontend y docs/captura-frontend.png.");
}

// Slide 5: conclusion and handoff.
{
  const slide = base(5, "Conclusión y límites");
  text(slide, "En estas nueve consultas, la búsqueda semántica mejoró el orden frente a BM25.",
    78, 218, 1100, 110, 33, green, true);
  rect(slide, 78, 360, 8, 155, clay);
  text(slide, "El catálogo es limitado y muchas fichas solo tienen temas.\nSemántico + Jev no forma parte de esta evaluación.",
    112, 365, 1060, 145, 26, ink);
  text(slide, "Repositorio: github.com/charlydeveloping/Bookery-AI", 78, 555, 1100, 55, 23, green, true);
  slide.speakerNotes.textFrame.setText(
    "Tiempo sugerido: 55 s. Explicar que el resultado vale para estas nueve consultas y este catálogo. La evaluación no mide la calidad de explicaciones LLM ni la variante Jev. Mostrar el repositorio y señalar README, informe y CSV revisado. Total sugerido: 5 minutos.");
}

await fs.mkdir(TMP_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL_PPTX), { recursive: true });
const candidatePath = path.join(TMP_DIR, "candidate-bookery-v3.pptx");
await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
const result = await finalizePresentation({
  workspaceDir,
  candidatePath,
  finalPath: FINAL_PPTX,
  pythonExecutable: RUNTIME_PYTHON,
  integrityValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_package_integrity.py"),
  layoutValidatorPath: path.join(SKILL_DIR, "container_tools/inspect_presentation_layout_geometry.py"),
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-heading-fit"],
  explicitTotalSlideCount: 5,
  requiredNativeTableOwnerSlides: [],
  requiredNativeChartOwnerSlides: [],
  fontPolicy: { basis: "design", families: [family] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(TMP_DIR, "bookery-validation-v3.json"),
});
for (let index = 0; index < presentation.slides.length; index++) {
  const slide = presentation.slides.getByIndex(index);
  const preview = await presentation.export({ slide, format: "png", scale: 1 });
  await fs.writeFile(path.join(TMP_DIR, `slide-${index + 1}.png`), new Uint8Array(await preview.arrayBuffer()));
}
console.log(JSON.stringify({ output: FINAL_PPTX, result }));
