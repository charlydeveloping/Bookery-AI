"""Build the Spanish technical report from the reviewed project results."""

import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph,
                               SimpleDocTemplate, Spacer, Table, TableStyle)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/informe-tecnico-bookery-ai.pdf"
FONT_DIR = Path("/System/Library/Fonts/Supplemental")
pdfmetrics.registerFont(TTFont("ArialBookery", str(FONT_DIR / "Arial.ttf")))
pdfmetrics.registerFont(TTFont("ArialBookeryBold", str(FONT_DIR / "Arial Bold.ttf")))

GREEN = colors.HexColor("#214d38")
CLAY = colors.HexColor("#a86f50")
INK = colors.HexColor("#24342d")
LIGHT = colors.HexColor("#edf3ec")

styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name="TitleBookery", fontName="ArialBookeryBold", fontSize=25,
                          leading=30, textColor=GREEN, spaceAfter=14))
styles.add(ParagraphStyle(name="DeckBookery", fontName="ArialBookery", fontSize=11,
                          leading=16, textColor=INK, spaceAfter=15))
styles.add(ParagraphStyle(name="H1Bookery", fontName="ArialBookeryBold", fontSize=14,
                          leading=18, textColor=GREEN, spaceBefore=13, spaceAfter=7))
styles.add(ParagraphStyle(name="BodyBookery", fontName="ArialBookery", fontSize=9.2,
                          leading=13.3, textColor=INK, spaceAfter=7))
styles.add(ParagraphStyle(name="SmallBookery", fontName="ArialBookery", fontSize=7.7,
                          leading=11, textColor=INK, spaceAfter=5))
styles.add(ParagraphStyle(name="CaptionBookery", fontName="ArialBookery", fontSize=8,
                          leading=11, textColor=CLAY, alignment=TA_LEFT, spaceBefore=4, spaceAfter=8))


def paragraph(text, style="BodyBookery"):
    return Paragraph(text, styles[style])


def heading(text):
    return paragraph(text, "H1Bookery")


def figure(path, width, caption):
    image = Image(str(ROOT / path))
    scale = width / image.imageWidth
    image.drawWidth = width
    image.drawHeight = image.imageHeight * scale
    return KeepTogether([image, paragraph(caption, "CaptionBookery")])


def page(canvas, document):
    canvas.setStrokeColor(colors.HexColor("#d8dfd6"))
    canvas.line(1.8 * cm, 1.5 * cm, A4[0] - 1.8 * cm, 1.5 * cm)
    canvas.setFont("ArialBookery", 8)
    canvas.setFillColor(colors.HexColor("#66736a"))
    canvas.drawString(1.8 * cm, 1.1 * cm, "Bookery AI  |  Proyecto final de Machine Learning")
    canvas.drawRightString(A4[0] - 1.8 * cm, 1.1 * cm, str(document.page))


with (ROOT / "results/evaluation_reviewed.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
by_method = {row["method"]: row for row in rows}

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
doc = SimpleDocTemplate(str(OUTPUT), pagesize=A4, rightMargin=1.8 * cm,
                        leftMargin=1.8 * cm, topMargin=1.6 * cm, bottomMargin=1.9 * cm)
story = []

story += [paragraph("Bookery AI", "TitleBookery"),
          paragraph("Recomendación de libros en español mediante recuperación semántica: comparación con BM25", "DeckBookery"),
          heading("1. Problema y objetivo"),
          paragraph("El usuario describe el libro que busca en lenguaje natural. El sistema devuelve hasta cinco títulos de un catálogo académico y una explicación basada en su ficha. La pregunta experimental es si la recuperación semántica ordena mejor los libros relevantes que BM25 para consultas conversacionales en español."),
          paragraph("La unidad evaluada es el par consulta-libro. El alcance es la recomendación; la aplicación no consulta inventario, precios ni ventas de Todo Libros."),
          heading("2. Datos y preparación"),
          paragraph("El catálogo contiene 215 libros con edición en español y 28 categorías principales. Combina fichas revisadas para el proyecto con obras seleccionadas de Open Library. Cada registro incluye identificador, título, autor, descripción, géneros, idioma, referencia de edición y fuente. Se eliminaron duplicados por ID y se exigieron campos obligatorios. El texto recuperable concatena título, autor, géneros y descripción."),
          paragraph("Parte de las fichas ampliadas solo tiene metadatos temáticos, sin sinopsis argumental verificada. La selección no representa todo el mercado editorial. El catálogo y sus anotaciones están versionados en <b>data/</b>; el índice de embeddings se regenera con el comando del README."),
          heading("3. Arquitectura funcional"),
          paragraph("Catálogo revisado → normalización y texto recuperable → BM25 / embeddings → FastAPI → interfaz Nuxt. La API valida la entrada, aplica filtros y ejecuta el recuperador seleccionado. Un modelo de lenguaje opcional redacta motivos para fichas con sinopsis revisada; sus frases no modifican el orden de los libros."),
          figure("docs/captura-frontend.png", 15.5 * cm,
                 "Figura 1. Consulta real en la interfaz; el método semántico está seleccionado por defecto."),
          PageBreak()]

story += [heading("4. Línea base y modelo propuesto"),
          paragraph("<b>BM25</b> es la línea base léxica: puntúa coincidencias de términos en el mismo texto de catálogo, con k1 = 1,5 y b = 0,75. <b>Semántico</b> reutiliza sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, revisión e8f8c211226b894fcb81acc59f3b34ba3efd5f42, de 384 dimensiones. No se entrenaron ni ajustaron pesos. Los vectores normalizados se ordenan por producto punto, equivalente a similitud coseno. Ambos métodos reciben las mismas consultas, filtros y catálogo."),
          heading("5. Validación y métricas"),
          paragraph("Se compararon nueve consultas fijas en español. Para cada una se reunieron los cinco primeros resultados de ambos métodos, se quitaron duplicados y una persona calificó 71 pares consulta-libro sin ver qué método propuso cada candidato. La escala fue 0 = no encaja, 1 = poco, 2 = bien y 3 = muy bien. Los parámetros del modelo no se ajustaron con estas notas."),
          paragraph("La métrica principal es <b>nDCG@5</b>, que premia situar arriba los libros con mayor relevancia. <b>Precision@5</b> cuenta los libros con nota 2 o 3 entre los cinco primeros. También se registra cumplimiento de filtros y duración media de search() tras cargar los índices. La latencia no incluye HTTP, LLM ni arranque."),
          heading("6. Resultados revisados"),
          Table([["Método", "nDCG@5", "Precision@5", "Filtros", "Tiempo medio"],
                 ["BM25", by_method["BM25"]["ndcg_at_5"], by_method["BM25"]["precision_at_5"],
                  by_method["BM25"]["preference_compliance"], f'{by_method["BM25"]["response_time_ms"]} ms'],
                 ["Semántico", by_method["Semantic"]["ndcg_at_5"], by_method["Semantic"]["precision_at_5"],
                  by_method["Semantic"]["preference_compliance"], f'{by_method["Semantic"]["response_time_ms"]} ms']],
                colWidths=[4.0 * cm, 2.8 * cm, 3.1 * cm, 2.3 * cm, 3.2 * cm],
                style=TableStyle([("BACKGROUND", (0, 0), (-1, 0), GREEN),
                                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                  ("FONTNAME", (0, 0), (-1, -1), "ArialBookery"),
                                  ("FONTNAME", (0, 0), (-1, 0), "ArialBookeryBold"),
                                  ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                                  ("BACKGROUND", (0, 2), (-1, 2), LIGHT),
                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                                  ("TOPPADDING", (0, 0), (-1, -1), 9),
                                  ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d8dfd6"))])),
          Spacer(1, 8),
          paragraph("El semántico obtuvo mejor nDCG@5 (0,8998 frente a 0,8456), por lo que ordenó mejor los candidatos según la métrica principal. BM25 obtuvo Precision@5 ligeramente mayor (0,4444 frente a 0,4222): equivale a 20 libros útiles frente a 19 en 45 posiciones evaluadas por método."),
          figure("docs/captura-resultados-evaluacion.png", 15.5 * cm,
                 "Figura 2. Comparación calculada desde la interfaz después de guardar las notas humanas."),
          PageBreak()]

story += [heading("7. Análisis de errores"),
          paragraph("Las notas bajas muestran que compartir un tema o palabra no basta para responder toda la consulta. Estos son candidatos observados en la evaluación; la calificación se refiere a la consulta, no a la calidad general del libro."),
          Table([["Consulta abreviada", "Candidato", "Nota", "Posible causa"],
                 ["Imperio galáctico", "Frankenstein", "0", "Coincide 'científico'; falta el imperio."],
                 ["Control de emociones", "Orgullo y prejuicio", "0", "Normas sociales sin distopía."],
                 ["Romance fingido", "Cómo ganar amigos", "0", "Relaciones sin trama romántica."]],
                colWidths=[3.4 * cm, 3.5 * cm, 1.0 * cm, 7.5 * cm],
                style=TableStyle([("BACKGROUND", (0, 0), (-1, 0), GREEN),
                                  ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                                  ("FONTNAME", (0, 0), (-1, -1), "ArialBookery"),
                                  ("FONTNAME", (0, 0), (-1, 0), "ArialBookeryBold"),
                                  ("FONTSIZE", (0, 0), (-1, -1), 8),
                                  ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                  ("BACKGROUND", (0, 2), (-1, 2), LIGHT),
                                  ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                                  ("TOPPADDING", (0, 0), (-1, -1), 9),
                                  ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.HexColor("#d8dfd6"))])),
          Spacer(1, 8),
          paragraph("Una consulta adicional sobre vigilancia digital y pérdida de libertad devolvió también títulos poco relacionados, como <i>American Psycho</i> y <i>Coraline</i>. Este caso se observó en la demostración y no se incluyó en las nueve consultas usadas para las cifras de la tabla. Las fichas temáticas breves reducen la información disponible para distinguir coincidencias superficiales."),
          heading("8. Pruebas del sistema"),
          paragraph("Se comprobó en el navegador una consulta válida: 'Quiero ciencia ficción sobre un imperio galáctico', con cinco resultados y <i>Fundación</i> en primer lugar. La entrada vacía mostró el mensaje 'Escribe qué te gustaría leer'. La consulta difícil anterior devolvió resultados, pero evidenció límites de relevancia. La pantalla de evaluación mostró 9 de 9 consultas guardadas y recalculó la misma tabla registrada en CSV."),
          paragraph("El proyecto incluye pruebas Python para la API, métricas, recuperación y persistencia de notas, y una prueba de navegador que recorre Nuxt → FastAPI → recuperador. La instalación y los comandos están descritos en el README."),
          heading("9. Limitaciones y conclusiones"),
          paragraph("La muestra de nueve consultas es pequeña y fue diseñada alrededor del catálogo conocido; las conclusiones no se extrapolan a otras librerías ni a consultas arbitrarias. Parte de las descripciones carece de trama verificada. El LLM opcional puede producir explicaciones incorrectas y el modo Semántico + Jev no forma parte de esta evaluación."),
          paragraph("En estas consultas, la recuperación semántica cumplió el objetivo principal de mejorar el orden frente a BM25. La diferencia de Precision@5 fue pequeña y favorable a BM25. El sistema funcional permite consultar, comparar y revisar resultados, pero requiere ampliar consultas y fichas para una validación más sólida."),
          heading("Referencias y reproducibilidad"),
          paragraph("Repositorio: github.com/charlydeveloping/Bookery-AI. Datos: Open Library Search API (openlibrary.org/dev/docs/api/search) y anotaciones del proyecto. Modelo: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2, revisión indicada arriba. Código, dependencias, instrucciones y juicios humanos: README.md, data/evaluation/reviewed_queries.json y results/evaluation_reviewed.csv del repositorio.", "SmallBookery")]

doc.build(story, onFirstPage=page, onLaterPages=page)
print(OUTPUT)
