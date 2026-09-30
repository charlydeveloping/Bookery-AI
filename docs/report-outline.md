# Guion del informe — Bookery AI

1. **Problema y objetivo:** pregunta experimental y alcance del chatbot.
2. **Dataset y preparación:** procedencia, licencia, variables, tamaño obtenido, limpieza y sesgos.
3. **Línea base:** fórmula BM25 y parámetros `k1=1.5`, `b=0.75`.
4. **Modelo propuesto:** modelo preentrenado, dimensión, coseno e índice exacto.
5. **Métricas:** nDCG@5, Precision@5, cumplimiento y latencia.
6. **Estrategia de validación:** consultas separadas, anotación humana graduada, mismos filtros y catálogo.
7. **Resultados:** insertar únicamente la tabla generada por `python -m ml.evaluate`.
8. **Arquitectura:** diagrama de preparación, recuperadores, API y Nuxt.
9. **Interfaz:** captura del formulario y tarjetas; comparación visual de métodos.
10. **Pruebas:** caso válido, consulta semántica difícil, entrada vacía y contrato HTTP.
11. **Análisis de errores:** consultas con vocabulario divergente, descripciones pobres, categorías ruidosas.
12. **Limitaciones:** cobertura y sesgo del catálogo, etiquetas humanas, idioma y latencia en CPU.
13. **Conclusiones:** responder la pregunta experimental solo cuando existan juicios y resultados reales.
14. **Referencias:** enlaces primarios del README y bibliografía de BM25 y Sentence Transformers.
