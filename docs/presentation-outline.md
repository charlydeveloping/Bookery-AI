# Exposición de Bookery AI (máximo cinco minutos)

1. **Problema y objetivo (45 s).** Una consulta conversacional expresa preferencias; comparar el ranking BM25 frente a embeddings sobre el mismo catálogo.
2. **Datos y métodos (60 s).** Origen, limpieza, búsqueda léxica y similitud coseno. Aclarar que el modelo es preentrenado.
3. **Resultados (65 s).** Usar `results/evaluation_reviewed.csv`: nDCG@5 de 0,8998 para semántica y 0,8456 para BM25; Precision@5 de 0,4222 y 0,4444, respectivamente. Son nueve consultas y 71 pares consulta-libro calificados manualmente.
4. **Demo (90 s).** Ejecutar consulta válida, consulta con palabras distintas de la descripción y consulta vacía. Alternar BM25/semántica.
5. **Conclusiones y límites (40 s).** Responder con evidencia, cobertura de datos y trabajo futuro.

No presentar las cifras piloto de `results/evaluation.csv` como evaluación final. Semántico + Jev es opcional y no forma parte de esta comparación.
