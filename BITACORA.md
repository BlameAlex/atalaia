# Bitácora — Entrega 3: Motor de IA de riesgo de inundación (AtalAIa · SOS)

Registro append-only de errores, descubrimientos, decisiones y resultados conforme suceden.
Formato: `### [AAAA-MM-DD HH:MM] [Agente/Rol] [ERROR | DESCUBRIMIENTO | DECISIÓN | RESULTADO]`

---

### [2026-11-11 22:50] [Líder] [DESCUBRIMIENTO]
El dataset entregado (`Riesgo de inundaciones para seguros en USA/`) es NFIP + NOAA Storm Events + FEMA a nivel condado-año: 83,804 filas, 3,511 condados, 2000–2026. No contiene llamadas 911, transcripciones ni la etiqueta `prioridad_target` prometida en las Entregas 1 y 2 (500 registros simulados).

### [2026-11-11 22:55] [Líder] [DECISIÓN]
Cambio de enfoque aprobado por el equipo: se usa solo el dataset USA. El flujo de llamadas 911 se reemplaza por un **botón SOS** que envía la ubicación del usuario (→ `county_fips`) al motor de IA, que estima su **nivel de riesgo de inundación (Bajo / Medio / Alto)**. Roles según EDT: Datos = Alexander Guillen; Modelado = Jesús Adrián Cárdenas; Pipeline = Andrés Ramos; Kevin Cano (PM) y Regina Zamacona en apoyo y documentación.

### [2026-11-11 22:56] [Líder] [DESCUBRIMIENTO]
`composite_risk_score` (rango 25.28–99.78) es una función determinista de cinco percentiles calculados con columnas del **mismo año**. Predecirlo con features del mismo año sería fuga de información (accuracy ≈100 % sin valor real). Decisión: plantear el problema como **pronóstico**: features hasta el año t → nivel de riesgo en t+1. El piso 25.28 corresponde a un condado-año sin actividad.

### [2026-11-11 22:57] [Líder] [DESCUBRIMIENTO]
Calidad de datos: (1) `state` tiene 112 valores distintos, mezclando abreviaturas y nombres completos (`AL` vs `ALABAMA`) → normalizar a abreviatura. (2) 41 códigos `xx000` son agregados estatales, no condados → eliminar. (3) 100 condados con un solo año → sin historia para lags, eliminar. (4) Panel incompleto: no todos los condados tienen los 27 años. (5) 2026 es año parcial (2,567 filas vs ~3,100). (6) `paid_loss_to_claim_coverage` 60.5 % nulo, coincide con filas sin reclamos → imputar 0. (7) 60.5 % de filas con 0 reclamos NFIP; montos muy sesgados → `log1p`.

### [2026-11-11 22:58] [Líder] [DECISIÓN]
Partición temporal en lugar de la aleatoria 80/20 de la Entrega 2: una partición aleatoria mezcla años futuros en train y filtra información. Train = años target 2001–2020 (~80 %), Test = 2021–2025 (~20 %). 2026 excluido como target (año parcial) pero usado como base del feature store de inferencia. Se mantiene `random_state = 42`.

### [2026-11-11 23:00] [Líder] [ERROR]
El Python del sistema (3.14.8) no tenía scikit-learn ni pandas (`ModuleNotFoundError: No module named 'sklearn'`). Acción: entorno virtual con `uv` e instalación desde `requirements.txt` (versiones fijadas: pandas 3.1.1, numpy 2.5.4, scikit-learn 1.9.1, joblib 1.6.0, matplotlib 3.11.2).

### [2026-11-11 23:02] [Líder] [RESULTADO]
Fase 0 completa: estructura `data/ src/ models/ reports/`, dataset copiado a `data/raw/county_year_climate_flood_risk.csv` (SHA-256 `1fdce7b1…51c0`, coincide con `source_manifest.json`), `.venv` operativo, plan en `docs/PLAN_ENTREGA3.md`.

### [2026-11-11 23:06] [Datos · Alexander Guillen] [ERROR]
Primera ejecución de `src/data/preprocess.py` falló en el assert de NaN (`RuntimeWarning: invalid value encountered in log1p`). Causa: 1 fila (condado 31011) con `nfip_paid_loss_usd` = -12,226.26 USD (ajuste contable NFIP), que arrastra `average_claim_severity_usd` y `paid_loss_to_claim_coverage` negativos. Acción: recortar montos y ratio a 0 (`clip(lower=0)`) en la limpieza, antes de `log1p`.

### [2026-11-11 23:06] [Datos · Alexander Guillen] [DESCUBRIMIENTO]
Normalización de `state`: los 112 valores quedan en 57 abreviaturas de 2 letras (50 estados + DC, PR, VI, GU, AS, MP, FM); ningún valor sin mapear. Se eliminan 225 filas de códigos `xx000` (83,804 → 83,579) y 115 condados con < 3 años observados reales (el brief pide < 3; el perfilado previo contaba 100 con 1 solo año). Mínimos observados de percentiles usados para años faltantes: claim_loss 0.0012, storm_damage 17.22, event_frequency 1.15, human_impact 44.57, declaration 31.34; piso del score = 25.2842.

### [2026-11-11 23:06] [Datos · Alexander Guillen] [RESULTADO]
Panel reindexado (primer año de cada condado → 2026): 89,361 filas, de ellas 5,920 imputadas (6.6 %). Dataset `data/processed/dataset_riesgo.csv`: 3,355 condados, 42 features, train = 65,878 filas (target 2001–2020), test = 16,773 (target 2021–2025). Umbrales sobre train: p33 = 39.3583, p67 = 56.2786. Balance train Bajo/Medio/Alto = 21,959 / 21,959 / 21,960; test = 7,043 / 5,375 / 4,355 (test con más Bajo: deriva temporal a la baja del score, relevante para Modelado). Feature store: 3,355 condados, todos con año base 2026. Asserts OK: sin NaN, splits correctos, sin target 2026, prueba de no-fuga (15 recálculos truncando a años ≤ t idénticos).

### [2026-11-11 23:14] [Modelado · Jesús Adrián Cárdenas] [DECISIÓN]
En Exp 3 el paralelismo se puso a nivel de GridSearchCV (`n_jobs=-1`) con cada RandomForest en `n_jobs=1` para no sobresuscribir los 16 núcleos; el modelo final se reconfigura a `n_jobs=-1` tras el refit. El grid completo (36 combinaciones × 3 folds de TimeSeriesSplit sobre train ordenado por `target_year`) tardó 312 s, dentro del límite de ~10 min, así que no se redujo. Test nunca se usó para elegir hiperparámetros.

### [2026-11-11 23:14] [Modelado · Jesús Adrián Cárdenas] [RESULTADO]
`src/models/train.py` ejecutado; mismo test (target 2021–2025, 16,773 filas) para los 4 experimentos.

| Exp | Modelo | F1 macro test | Acc test | Recall Alto test | F1 macro train | Diagnóstico |
|---|---|---|---|---|---|---|
| 0 | Persistencia | 0.5237 | 0.5325 | 0.5451 | 0.5159 | Underfitting |
| 1 | LogReg + StandardScaler | 0.5652 | 0.5874 | 0.5444 | 0.5711 | Underfitting |
| 2 | RF por defecto | 0.5627 | 0.5647 | 0.6285 | 0.9961 | Overfitting |
| 3 | RF ajustado | **0.5751** | 0.5800 | **0.6489** | 0.5912 | Modelo Ganador |

Ganador: Exp 3, best_params = {max_depth: 8, min_samples_leaf: 5, max_features: "sqrt", class_weight: None} (F1 macro CV = 0.5688). Serializado en `models/modelo_candidato.joblib`; `requiere_scaler = false`. Por la regla automática (F1 train < 0.6) el Exp 3 también calificaría como "Underfitting".

### [2026-11-11 23:14] [Modelado · Jesús Adrián Cárdenas] [DESCUBRIMIENTO]
El RF por defecto memoriza train (F1 0.996) pero en test rinde igual que la LR (0.563 vs 0.565): brecha de 0.43 = overfitting claro. El ajuste eligió el árbol más superficial del grid (max_depth=8, el extremo inferior), lo que cierra la brecha a 0.016 (train 0.591 / test 0.575) y sube el F1 test +0.012 y el recall Alto +0.020. Que el óptimo quede en el borde del grid sugiere que profundidades menores podrían rendir igual o mejor; no se exploró para no ampliar el alcance. La mejora sobre persistencia es real pero modesta (+0.05 F1).

### [2026-11-11 23:14] [Modelado · Jesús Adrián Cárdenas] [DESCUBRIMIENTO]
Comparación honesta contra las metas de la Entrega 2 (F1 macro ≥ 0.82, Accuracy ≥ 85 %): **no se alcanzan** (mejor modelo: F1 0.575, Acc 58.0 %). Las metas de E2 suponían otro problema (clasificación de llamadas 911 con datos simulados); aquí se pronostica el nivel del año siguiente a partir de eventos climáticos con alto ruido interanual. La clase Medio es la más difícil (F1 ≈ 0.44): la mayoría de los errores son entre clases adyacentes; las confusiones Bajo↔Alto son pocas (560 y 343 de 16,773). No se forzó el resultado con features del año t+1.

### [2026-11-11 23:15] [Pipeline · Andrés Ramos] [DECISIÓN]
`predecir()` fija `n_jobs = 1` en el RandomForest al cargarlo (Modelado lo serializó con `n_jobs = -1`). Con una sola fila por solicitud SOS, el paralelismo de hilos solo agregaba overhead: latencia media 58.99 ms → 9.73 ms, p95 73.11 ms → 10.20 ms (100 llamadas, seed 42). Las predicciones no cambian.

### [2026-11-11 23:15] [Pipeline · Andrés Ramos] [DESCUBRIMIENTO]
El condado de menor `composite_risk_score_ma3` es `99117` con `state = PR` en el CSV crudo. Ese no es un FIPS estándar de Puerto Rico (los de PR son 72xxx); viene así en el dataset original. Se deja tal cual y se avisa a Datos: no afecta la inferencia, que solo busca por clave.

### [2026-11-11 23:15] [Pipeline · Andrés Ramos] [RESULTADO]
`src/models/predict.py` funciona con el ganador "Exp3 RandomForest ajustado" (sin scaler). Ejemplo: entrada `{"id_solicitud": "SOS-0001", "county_fips": "48439"}` → TX, año base 2026, pronóstico 2027, `nivel_riesgo = Alto`, probabilidades Bajo 0.0082 / Medio 0.0302 / Alto 0.9616, confianza 96.16, `requiere_revision = false`. Factores clave: puntaje compuesto con media histórica 88.91, media 5 años 88.73 y media 3 años 88.57 (todas sobre p75). El condado bajo (`99117`, PR) sale Bajo con confianza 93.02; `99999` devuelve error "no encontrado". Asserts de forma OK. Latencia en 100 llamadas: media 9.73 ms, p95 10.20 ms (< 100 ms). Salida del demo en `reports/demo_inferencia.txt`.

### [2026-11-11 23:20] [Documentación · Regina Zamacona] [DESCUBRIMIENTO]
Inconsistencia de latencia: `BITACORA.md` (entrada de Pipeline 23:15) registra media 9.73 ms / p95 10.20 ms, pero el `reports/demo_inferencia.txt` vigente dice media 12.07 ms / p95 22.02 ms (proviene de una ejecución posterior del demo). Ambas cumplen p95 < 100 ms. En el documento se reporta la del archivo vigente y se menciona la diferencia. Además, `src/data/preprocess.py` acepta un argumento `--models` (por defecto `models`) que no aparece en el comando de una línea del plan; se documenta. El Exp 3 (ganador) también cae en "Underfitting" según la regla automática de `train.py` (F1 train < 0.6); se reporta tal cual. El tiempo de 312.48 s del Exp 3 es la búsqueda completa (36 × 3 ajustes + refit de 13.17 s), no un solo entrenamiento.

### [2026-11-11 23:20] [Documentación · Regina Zamacona] [RESULTADO]
`ENTREGA3_Experimentacion.md` generado en la raíz, siguiendo la plantilla de `INSTRUCCIONES.md` (sin modificarla) sección por sección: encabezado con roles y nota de cambio de alcance; 1) Estrategia Comparativa RF vs. LR + ajuste fino + persistencia; 2) rutas reales y comando de una línea; 3) matriz Exp 0–3 con análisis crítico y comparación honesta contra metas de E2 (no alcanzadas: F1 0.5751 vs. 0.82, Acc 58.00 % vs. 85 %); 4) `train_log.txt` completo, matriz de confusión del ganador (imagen + tabla) e importancias; 5) `predict.py` copiado tal cual y entrada/salida/error reales del demo; 6) avance 90 %, semáforo "Con retraso moderado" (entrega 11 nov, límite 9 nov), blockers y plan de Entrega 4. Los bloques de log, código y JSON se insertaron por script desde los archivos; una verificación automática comparó cada cifra de las secciones 3–5 contra `reports/metrics.json`, `models/metadata.json`, `reports/train_log.txt` y `reports/demo_inferencia.txt` sin diferencias (se corrigió un +0.0099 calculado con valores redondeados a +0.0100).

### [2026-11-11 23:27] [Líder] [DECISIÓN]
`confianza` en la salida de `predecir()` se expresa en escala 0–100 (umbral de revisión manual = 70), igual que el ejemplo de la plantilla oficial (`"confianza": 82.45`), en lugar de la fracción 0–1 escrita en el borrador del plan. `probabilidades` sí se mantiene en 0–1.

### [2026-11-11 23:27] [Revisor final] [RESULTADO]
Revisión de contexto fresco sobre `src/` y artefactos: 0 hallazgos Critical/Important. Verificado sin fuga temporal (shift/rolling/expanding dentro de cada condado; umbrales, p75, scaler y GridSearch solo con train), target alineado al mismo condado, comando de una línea reproducible (metadata idéntica byte a byte en clon limpio), `predecir()` robusto ante None, int, str con espacios, FIPS de 4 dígitos, dict vacío, NaN y listas, y métricas recalculadas coinciden (F1 macro 0.57511, recall Alto 0.64891). 3 hallazgos Minor diferidos: (1) `tiempo_entrenamiento_s` del Exp 3 incluye el grid completo (312 s, 108 ajustes); el ajuste final está en `tiempo_refit_s`; (2) escala de `confianza` (registrada arriba); (3) `str.isdigit()` acepta dígitos Unicode como "²" (hoy solo produce un error controlado).

### [2026-11-11 23:28] [Documentación · Regina Zamacona] [DECISIÓN]
Por indicación del líder, en la matriz de la sección 3 el tiempo de entrenamiento del Exp 3 se reporta como el refit del modelo final (`tiempo_refit_s` = 13.17 s en `reports/metrics.json`). Aparte se aclara que la búsqueda `GridSearchCV` completa tardó 312.48 s (36 combinaciones × 3 folds = 108 ajustes, más el refit); el mismo aviso se añadió junto al log de la sección 4, porque `train_log.txt` rotula los 312.48 s como "Entrenamiento finalizado". Las secciones 5 y 6 explicitan que `confianza` va en escala 0–100 (las probabilidades, en 0–1) y que `requiere_revision` se activa con confianza < 70. Se volvió a ejecutar la verificación automática de cifras: sin diferencias.
