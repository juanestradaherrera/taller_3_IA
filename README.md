#  NLP & LLMs Avanzado — EAFIT 2026-1

**Parcial — NLP y LLMs Avanzado: Sintonización de Parámetros, Métricas de Similitud y Evaluación de Modelos de Lenguaje**

| Campo | Detalle |
|-------|---------|
| **Curso** | Inteligencia Artificial ECA&I — Posgrado |
| **Docente** | Jorge Iván Padilla-Buriticá |
| **Universidad** | EAFIT — Período 2026-1 |
| **Integrantes** | Ana Patricia Montes Pimienta · Karen Melissa Gómez Montoya · Juan Esteban Estrada Herrera |
| **API utilizada** | Groq (`llama-3.3-70b-versatile`) |


---

##  Estructura del Proyecto

```
nlp_llms_parcial/
├── app.py                    # Aplicación principal Streamlit (4 pestañas)
├── requirements.txt          # Dependencias del proyecto
├── .gitignore                # Excluye secrets.toml y archivos sensibles
├── .streamlit/
│   └── secrets.toml          
└── README.md
```

---

##  Instalación y Ejecución

```bash
# 1. Clonar el repositorio
git clone <url-del-repo>
cd nlp_llms_parcial

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Configurar API Key (NUNCA escribirla en código plano)
mkdir -p .streamlit
cat > .streamlit/secrets.toml << 'TOML'
[general]
GROQ_API_KEY = "gsk_TU_CLAVE_AQUI"
TOML

# 4. Ejecutar la aplicación
streamlit run app.py
```

---

##  Descripción de las Partes

###  Parte 01 — Evaluación Conceptual (Quiz Teórico) 

Respuestas técnicas a 5 preguntas conceptuales sobre NLP y LLMs:

| # | Tema | Concepto clave |
|---|------|----------------|
| 1 | Métricas de similitud | Similitud coseno vs. distancia euclidiana en embeddings |
| 2 | Distribución de probabilidad | Temperatura, Top-p y Top-k sobre la distribución Softmax |
| 3 | Métricas de evaluación | BLEU, ROUGE-L y BERTScore — diferencias semánticas |
| 4 | Adaptación de modelos | Fine-tuning supervisado (SFT) vs. LoRA — parámetros entrenables |
| 5 | Evaluación automática | LLM-as-a-Judge — ventajas, position bias y verbosity bias |

---

###  Parte 02 — Laboratorio de Parámetros 

Pestaña interactiva para experimentar con los hiperparámetros de generación de texto.

#### Panel de Control (6 parámetros)

| Parámetro | Rango | Efecto observado |
|-----------|-------|-----------------|
| **Temperatura** | 0.0 – 2.0 | Controla creatividad vs. determinismo. Valores bajos → respuestas precisas y repetibles; valores altos → mayor variedad y riesgo de incoherencia |
| **Top-p** (nucleus) | 0.0 – 1.0 | Limita el vocabulario efectivo acumulando masa de probabilidad. Top-p=0.3 restringe a tokens muy probables; Top-p=0.9 permite mayor diversidad |
| **Top-k** | 1 – 100 | Número máximo de tokens candidatos en cada paso de generación |
| **Max Tokens** | 50 – 2048 | Longitud máxima de la respuesta generada |
| **Frequency Penalty** | 0.0 – 2.0 | Penaliza tokens que ya aparecieron frecuentemente → reduce repetición |
| **Presence Penalty** | 0.0 – 2.0 | Penaliza cualquier token que ya haya aparecido → fomenta temas nuevos |

#### Experimento Comparativo (4 configuraciones contrastantes)

Prompt fijo: *"Explica el concepto de atención en transformers"*

| Config | Temperatura | Top-p | Efecto esperado |
|--------|-------------|-------|-----------------|
| A | 0.1 | 0.9 | Respuesta determinista con vocabulario amplio |
| B | 1.5 | 0.9 | Respuesta creativa con vocabulario amplio |
| C | 0.1 | 0.3 | Respuesta determinista con vocabulario restringido |
| D | 1.5 | 0.3 | Respuesta creativa con vocabulario muy restringido |

**Métricas calculadas por configuración:**
- Longitud en tokens
- Diversidad léxica — Type-Token Ratio (TTR = tokens únicos / total tokens)

**Visualización:** Gráficas Plotly en columnas paralelas (`st.columns(4)`) + campo de observaciones documentadas.

#### 🔬 Análisis de Parámetros

> **Temperatura baja (0.1):** El modelo converge hacia las predicciones de mayor probabilidad, generando respuestas coherentes, precisas y con baja variabilidad entre ejecuciones. Ideal para tareas que requieren exactitud factual.

> **Temperatura alta (1.5):** La distribución Softmax se aplana, redistribuyendo masa de probabilidad hacia tokens menos probables. El resultado es texto más variado y creativo, pero con mayor riesgo de inconsistencias o alucinaciones.

> **Top-p bajo (0.3):** El modelo solo considera los tokens que en conjunto acumulan el 30% de la probabilidad. El vocabulario efectivo se reduce significativamente, lo que aumenta la coherencia local pero puede reducir la riqueza expresiva.

> **Top-p alto (0.9):** Se incluye una gama mucho más amplia de tokens candidatos, favoreciendo la diversidad léxica (TTR más alto) sin sacrificar demasiado la coherencia.

> **Interacción Temperatura + Top-p:** La combinación Temp=1.5 / Top-p=0.3 produjo las respuestas más impredecibles: alta temperatura amplifica la estocasticidad, pero Top-p bajo la contiene parcialmente, resultando en texto inesperado pero acotado en vocabulario.

---

###  Parte 03 — Métricas de Similitud 

Pestaña para comparar cuantitativamente un texto de referencia (*ground truth*) contra la respuesta generada por el LLM.

#### Métricas Implementadas

| Métrica | Librería | Qué mide | Rango |
|---------|----------|----------|-------|
| **Similitud Coseno** | `sentence-transformers` (all-MiniLM-L6-v2) | Cercanía semántica en espacio de embeddings. Invariante a la magnitud del vector — mide el ángulo entre representaciones. | 0 – 1 |
| **BLEU Score** | `nltk.translate.bleu_score` | Solapamiento de n-gramas entre referencia y generación. Penaliza respuestas más cortas que la referencia (brevity penalty). | 0 – 1 |
| **ROUGE-L** | `rouge-score` | Subsecuencia común más larga (LCS) entre referencia y candidato. Captura coherencia estructural y orden de palabras. | 0 – 1 |
| **BERTScore** | `bert-score` | Similitud semántica token a token usando representaciones BERT. Captura paráfrasis y sinónimos que BLEU ignora. | 0 – 1 |
| **LLM-as-Judge** | Segunda llamada a la API (Groq) | Evaluación cualitativa estructurada en JSON: score (1-10), veracidad, coherencia, relevancia, fortalezas y debilidades. | 1 – 10 |

#### ¿Por qué BERTScore supera a BLEU en semántica?

BLEU evalúa coincidencias exactas de n-gramas: si la respuesta usa *"transformación lineal"* en lugar de *"proyección matricial"*, BLEU lo penaliza aunque el significado sea equivalente. BERTScore compara representaciones contextuales (embeddings BERT) de cada token, capturando similitud semántica independientemente del vocabulario exacto empleado.

#### Flujo de la pestaña

```
Usuario ingresa texto de referencia (ground truth)
          ↓
Usuario ingresa prompt → LLM genera respuesta candidata
          ↓
Cálculo de 4 métricas automáticas → st.metric / tabla
          ↓
Segunda llamada a la API (LLM-as-Judge) → JSON estructurado
          ↓
Radar chart Plotly con métricas normalizadas
```

#### Visualización — Radar Chart

El radar chart muestra las 5 métricas normalizadas al rango [0,1] para comparación visual inmediata entre similitud de superficie (BLEU/ROUGE) y similitud semántica (Coseno/BERTScore/LLM-Judge).

---

###  Parte 04 — Agente Conversacional: TutorML 

Agente conversacional especializado en **Machine Learning**, con personalidad definida, memoria de conversación y métricas de producción en tiempo real.

#### Descripción del Agente

> **Nombre:** TutorML
> **Dominio:** Machine Learning — teoría, algoritmos, métricas, buenas prácticas y casos de uso
> **Personalidad:** Tutor experto, didáctico y preciso. Responde con ejemplos concretos, adapta el nivel técnico al contexto de la pregunta y señala explícitamente cuando un concepto está fuera de su dominio.
> **Restricciones:** No responde preguntas fuera del ámbito de ML/Data Science. Redirige amablemente hacia temas de su especialidad.

**System Prompt base:**
```
Eres TutorML, un asistente experto en Machine Learning diseñado para la 
Maestría en Ciencia de Datos de EAFIT. Tu misión es explicar conceptos de ML 
con precisión técnica y ejemplos claros. Adapta la profundidad de tus respuestas 
al nivel de la pregunta. Si la pregunta está fuera del ámbito de ML o Data Science, 
indícalo amablemente y redirige la conversación.
```

#### Funcionalidades

| Funcionalidad | Implementación |
|---------------|----------------|
| Memoria de conversación | `st.session_state` — historial completo enviado en cada turno |
| Renderizado del chat | `st.chat_message` con roles user/assistant |
| Limpiar conversación | Botón que reinicia `st.session_state` |
| Controles en sidebar | Sliders de temperatura y max_tokens en tiempo real |
| Historial de métricas | Gráfica de línea Plotly por turno |

####  Métricas de Producción (tiempo real)

| Métrica | Cálculo | Descripción |
|---------|---------|-------------|
| **Latencia (s)** | `time.time()` — wall-clock end-to-end | Tiempo total desde envío hasta último token recibido |
| **Tokens/segundo (TPS)** | `completion_tokens / latencia` | Velocidad de generación extraída de `response.usage` |
| **Tokens de entrada** | `response.usage.prompt_tokens` | Tokens del historial + prompt actual enviados a la API |
| **Tokens de salida** | `response.usage.completion_tokens` | Tokens generados en la respuesta |
| **Costo estimado (USD)** | Pricing público Groq | Calculado con tarifas por millón de tokens de entrada y salida |
| **Puntuación LLM-Judge** | Segunda llamada a la API | Auto-evaluación de la última respuesta en escala 1-10 |

---

##  Capturas de Pantalla

> Las capturas se agregarán tras ejecutar la aplicación. Se incluirán vistas de:
> - Panel de parámetros con las 4 configuraciones comparativas
> - Radar chart de métricas de similitud
> - Conversación con TutorML y panel de métricas en tiempo real
> - Gráfica de historial de métricas por turno

---

##  Seguridad

- La API Key **nunca** se escribe en código plano ni se sube al repositorio
- Se usa `.streamlit/secrets.toml` (local) o `st.secrets` (Streamlit Cloud)
- Alternativamente: `input(type="password")` como mecanismo en UI
- El archivo `secrets.toml` está incluido en `.gitignore`

```gitignore
.streamlit/secrets.toml
.env
__pycache__/
*.pyc
```

---

##  Librerías y Dependencias

| Categoría | Librería / Paquete | Uso |
|-----------|-------------------|-----|
| API LLM | `groq` | Cliente para `llama-3.3-70b-versatile` |
| UI | `streamlit` | Interfaz de usuario con pestañas |
| Embeddings | `sentence-transformers` | Modelo `all-MiniLM-L6-v2` para similitud coseno |
| Métricas NLP | `nltk` | BLEU Score |
| Métricas NLP | `rouge-score` | ROUGE-L |
| Métricas NLP | `bert-score` | BERTScore token a token |
| Visualización | `plotly` | Radar chart, barras, líneas de tiempo |
| Datos | `pandas` | Manejo de tablas de métricas |
| Utilidades | `time`, `json`, `re` | Latencia, parsing JSON, expresiones regulares |

---

##  Checklist de Entrega

- [x] `app.py` ejecutable con `streamlit run app.py` sin errores
- [x] 4 pestañas implementadas (Quiz, Laboratorio, Métricas, Agente)
- [x] API Key manejada con `st.secrets` — nunca en código plano
- [x] Visualizaciones interactivas con Plotly en cada sección
- [x] Funciones documentadas con docstrings
- [x] Experimento comparativo con 4 configuraciones en columnas paralelas
- [x] 5 métricas de evaluación (Coseno, BLEU, ROUGE-L, BERTScore, LLM-Judge)
- [x] Radar chart con métricas normalizadas
- [x] Agente TutorML con memoria (`st.session_state`)
- [x] 6 métricas de producción en tiempo real
- [x] Historial de métricas por turno (gráfica de línea)
- [x] `requirements.txt` completo
- [x] `.gitignore` con `secrets.toml` excluido

---
