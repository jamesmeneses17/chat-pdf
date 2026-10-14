# Chat con PDFs (RAG local)

Sube PDFs y chatea con ellos. Las respuestas salen solo del contenido de los
documentos y se muestra la fuente (archivo y página). Todo corre en tu
computador, sin APIs externas ni servicios de pago.

Funciona en **modo prueba (mock)** sin Ollama; luego se conecta a Ollama
cambiando una sola variable del `.env`.

## 1. Requisitos (Windows)

- Python 3.11 o superior
- Docker Desktop (abierto y corriendo)

## 2. Levantar la base de datos

```powershell
docker compose up -d
```

Crea PostgreSQL con pgvector en el puerto 5432 (base `rag`, clave `1234`).
Los datos quedan guardados en un volumen de Docker.

## 3. Instalar dependencias y ejecutar

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

En una terminal (API):

```powershell
uvicorn app.api:app --port 8000
```

En otra terminal (interfaz; activa también el entorno virtual):

```powershell
streamlit run ui.py
```

Abre http://localhost:8501. La API crea las tablas sola al iniciar.

### Qué hace la aplicación
- **Visor y resumen:** al subir un PDF se abre a la izquierda (con scroll y las hojas escaneadas
  de lado ya enderezadas) y su resumen se escribe en el chat, con 3 preguntas sugeridas.
- **Puntos clave y glosario:** botones bajo el visor; el resultado aparece en el chat.
- **Citas y resaltado:** cada respuesta tiene un botón «📍 Ver en el PDF». El visor se abre en la
  página de donde salió la respuesta y **pinta de amarillo el pasaje exacto** (también en
  documentos escaneados, usando OCR, y en las hojas giradas). Además se pueden ver los
  fragmentos consultados.
- **Detalle de los fragmentos:** cada respuesta muestra cuántos fragmentos se consultaron (de cuántos hay),
  y de cada uno su texto completo, su posición («fragmento 12 de 42»), el parecido por significado,
  las palabras de la pregunta que contiene, el puntaje final y por qué búsqueda se halló.
- **Resumen con un fragmento por página:** se toma el fragmento más largo de cada página y se
  resumen juntos (así el resumen cubre todas las páginas). Debajo se indica cuántos fragmentos y
  páginas se usaron y qué porcentaje del texto es; si el resumen menciona cifras que no están en
  el texto leído, lo avisa.
- **Imágenes del PDF:** «¿cuántas imágenes trae el PDF?» se contesta leyendo el archivo (sin IA),
  con las páginas que las tienen y las páginas que son un escaneo completo. Qué muestra una imagen
  (un perro, una planta...) **todavía no se puede saber**: haría falta un modelo de visión.
- **Varias claves de Gemini:** puedes poner hasta 3 (`GEMINI_API_KEY`, `GEMINI_API_KEY_2`,
  `GEMINI_API_KEY_3`). Se usan en orden; si una llega a su límite de uso (queda en pausa ~1 min, o ~1 h
  si se repite) o es rechazada, el sistema pasa solo a la siguiente y **te avisa** en la respuesta.
  En la barra lateral («☁️ Claves de Gemini») ves el estado de cada una y la enciendes o apagas; eso
  queda guardado. Las claves nunca se muestran. Tras editar el `.env`, reinicia `uvicorn`.
- **Buscar dentro de las imágenes con Gemini (nube, opcional):** con una clave en `GEMINI_API_KEY`
  del `.env`, al subir un PDF sus imágenes se envían a Google (`gemini-embedding-2`, que pone texto e
  imágenes en un mismo espacio de vectores) y quedan guardadas. Después se pregunta «¿hay una imagen
  de un perro?» y se muestran **solo las que coinciden**: las que tienen un parecido de al menos
  `IMAGENES_MIN_SIM` (0,585) y están a menos de `IMAGENES_MARGEN` (0,06) de la mejor. Así, si se
  pregunta por el gato, sale el gato y no el perro ni la lombriz; si se pregunta por «un animal»,
  salen todos los animales; y si no hay nada parecido («una persona»), lo dice y muestra cuánto
  se acercó la más cercana. Esos valores se calibraron con puntajes reales de Gemini sobre 6
  ilustraciones y 18 frases: 12 de 12 pruebas exactas, frente a un 43 % de acierto al mostrar
  siempre las 3 primeras. Con fotos reales pueden necesitar ajuste. El parecido sigue siendo una
  estimación: confírmalo mirando la imagen. Las páginas que son un
  escaneo completo **no** se envían por defecto (`IMAGENES_INCLUIR_ESCANEADAS=0`). En el plan
  gratuito Google puede usar el contenido enviado para mejorar sus productos. Sin clave, esta
  función se desactiva y todo lo demás sigue local.
- **Chat con imágenes:** se puede adjuntar una imagen PNG/JPG junto a la pregunta; se lee su
  **texto** con OCR y se usa para buscar en los documentos. No describe lo que la imagen muestra.
- **Varios PDFs:** se busca en todos tus documentos o solo en los que marques en la barra lateral.
- **Memoria:** entiende seguimientos como «¿y eso qué significa?» usando los últimos mensajes.
- **Exportar:** el resumen (con puntos clave y glosario) y la conversación se descargan en PDF o Word.
- **Usuarios e historial:** escribe un nombre de usuario en la barra lateral. Cada usuario ve solo
  sus documentos y sus conversaciones guardadas. **No usa contraseña**: separa la información por
  nombre, pensado para uso local.
- **Búsqueda híbrida:** combina vectores (pgvector) con búsqueda de texto completo de PostgreSQL y
  reordena los candidatos con un puntaje final. Se ajusta con `PESO_PALABRAS`, `PESO_FTS`,
  `PESO_RRF` y `BUSQUEDA_CANDIDATOS` en el `.env`.

Las bases creadas con versiones anteriores se migran solas al iniciar la API: lo que ya tenías
queda asignado al usuario `invitado`.

## 4. Conectar Ollama

1. Instala Ollama desde https://ollama.com.
2. Descarga los modelos:
   ```powershell
   ollama pull nomic-embed-text
   ollama pull llama3.2:3b
   ```
   (PC lenta: usa `llama3.2:1b` y cambia `CHAT_MODEL=llama3.2:1b` en el `.env`.)
3. En el `.env` cambia `AI_MODE=ollama` y reinicia `uvicorn` y `streamlit`.
4. Si ya habías subido PDFs en modo mock, **bórralos y vuelve a subirlos**:
   los vectores de prueba no sirven con el modelo real.

No hay que tocar ningún otro archivo: solo `embeddings.py` y `llm.py` leen
`AI_MODE`.

## 5. Cómo funciona el RAG (para la exposición)

RAG = *Retrieval-Augmented Generation*: el modelo responde apoyándose en
documentos tuyos en lugar de su memoria.

**Al subir un PDF (ingesta)**
1. PyMuPDF extrae el texto de cada página.
2. El texto se corta en fragmentos de ~800 caracteres con solape de 150
   (para no perder ideas en el borde), sin partir palabras.
3. Cada fragmento se convierte en un *embedding*: un vector de 768 números que
   representa su significado. Textos parecidos dan vectores cercanos.
4. Se guarda fragmento + página + documento + vector en PostgreSQL (pgvector).

**Al preguntar (consulta)**
1. La pregunta también se convierte en un vector.
2. pgvector busca los 4 fragmentos más cercanos (distancia coseno, operador `<=>`).
3. Se arma un prompt: "responde SOLO con este contexto" + fragmentos + pregunta.
4. El LLM redacta la respuesta y la interfaz muestra las fuentes.

**Modo prueba:** el embedding se calcula con el *hashing trick* (cada palabra
suma 1 en una posición del vector según su hash) y la "respuesta" solo muestra
los fragmentos encontrados. Así se ve que la búsqueda funciona sin IA real.

## Estructura

```
app/config.py       variables del .env
app/db.py           conexión, tablas, listar/borrar documentos
app/pdf_loader.py   texto por página
app/chunker.py      división en fragmentos
app/embeddings.py   obtener_embedding(): mock u ollama
app/llm.py          generar_respuesta(): mock u ollama
app/ingest.py       PDF -> fragmentos -> vectores -> BD
app/search.py       búsqueda por similitud
app/api.py          endpoints FastAPI (/subir /documentos /chat /estado)
ui.py               interfaz Streamlit
```

## Notas

- **PDFs escaneados (solo imágenes):** se leen con OCR (Tesseract), solo en las páginas
  que no tienen texto. Requiere instalar Tesseract con el idioma español
  (https://github.com/UB-Mannheim/tesseract/wiki) y revisar en el `.env`:
  `OCR_LANG=spa` y `TESSDATA_DIR` (por defecto `C:\Program Files\Tesseract-OCR\tessdata`).
  Las hojas escaneadas de lado o al revés se detectan solas: si el texto sale ilegible,
  el OCR gira la página y se queda con la rotación más legible. Si alguna página sigue
  poco legible, la interfaz lo avisa al subir el PDF.
  El OCR es más lento (~2,5 s por página) y puede fallar con imágenes borrosas, casillas
  marcadas o letra a mano.
- Si cambias `EMBED_DIM`, hay que recrear la tabla:
  `docker exec chatpdf_db psql -U postgres -d rag -c "DROP TABLE chunks"`.
