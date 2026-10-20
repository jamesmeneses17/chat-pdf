"""Interfaz Streamlit de «Base de Conocimiento IA»: documento (izquierda) y chat (derecha) al mismo tiempo."""
import hashlib
from contextlib import nullcontext
import json
import queue
import re
import threading
import time
import unicodedata
from urllib.parse import quote

import httpx
import pymupdf
import streamlit as st

from app import config

API = config.API_URL
ALTURA = 700  # alto (px) del visor de PDF y de la caja del chat
USUARIO_POR_DEFECTO = "invitado"

st.set_page_config(page_title="Base de Conocimiento IA", page_icon="🧠", layout="wide")


# ---------------------------------------------------------------- usuario
def usuario_actual() -> str:
    return (st.session_state.get("usuario_txt") or "").strip() or USUARIO_POR_DEFECTO


def cabeceras(usuario: str | None = None) -> dict:
    """Cabecera con el usuario, que la API usa para separar los datos de cada persona."""
    return {"X-Usuario": quote(usuario or usuario_actual())}


# Al cambiar de usuario (o la primera vez) se empieza de cero: cada uno ve lo suyo
if st.session_state.get("usuario_activo") != usuario_actual():
    for clave in list(st.session_state):
        if clave.startswith(("buscar_", "visibles_", "giro_", "sug_")):
            del st.session_state[clave]
    for clave in ("doc_actual", "pregunta_pendiente", "resaltar", "resaltar_pendiente"):
        st.session_state.pop(clave, None)
    st.session_state.update(mensajes=[], conv_id=None, pendientes=[], procesados=set(), miniaturas={},
                            usuario_activo=usuario_actual())

# Un resaltado pedido en la ejecución anterior se aplica ahora, antes de dibujar los widgets
if "resaltar_pendiente" in st.session_state:
    st.session_state.resaltar = st.session_state.pop("resaltar_pendiente")
    st.session_state.doc_actual = st.session_state.resaltar["documento"]


# --------------------------------------------------------- llamadas a la API
# `cab` se calcula en el hilo principal: los hilos de fondo no pueden leer st.session_state.
def llamar(metodo: str, ruta: str, cab: dict | None = None, **kwargs):
    """Llama a la API. Devuelve (datos, error) con mensajes amigables."""
    try:
        r = httpx.request(metodo, API + ruta, headers=cab or cabeceras(), timeout=300, **kwargs)
    except httpx.HTTPError:
        return None, "No se pudo conectar con la API. ¿Está corriendo uvicorn?"
    if r.status_code >= 400:
        try:
            return None, r.json().get("detail", "Ocurrió un error.")
        except ValueError:
            return None, "Ocurrió un error en el servidor."
    return r.json(), None


def obtener_pdf(nombre: str):
    """Descarga el PDF original desde la API. Devuelve (bytes, error)."""
    try:
        r = httpx.get(f"{API}/documentos/{quote(nombre)}/pdf", headers=cabeceras(), timeout=60)
    except httpx.HTTPError:
        return None, "No se pudo conectar con la API. ¿Está corriendo uvicorn?"
    if r.status_code >= 400:
        try:
            return None, r.json().get("detail", "No se pudo cargar el PDF.")
        except ValueError:
            return None, "No se pudo cargar el PDF."
    return r.content, None


def leer_stream(ruta: str, resultado: dict, cuerpo: dict | None = None, cab: dict | None = None):
    """Lee una respuesta de la API que llega por pedazos (una línea JSON cada uno).

    Va entregando el texto; lo demás (fuentes, tiempos, error...) queda en `resultado`.
    """
    try:
        with httpx.stream("POST", API + ruta, json=cuerpo, headers=cab or cabeceras(),
                          timeout=300) as r:
            if r.status_code >= 400:
                resultado["error"] = "Ocurrió un error en el servidor."
                return
            for linea in r.iter_lines():
                if not linea:
                    continue
                dato = json.loads(linea)
                if "texto" in dato:
                    yield dato["texto"]
                else:
                    resultado.update(dato)  # fuentes, tiempos, citas o error
    except httpx.HTTPError:
        resultado["error"] = "No se pudo conectar con la API. ¿Está corriendo uvicorn?"


def leer_stream_con_contador(ruta: str, resultado: dict, reloj, inicio: float):
    """Como leer_stream, pero actualiza un contador de segundos en `reloj` mientras espera.

    La lectura se hace en un hilo; este generador va sacando los pedazos de una cola y,
    si tardan en llegar (por ejemplo, mientras Ollama lee el documento), mueve el reloj.
    """
    cola = queue.Queue()
    cab = cabeceras()

    def lector():
        for trozo in leer_stream(ruta, resultado, cab=cab):
            cola.put(trozo)
        cola.put(None)  # señal de que terminó

    threading.Thread(target=lector, daemon=True).start()
    while True:
        reloj.caption(f"⏱ {int(time.perf_counter() - inicio)} s")
        try:
            trozo = cola.get(timeout=0.5)
        except queue.Empty:
            continue
        if trozo is None:
            return
        yield trozo


def con_contador(texto: str, funcion):
    """Ejecuta `funcion` en segundo plano y muestra un contador de segundos mientras tarda.

    Devuelve (lo que devolvió la función, segundos que tardó).
    """
    salida = {}

    def tarea():
        try:
            salida["valor"] = funcion()
        except Exception:
            salida["valor"] = (None, "Ocurrió un error inesperado.")

    hilo = threading.Thread(target=tarea, daemon=True)
    hilo.start()
    caja = st.empty()
    inicio = time.perf_counter()
    while hilo.is_alive():
        caja.info(f"⏳ {texto}... **{int(time.perf_counter() - inicio)} s**")
        time.sleep(0.5)
    hilo.join()
    caja.empty()
    return salida["valor"], time.perf_counter() - inicio


# --------------------------------------------------------------- visor de PDF
PAGINAS_POR_TANDA = 10  # páginas que se dibujan de una vez (evita cargar PDFs enormes de golpe)


@st.cache_data(show_spinner=False)
def contar_paginas(_pdf: bytes, clave: str) -> int:
    """`clave` identifica el PDF para el caché (los parámetros con _ no se usan para eso)."""
    with pymupdf.open(stream=_pdf, filetype="pdf") as doc:
        return len(doc)


@st.cache_data(show_spinner=False, max_entries=200)
def dibujar_pagina(_pdf: bytes, clave: str, numero: int, giro: int = 0) -> bytes:
    """Convierte una página en imagen PNG, girada `giro` grados."""
    with pymupdf.open(stream=_pdf, filetype="pdf") as doc:
        pagina = doc[numero - 1]
        pagina.set_rotation((pagina.rotation + giro) % 360)
        return pagina.get_pixmap(dpi=110).tobytes("png")


@st.cache_data(show_spinner=False, max_entries=50)
def imagen_resaltada(usuario: str, nombre: str, pagina: int, pasajes: tuple, giro_extra: int):
    """Pide a la API la página con los pasajes pintados. Devuelve (png, encontrado).

    Si falla lanza RuntimeError (y así el error no queda guardado en el caché).
    """
    try:
        r = httpx.post(f"{API}/documentos/{quote(nombre)}/resaltar", headers=cabeceras(usuario),
                       json={"pagina": pagina, "pasajes": list(pasajes), "giro_extra": giro_extra},
                       timeout=120)
    except httpx.HTTPError:
        raise RuntimeError("No se pudo conectar con la API. ¿Está corriendo uvicorn?")
    if r.status_code >= 400:
        try:
            raise RuntimeError(r.json().get("detail", "No se pudo marcar el pasaje."))
        except ValueError:
            raise RuntimeError("No se pudo marcar el pasaje.")
    return r.content, r.headers.get("x-encontrado") == "1"


def ver_en_pdf(detalle: dict):
    """Se ejecuta al pulsar '📍 Ver en el PDF': el visor muestra ese pasaje pintado.

    Si la fuente es un audio o video, el reproductor del visor arranca en ese minuto.
    """
    if detalle.get("inicio") is not None:
        st.session_state.reproducir = detalle
    else:
        st.session_state.resaltar = detalle
    st.session_state.doc_actual = detalle["documento"]


def quitar_resaltado():
    st.session_state.pop("resaltar", None)


def visor_pdf(nombre: str, pdf: bytes, giros: dict, resaltado: dict | None = None, caja: bool = True):
    """Muestra todas las páginas una debajo de otra, con scroll, ajustadas al ancho.

    Así se ven completas tanto las hojas verticales como las horizontales.
    `giros` trae los grados para enderezar las hojas escaneadas de lado.
    `resaltado`: página con el pasaje de la respuesta pintado (se muestra arriba de todo).
    """
    clave = hashlib.md5(pdf).hexdigest()
    total = contar_paginas(pdf, clave)
    k_visibles = f"visibles_{nombre}"
    visibles = min(st.session_state.get(k_visibles, PAGINAS_POR_TANDA), total)

    def girar(n: int):
        # Cada página recuerda cuánto la giró el usuario, además del giro automático
        k = f"giro_{nombre}_{n}"
        st.session_state[k] = (st.session_state.get(k, 0) + 90) % 360

    def cargar_mas():
        st.session_state[k_visibles] = visibles + PAGINAS_POR_TANDA

    # caja con scroll para recorrer todo el PDF (si hay varios documentos, la caja es una sola y la pone el panel)
    with (st.container(height=ALTURA) if caja else nullcontext()):
        if resaltado:
            c1, c2 = st.columns([6, 1], vertical_alignment="center")
            c1.markdown(f"📍 **{resaltado['titulo']} — pág. {resaltado['pagina']}**")
            c2.button("✖", key="quitar_resaltado", on_click=quitar_resaltado,
                      help="Quitar el resaltado")
            if resaltado["con_pasajes"] and not resaltado["encontrado"]:
                st.caption("No pude ubicar el pasaje exacto; esta es la página citada.")
            st.image(resaltado["png"], width="stretch")
            st.divider()
        for n in range(1, visibles + 1):
            giro = (giros.get(str(n), 0) + st.session_state.get(f"giro_{nombre}_{n}", 0)) % 360
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            c1.caption(f"Página {n} de {total}")
            c2.button("↻", key=f"girar_{nombre}_{n}", on_click=girar, args=(n,),
                      help="Girar esta página 90°")
            st.image(dibujar_pagina(pdf, clave, n, giro), width="stretch")
        if visibles < total:
            st.button(f"Cargar más páginas ({visibles} de {total})",
                      key=f"mas_{nombre}", on_click=cargar_mas)


# ------------------------------------------------------ conversaciones guardadas
def agregar_mensaje(mensaje: dict):
    """Agrega un mensaje al chat y lo guarda en el historial del usuario."""
    st.session_state.mensajes.append(mensaje)
    if st.session_state.conv_id is None:
        datos, _ = llamar("POST", "/conversaciones")
        if datos:
            st.session_state.conv_id = datos["id"]
            st.session_state.refrescar = True  # para que aparezca en la lista de la barra lateral
    if st.session_state.conv_id is not None:
        extra = {k: v for k, v in mensaje.items() if k not in ("rol", "texto")}
        llamar("POST", f"/conversaciones/{st.session_state.conv_id}/mensajes",
               json={"rol": mensaje["rol"], "texto": mensaje["texto"], "extra": extra})


def nueva_conversacion():
    st.session_state.update(mensajes=[], conv_id=None, pendientes=[], miniaturas={})


def abrir_conversacion(conv_id: int):
    mensajes, error = llamar("GET", f"/conversaciones/{conv_id}/mensajes")
    if not error:
        st.session_state.update(mensajes=mensajes, conv_id=conv_id, pendientes=[], miniaturas={})


def borrar_conversacion(conv_id: int):
    llamar("DELETE", f"/conversaciones/{conv_id}")
    if st.session_state.conv_id == conv_id:
        nueva_conversacion()


def cambiar_clave(numero: int):
    """Enciende o apaga una clave de Gemini según su interruptor."""
    llamar("POST", f"/gemini/claves/{numero}", json={"activa": bool(st.session_state[f"clave_gemini_{numero}"])})


def guardar_clave(numero: int):
    """Guarda la clave pegada en el campo y lo vacía (la clave no se vuelve a mostrar)."""
    texto = st.session_state.get(f"clave_txt_{numero}", "")
    _, error = llamar("PUT", f"/gemini/claves/{numero}/clave", json={"clave": texto})
    st.session_state.mensaje_clave = (numero, error or "Clave guardada.", bool(error))
    st.session_state[f"clave_txt_{numero}"] = ""


NOMBRES_EMBEDDINGS = {"ollama": "Ollama · nomic-embed-text (local)",
                      "gemma2": "EmbeddingGemma 2 (local, multimodal)",
                      "gemma": "Ollama · EmbeddingGemma 1 (local)",
                      "gemini": "Gemini · gemini-embedding-2 (nube)"}


def cambiar_embeddings():
    """Elige con qué modelo se calculan los vectores de los PDFs nuevos."""
    proveedor = st.session_state.get("embeddings_modelo", "ollama")
    _, error = llamar("POST", "/embeddings", json={"proveedor": proveedor})
    if error:
        st.session_state.mensaje_clave = (0, error, True)


NOMBRES_CHAT = {"ollama": "Local · Ollama (qwen2.5:1.5b)", "llamacpp": "Local · llama.cpp (qwen2.5:1.5b)",
                "gemini": "Nube · Gemini (API)"}


def cambiar_chat():
    """Elige quién escribe las respuestas: el modelo local o uno de la nube."""
    _, error = llamar("POST", "/chat/modelo", json={"proveedor": st.session_state.get("chat_modelo")})
    if error:
        st.session_state.mensaje_clave = (0, error, True)


NOMBRES_MEDIOS = {"gemma2": "EmbeddingGemma 2 (local): imágenes, audio y video",
                  "gemini": "Gemini (nube): solo imágenes"}


def cambiar_medios():
    """Elige con qué modelo se calculan los vectores de imágenes, audios y videos nuevos."""
    _, error = llamar("POST", "/embeddings/medios", json={"proveedor": st.session_state.get("medios_modelo")})
    if error:
        st.session_state.mensaje_clave = (0, error, True)


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.markdown("## 🧠 Base de Conocimiento IA")
    st.caption("PDFs · imágenes · audio · video")
    st.text_input("👤 Usuario", key="usuario_txt", value=USUARIO_POR_DEFECTO,
                  help="Cada usuario ve solo sus documentos y conversaciones. "
                       "No usa contraseña: solo separa la información de cada nombre.")

    estado, error = llamar("GET", "/estado")
    if error:
        st.error(error)
    else:
        if estado["modo"] == "ollama":
            st.success("Modo: OLLAMA")
        else:
            st.warning("Modo: PRUEBA")
        # Qué modelos de IA se están usando ahora
        st.markdown(f"🧠 **Chat:** `{estado['chat']}`  \n🔎 **Embeddings:** `{estado['embeddings']}`")
        if estado.get("gemini"):
            st.caption("☁️ Gemini activo: se pueden buscar imágenes dentro de los PDFs.")
        else:
            st.caption("☁️ Gemini sin clave activa: revisa las claves de abajo o pega `GEMINI_API_KEY` en el `.env`.")
        datos_claves, _ = llamar("GET", "/gemini/claves")
        with st.expander("☁️ Claves de Gemini"):
            iconos = {"lista": "🟢", "en_pausa": "🟠", "invalida": "🔴", "desactivada": "⚪", "sin_configurar": "⚫"}
            for c in (datos_claves or {}).get("claves", []):
                st.toggle(f"Clave {c['numero']}: {iconos.get(c['estado'], '')} {c['texto']}",
                          value=c["activa"], disabled=not c["configurada"], key=f"clave_gemini_{c['numero']}",
                          on_change=cambiar_clave, args=(c["numero"],))
            st.caption("Se usan en orden 1, 2, 3. Si una llega a su límite o falla, pasa sola a la siguiente.")
            st.markdown("**Agregar o cambiar una clave**")
            numero_clave = st.selectbox("Clave", [1, 2, 3], key="clave_elegida", label_visibility="collapsed",
                                        format_func=lambda n: f"Clave {n}")
            st.text_input("Pega la clave", type="password", key=f"clave_txt_{numero_clave}",
                          placeholder="Pega aquí la clave de Gemini", label_visibility="collapsed")
            st.button("💾 Guardar clave", on_click=guardar_clave, args=(numero_clave,), width="stretch")
            aviso = st.session_state.pop("mensaje_clave", None)
            if aviso:
                (st.error if aviso[2] else st.success)(aviso[1])
            st.caption("Se guardan en la base de datos de este computador y nunca se muestran. "
                       "Para borrar una, guarda el campo vacío.")
        modelo_emb, _ = llamar("GET", "/embeddings")
        actual = (modelo_emb or {}).get("proveedor", "ollama")
        st.session_state["embeddings_modelo"] = actual  # siempre refleja lo guardado (p. ej. si fue rechazado)
        st.selectbox("🔎 Embeddings al subir PDFs", list(NOMBRES_EMBEDDINGS), key="embeddings_modelo",
                     format_func=NOMBRES_EMBEDDINGS.get, on_change=cambiar_embeddings,
                     help="Con qué modelo se calculan los vectores de los PDFs que subas desde ahora. "
                          "Los PDFs ya subidos conservan el suyo. Gemini (nube) es mucho más rápido, pero "
                          "el texto de los PDFs se envía a Google. La respuesta del chat siempre es local.")
        if actual == "gemini":
            st.caption("⚠️ El texto de los PDFs nuevos se envía a Google. La respuesta del chat sigue siendo local.")
        chat_emb, _ = llamar("GET", "/chat/modelo")
        st.session_state["chat_modelo"] = (chat_emb or {}).get("proveedor", "ollama")
        st.selectbox("🧠 Quién escribe las respuestas", list(NOMBRES_CHAT), key="chat_modelo",
                     format_func=lambda k: (f"Nube · Gemini ({(chat_emb or {}).get('modelo_nube', 'API')})"
                                            if k == "gemini" else
                                            NOMBRES_CHAT[k].replace("qwen2.5:1.5b", (chat_emb or {}).get("modelo_local", ""))),
                     on_change=cambiar_chat,
                     help="Local: el modelo corre en tu computador, sin internet. Nube: Gemini por API (con tus "
                          "claves); también es el que puede VER las imágenes para describirlas.")
        if (chat_emb or {}).get("proveedor") == "gemini":
            st.caption("⚠️ Tus preguntas y los fragmentos encontrados se envían a Google para escribir la respuesta.")
        medios_emb, _ = llamar("GET", "/embeddings/medios")
        st.session_state["medios_modelo"] = (medios_emb or {}).get("proveedor", "gemma2")
        st.selectbox("🖼️ Embeddings de imágenes, audio y video", list(NOMBRES_MEDIOS), key="medios_modelo",
                     format_func=NOMBRES_MEDIOS.get, on_change=cambiar_medios,
                     help="EmbeddingGemma 2 entiende texto, imágenes, audio y video en un mismo espacio y corre "
                          "en tu computador. Gemini solo sirve para imágenes y las envía a Google.")
        if estado["ollama_activo"] is False:
            st.error("Ollama no responde. ¿Está instalado y corriendo?")
        for faltante in estado["faltan"]:
            st.warning(f"El modelo `{faltante}` no está instalado. Ejecuta: `ollama pull {faltante}`")
        if not estado["base_de_datos"]:
            st.error("La base de datos no responde. ¿Está corriendo Docker?")

    st.divider()
    st.subheader("📄 Documentos")
    # La clave incluye al usuario: al cambiar de usuario el cargador queda vacío
    archivos = st.file_uploader("Sube PDFs, audios o videos cortos",
                                type=["pdf", "mp3", "wav", "m4a", "ogg", "flac", "mp4", "webm", "mov", "mkv", "avi"],
                                accept_multiple_files=True,
                                key=f"subir_{usuario_actual()}")
    if archivos:
        for archivo in archivos:
            clave = (archivo.name, archivo.size)
            if clave in st.session_state.procesados:
                continue
            contenido = archivo.getvalue()
            cab = cabeceras()
            (datos, error), segundos = con_contador(
                f"Procesando {archivo.name} (leer, OCR e indexar)",
                lambda n=archivo.name, c=contenido, h=cab: llamar(
                    "POST", "/subir", cab=h, files={"archivo": (n, c, "application/pdf")}
                ),
            )
            st.session_state.procesados.add(clave)
            if error:
                st.error(f"{archivo.name}: {error}")
                continue
            st.success(f"{datos['mensaje']} ⏱ {segundos:.0f} s")
            for aviso in datos.get("advertencias", []):
                st.warning(aviso)
            # El documento se muestra a la izquierda y su resumen se escribe en el chat
            st.session_state.doc_actual = archivo.name
            st.session_state.pendientes.append(("resumen", archivo.name))

    carpeta, _ = llamar("GET", "/carpeta")
    if carpeta and carpeta["existe"]:
        faltan = [a for a in carpeta["archivos"] if not a["indexado"]]
        if st.button(f"📂 Indexar carpeta de conocimiento ({len(faltan)} nuevos)", width="stretch",
                     disabled=not faltan,
                     help=f"Indexa los PDFs, audios y videos que haya en {carpeta['carpeta']}"):
            cab = cabeceras()
            (indexada, error), seg = con_contador(
                "Indexando la carpeta (los audios y videos tardan más)",
                lambda: llamar("POST", "/carpeta/indexar", cab=cab))
            if error:
                st.error(error)
            else:
                for r in indexada["resultados"]:
                    (st.success if r["ok"] else st.error)(f"{r['nombre']}: {r['mensaje']}")
                    for aviso in r["advertencias"]:
                        st.warning(aviso)
                st.caption(f"⏱ {seg:.0f} s")

    docs, error = llamar("GET", "/documentos")
    nombres = [d["nombre"] for d in docs] if docs else []
    tipo_de = {d["nombre"]: d.get("tipo", "pdf") for d in docs or []}
    if docs:
        st.caption("Marca los documentos en los que quieres buscar:")
        for d in docs:
            col1, col2 = st.columns([6, 1], vertical_alignment="center")
            fecha = (d.get("creado") or "")[:10]
            nube = {"gemini": " ☁️", "gemma": " 💎", "gemma2": " 💎2"}.get(d.get("embeddings"), "")
            icono = {"audio": "🎵 ", "video": "🎬 "}.get(d.get("tipo"), "")
            col1.checkbox(f"{icono}{d['nombre']} ({d['fragmentos']}){nube}", value=True,
                          key=f"buscar_{d['nombre']}", help=f"Subido el {fecha}" if fecha else None)
            if col2.button("🗑", key=f"del_{d['nombre']}", help="Eliminar"):
                _, error = llamar("DELETE", f"/documentos/{quote(d['nombre'])}")
                if error:
                    st.error(error)
                # Permite volver a subir el mismo archivo después de borrarlo
                st.session_state.procesados -= {c for c in st.session_state.procesados
                                                if c[0] == d["nombre"]}
                st.rerun()
    elif docs is not None:
        st.caption("Aún no hay documentos.")

    st.divider()
    st.subheader("💬 Conversaciones")
    st.button("➕ Nueva conversación", on_click=nueva_conversacion, width="stretch")
    conversaciones, _ = llamar("GET", "/conversaciones")
    for conv in conversaciones or []:
        col1, col2 = st.columns([6, 1], vertical_alignment="center")
        activa = conv["id"] == st.session_state.conv_id
        col1.button(conv["titulo"], key=f"conv_{conv['id']}", on_click=abrir_conversacion,
                    args=(conv["id"],), type="primary" if activa else "secondary", width="stretch")
        col2.button("🗑", key=f"delconv_{conv['id']}", on_click=borrar_conversacion,
                    args=(conv["id"],), help="Eliminar esta conversación")


def documentos_elegidos() -> list[str] | None:
    """Documentos marcados para buscar. None = todos (no hace falta filtrar)."""
    elegidos = [n for n in nombres if st.session_state.get(f"buscar_{n}", True)]
    return None if len(elegidos) == len(nombres) else elegidos


# ------------------------------------------------------ panel del documento (izquierda)
def descargar(ruta: str, usuario: str, params: dict | None = None):
    """Función que trae el archivo a exportar. Se ejecuta solo al pulsar el botón de descarga."""
    def traer() -> bytes:
        r = httpx.get(API + ruta, params=params, headers=cabeceras(usuario), timeout=120)
        r.raise_for_status()
        return r.content
    return traer


def botones_exportar(nombre: str, tiene_resumen: bool):
    """Descargas del resumen del documento y de la conversación, en PDF o Word."""
    usuario = usuario_actual()
    base = nombre.rsplit(".", 1)[0]
    conv_id = st.session_state.conv_id
    tipos = {"pdf": "application/pdf",
             "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"}
    c1, c2, c3, c4 = st.columns(4)
    for col, formato in zip((c1, c2), tipos):
        col.download_button(
            f"Resumen {formato.upper()}", key=f"exp_res_{formato}", mime=tipos[formato],
            data=descargar(f"/exportar/resumen/{quote(nombre)}", usuario, {"formato": formato}),
            file_name=f"resumen_{base}.{formato}", disabled=not tiene_resumen, width="stretch")
    for col, formato in zip((c3, c4), tipos):
        col.download_button(
            f"Chat {formato.upper()}", key=f"exp_chat_{formato}", mime=tipos[formato],
            data=descargar(f"/exportar/chat/{conv_id}", usuario, {"formato": formato}),
            file_name=f"conversacion.{formato}", disabled=conv_id is None, width="stretch")
    if not tiene_resumen:
        st.caption("Genera el resumen, los puntos clave o el glosario para poder exportarlos.")


def panel_documento():
    """Muestra el selector de documento, el PDF abierto y los botones para trabajar con él."""
    st.subheader("📄 Documento")
    if not nombres:
        st.info("Sube un PDF desde la barra lateral para verlo aquí.")
        return
    # Se muestran solo los documentos marcados en la barra lateral
    marcados = [n for n in nombres if st.session_state.get(f"buscar_{n}", True)]
    pedido = st.session_state.get("resaltar")
    escuchar = st.session_state.get("reproducir")
    for cita in (pedido, escuchar):
        if cita and cita["documento"] in nombres and cita["documento"] not in marcados:
            marcados.append(cita["documento"])  # la cita que se quiere ver siempre se muestra
    if not marcados:
        st.info("Marca en la barra lateral el documento que quieres ver.")
        return
    if st.session_state.get("doc_actual") not in marcados:
        st.session_state.doc_actual = marcados[0]
    if len(marcados) > 1:
        nombre = st.selectbox("Documento para resumen, puntos clave y exportar", marcados, key="doc_actual")
    else:
        nombre = marcados[0]
        st.session_state.doc_actual = nombre
        st.caption(f"📄 {nombre}")

    multiples = len(marcados) > 1
    with (st.container(height=ALTURA) if multiples else nullcontext()):  # varios: un solo scroll
        for doc in marcados:
            if multiples:
                st.markdown(f"#### {'🎵' if tipo_de.get(doc) == 'audio' else '🎬' if tipo_de.get(doc) == 'video' else '📄'} {doc}")
            if tipo_de.get(doc, "pdf") != "pdf":  # audio o video suelto: se reproduce
                datos_medio = obtener_medio_suelto(usuario_actual(), doc)
                if datos_medio is None:
                    st.warning("No pude cargar ese archivo.")
                else:
                    desde = int(escuchar["inicio"]) if escuchar and escuchar["documento"] == doc else 0
                    if desde:
                        st.caption(f"▶ Reproduciendo desde {_minutos(desde)}")
                    (st.audio if tipo_de[doc] == "audio" else st.video)(datos_medio, start_time=desde)
                continue
            pdf, error = obtener_pdf(doc)
            if error:
                st.warning(error)
                continue
            datos, _ = llamar("GET", f"/documentos/{quote(doc)}/giros")
            resaltado = None
            if pedido and pedido["documento"] == doc:
                giro_extra = st.session_state.get(f"giro_{doc}_{pedido['pagina']}", 0)
                try:
                    with st.spinner("Marcando la fuente en el PDF..."):
                        png, encontrado = imagen_resaltada(usuario_actual(), doc, pedido["pagina"],
                                                           tuple(pedido["pasajes"]), giro_extra)
                    resaltado = {"pagina": pedido["pagina"], "png": png, "encontrado": encontrado,
                                 "titulo": pedido.get("titulo", "Fuente de la respuesta"),
                                 "con_pasajes": bool(pedido["pasajes"])}
                except RuntimeError as e:
                    st.warning(str(e))
            visor_pdf(doc, pdf, datos["giros"] if datos else {}, resaltado, caja=not multiples)

    # El resultado de estos botones se escribe en el chat, no aquí
    b1, b2, b3 = st.columns(3)
    if b1.button("📝 Resumen", width="stretch"):
        st.session_state.pendientes.append(("resumen", nombre))
    if b2.button("🔑 Puntos clave", width="stretch"):
        st.session_state.pendientes.append(("puntos_clave", nombre))
    if b3.button("📚 Glosario", width="stretch"):
        st.session_state.pendientes.append(("glosario", nombre))

    if tipo_de.get(nombre, "pdf") == "pdf" and st.button("🖼️ Indexar imágenes, audio y video", width="stretch",
                 help="Calcula los vectores de las imágenes de este PDF y de los audios y videos que traiga "
                      "adjuntos, para poder buscarlos con una frase. Con EmbeddingGemma 2 todo queda en tu "
                      "computador; con Gemini las imágenes se envían a Google."):
        cab = cabeceras()
        (datos, error), seg = con_contador(
            "Indexando imágenes, audio y video",
            lambda: llamar("POST", f"/documentos/{quote(nombre)}/indexar_imagenes", cab=cab))
        if error:
            st.error(error)
        else:
            st.success(f"Imágenes indexadas: {datos['indexadas']} de {datos['total']} · audios: "
                       f"{datos.get('audios', 0)} · videos: {datos.get('videos', 0)} ({seg:.0f} s).")
            if datos.get("error"):
                st.warning(datos["error"])

    guardado, _ = llamar("GET", f"/documentos/{quote(nombre)}/resumen")
    with st.expander("⬇️ Exportar a PDF o Word"):
        botones_exportar(nombre, bool(guardado and any(guardado.values())))


# ------------------------------------------------------------ panel del chat (derecha)
def _normalizada(texto: str) -> str:
    sin = "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                  if unicodedata.category(c) != "Mn")
    return " ".join(re.findall(r"\w+", sin))


def detalle_tecnico(f: dict) -> str:
    """Posición del fragmento y por qué se eligió: 'fragmento 12 de 42 · significado 72 % · …'"""
    partes = []
    if f.get("posicion") and f.get("total_documento"):
        partes.append(f"fragmento {f['posicion']} de {f['total_documento']} del documento")
    if f.get("id") is not None:
        partes.append(f"id {f['id']}")
    if "similitud" in f:
        partes.append(f"parecido por significado {f['similitud'] * 100:.0f} %")
    if "coincidencia" in f:
        partes.append(f"palabras de la pregunta {f['coincidencia'] * 100:.0f} %")
    if "puntaje" in f:
        partes.append(f"puntaje final {f['puntaje']:.2f}")
    if f.get("hallado_por"):
        partes.append(f"hallado por {f['hallado_por']}")
    return " · ".join(partes)


def mostrar_citas(fuentes: list[dict] | None, citas: list[int] | None,
                  detalle: list[dict] | None = None, indice: int = 0,
                  total_buscado: int | None = None):
    """De dónde salió la respuesta: botones para verla pintada en el PDF y los fragmentos."""
    if not fuentes and not detalle:
        return
    citas = citas or []
    detalle = detalle or []
    if len(detalle) > 4:  # p. ej. el resumen: una fuente por página, en un desplegable y con botones cortos
        with st.expander(f"📍 Ver en el PDF de dónde salió esto ({len(detalle)} páginas)"):
            columnas = st.columns(3)
            for i, d in enumerate(detalle):
                columnas[i % 3].button(f"📍 pág. {d['pagina']}", key=f"ver_{indice}_{d['n']}",
                                       on_click=ver_en_pdf, args=(d,), width="stretch")
    else:
        for d in detalle:
            if d.get("inicio") is not None:
                etiqueta = f"▶ Reproducir desde {_minutos(d['inicio'])}: {d['documento']}"
            else:
                etiqueta = f"📍 Ver en el PDF: {d['documento']}, pág. {d['pagina']}"
            st.button(etiqueta, key=f"ver_{indice}_{d['n']}", on_click=ver_en_pdf, args=(d,))
    if not fuentes:
        return
    if not detalle:
        usadas = [f for f in fuentes if f.get("n") in citas]
        if usadas:
            st.caption("📎 **Fuente:** " + " · ".join(
                f"{f['documento']}, pág. {f['pagina']}" for f in usadas))
    de_total = f" de {total_buscado} en los documentos buscados" if total_buscado else ""
    with st.expander(f"Fragmentos consultados: {len(fuentes)}{de_total}"):
        for f in fuentes:
            marca = " ✅ usado" if f.get("n") in citas else ""
            donde = f"min {_minutos(f['inicio'])}" if f.get("inicio") is not None else f"pág. {f['pagina']}"
            st.markdown(f"**[{f.get('n', '·')}] {f['documento']} — {donde}**{marca}")
            tecnico = detalle_tecnico(f)
            if tecnico:
                st.caption(tecnico)
            if f.get("fragmento"):
                st.markdown("> " + f["fragmento"].replace("\n", " "))


def mostrar_interpretada(pregunta: str, interpretada: str | None):
    """Si la pregunta era un seguimiento, muestra cómo la entendió el sistema."""
    if interpretada and _normalizada(interpretada) != _normalizada(pregunta):
        st.caption(f"🔎 Entendí tu pregunta como: «{interpretada}»")


def mostrar_tiempos(t, modelo: str | None = None):
    """Línea pequeña con el modelo usado y los tiempos de cada etapa (pruebas de velocidad)."""
    if t:
        # Solo las etapas que ocurrieron (una respuesta sin modelo no tiene embedding ni Ollama)
        etapas = []
        if t.get("reformular_ms"):
            etapas.append(f"Reformular {t['reformular_ms']} ms")
        if "embedding_ms" in t:
            etapas.append(f"Embedding {t['embedding_ms']} ms")
        if "busqueda_db_ms" in t:
            etapas.append(f"Búsqueda BD {t['busqueda_db_ms']} ms")
        if "ollama_ms" in t:
            etapas.append(f"Ollama: 1ª palabra {t.get('ollama_primer_token_ms', 0)} ms, "
                          f"total {t['ollama_ms']} ms")
        etapas.append(f"**Total {t.get('total_ms', 0)} ms**")
        st.caption(f"🧠 {modelo or '—'} · ⏱ " + " · ".join(etapas))


def preguntar(texto: str):
    """Se ejecuta al pulsar una pregunta sugerida: la deja lista para enviarla al chat."""
    st.session_state.pregunta_pendiente = texto


def mostrar_sugerencias(indice: int, sugerencias: list[str] | None):
    """Muestra las preguntas sugeridas como botones. `indice` hace únicas sus claves."""
    if not sugerencias:
        return
    st.caption("💡 Preguntas que puedes hacer:")
    for i, sugerencia in enumerate(sugerencias):
        st.button(sugerencia, key=f"sug_{indice}_{i}", on_click=preguntar, args=(sugerencia,))


def modelo_en_uso() -> str:
    return (estado or {}).get("chat", "—")


def texto_de_cobertura(c: dict) -> str:
    """'🧩 Resumen basado en 12 de 42 fragmentos (uno por página, págs. 1–12), ≈ 38 % del texto.'"""
    paginas = c["paginas_usadas"]
    if len(paginas) > 1 and paginas == list(range(paginas[0], paginas[-1] + 1)):
        lista = f"págs. {paginas[0]}–{paginas[-1]}"
    else:
        lista = "págs. " + ", ".join(str(p) for p in paginas)
    parcial = "" if c["paginas_usadas"] == c["paginas_total"] or len(paginas) == c["paginas_total"] \
        else f" de {c['paginas_total']} páginas"
    return (f"🧩 Resumen basado en {c['fragmentos_usados']} de {c['fragmentos_total']} fragmentos "
            f"(uno por página: {lista}{parcial}), ≈ {c['porcentaje']} % del texto.")


def resumir_en_chat(nombre: str):
    """Escribe el resumen del PDF dentro de la conversación, palabra por palabra."""
    sugerencias, detalle_resumen = [], []
    with st.chat_message("assistant"):
        encabezado = f"📄 **{nombre}** — resumen:"
        st.write(encabezado)
        resultado = {}
        reloj = st.empty()  # aquí corre el contador de segundos
        inicio = time.perf_counter()
        texto = st.write_stream(
            leer_stream_con_contador(
                f"/documentos/{quote(nombre)}/resumen_stream", resultado, reloj, inicio
            )
        )
        segundos = time.perf_counter() - inicio
        if resultado.get("error"):
            reloj.empty()
            st.error(f"No pude generar el resumen: {resultado['error']}")
            texto = f"No pude generar el resumen: {resultado['error']}"
        else:
            pie = f"🧠 {modelo_en_uso()} · ⏱ Resumen generado en {segundos:.0f} s"
            reloj.caption(pie)
            texto += f"\n\n*{pie}*"
            cobertura = resultado.get("cobertura")
            if cobertura:  # cuántos fragmentos y páginas usó realmente el resumen
                aviso = texto_de_cobertura(cobertura)
                st.caption(aviso)
                texto += f"\n\n*{aviso}*"
                detalle_resumen = cobertura.get("detalle") or []
                mostrar_citas(None, None, detalle_resumen, len(st.session_state.mensajes))
                if cobertura.get("cifras_sin_respaldo"):  # el modelo pudo inventar un número
                    alerta = ("⚠️ El resumen menciona cifras que no aparecen en el texto leído: "
                              + ", ".join(cobertura["cifras_sin_respaldo"]) + ". Verifícalas en el PDF.")
                    st.warning(alerta)
                    texto += f"\n\n*{alerta}*"
            # Después del resumen, 3 preguntas sugeridas que se envían con un clic
            cab = cabeceras()
            (datos, _), _ = con_contador(
                "Preparando preguntas sugeridas",
                lambda: llamar("POST", f"/documentos/{quote(nombre)}/preguntas", cab=cab),
            )
            sugerencias = datos["preguntas"] if datos else []
            mostrar_sugerencias(len(st.session_state.mensajes), sugerencias)
    agregar_mensaje({"rol": "assistant", "texto": f"{encabezado}\n\n{texto}",
                     "sugerencias": sugerencias, "citas_detalle": detalle_resumen})


ETIQUETAS = {"puntos_clave": ("🔑", "puntos clave"), "glosario": ("📚", "glosario")}


def generar_en_chat(tipo: str, nombre: str):
    """Genera los puntos clave o el glosario del PDF y los escribe en la conversación."""
    icono, etiqueta = ETIQUETAS[tipo]
    with st.chat_message("assistant"):
        encabezado = f"{icono} **{nombre}** — {etiqueta}:"
        st.write(encabezado)
        cab = cabeceras()
        (datos, error), segundos = con_contador(
            f"Generando {etiqueta}",
            lambda: llamar("POST", f"/documentos/{quote(nombre)}/{tipo}", cab=cab),
        )
        if error:
            texto = f"No pude generar los {etiqueta}: {error}"
            st.error(texto)
        else:
            pie = f"🧠 {modelo_en_uso()} · ⏱ {segundos:.0f} s"
            st.markdown(datos[tipo])
            st.caption(pie)
            texto = f"{datos[tipo]}\n\n*{pie}*"
    agregar_mensaje({"rol": "assistant", "texto": f"{encabezado}\n\n{texto}"})


@st.cache_data(show_spinner=False, max_entries=100)
def obtener_miniatura(usuario: str, imagen_id: int) -> bytes | None:
    """Descarga una imagen indexada de los documentos (para mostrarla en el chat)."""
    try:
        r = httpx.get(f"{API}/imagenes/{imagen_id}", headers=cabeceras(usuario), timeout=30)
    except httpx.HTTPError:
        return None
    return r.content if r.status_code == 200 else None


@st.cache_data(show_spinner=False, max_entries=20)
def obtener_archivo_de_medio(usuario: str, imagen_id: int) -> bytes | None:
    """El audio o video original al que pertenece un resultado."""
    try:
        r = httpx.get(f"{API}/medios/{imagen_id}/archivo", headers=cabeceras(usuario), timeout=120)
    except httpx.HTTPError:
        return None
    return r.content if r.status_code == 200 else None


@st.cache_data(show_spinner=False, max_entries=10)
def obtener_medio_suelto(usuario: str, nombre: str) -> bytes | None:
    """El audio o video de un documento suelto, para el visor."""
    try:
        r = httpx.get(f"{API}/documentos/{quote(nombre)}/archivo_medio", headers=cabeceras(usuario), timeout=120)
    except httpx.HTTPError:
        return None
    return r.content if r.status_code == 200 else None


def _minutos(segundos: float | None) -> str:
    s = int(segundos or 0)
    return f"{s // 60}:{s % 60:02d}"


def mostrar_imagenes_encontradas(imagenes: list[dict] | None):
    """Imágenes, audios y videos más parecidos a lo que se preguntó, con su porcentaje de parecido."""
    if not imagenes:
        return
    for inicio in range(0, len(imagenes), 3):
        grupo = imagenes[inicio:inicio + 3]
        for columna, im in zip(st.columns(3), grupo):
            tipo = im.get("tipo", "imagen")
            parecido = f"parecido {im['similitud'] * 100:.0f} %"
            with columna:
                if tipo == "imagen":
                    png = obtener_miniatura(usuario_actual(), im["id"])
                    if png:
                        st.image(png, width="stretch",
                                 caption=f"{im['documento']} · pág. {im['pagina']} · {parecido}")
                    continue
                desde = f"desde {_minutos(im.get('inicio'))}"
                archivo = obtener_archivo_de_medio(usuario_actual(), im["id"])
                if tipo == "audio":
                    st.caption(f"🎵 {im.get('nombre') or im['documento']} · {desde} · {parecido}")
                    if archivo:
                        st.audio(archivo, start_time=int(im.get("inicio") or 0))
                else:
                    st.caption(f"🎬 {im.get('nombre') or im['documento']} · {desde} · {parecido}")
                    if archivo:
                        st.video(archivo, start_time=int(im.get("inicio") or 0))


def mostrar_imagen_adjunta(indice: int, mensaje: dict):
    """Miniatura de la imagen adjunta y el texto que se leyó en ella."""
    imagen = mensaje.get("imagen")
    if not imagen:
        return
    for datos in st.session_state.get("miniaturas", {}).get(indice, []):
        st.image(datos, width=220)
    if imagen.get("texto"):
        st.caption(f"🖼️ Texto que leí en la imagen: «{imagen['texto']}»")
    else:
        st.caption("🖼️ No encontré letras en la imagen. Todavía no puedo describir lo que muestra: "
                   "solo leo el texto que contiene.")


def leer_imagenes(imagenes: list) -> list[tuple[str, str]]:
    """Lee el texto de cada imagen adjunta. Devuelve [(nombre, texto)]; texto vacío si no hay letras."""
    leidas = []
    for imagen in imagenes:
        try:
            r = httpx.post(f"{API}/imagen/leer", headers=cabeceras(), timeout=120,
                           files={"archivo": (imagen.name, imagen.getvalue())})
            texto = r.json().get("texto", "") if r.status_code < 400 else ""
        except httpx.HTTPError:
            texto = ""
        leidas.append((imagen.name, texto))
    return leidas


def responder(pregunta: str, imagenes: list | None = None):
    """Envía la pregunta a la API (con la memoria de la conversación) y muestra la respuesta.

    Si se adjuntaron imágenes, se lee su texto y se agrega a la pregunta para buscar con él.
    """
    elegidos = documentos_elegidos()
    if elegidos == []:
        st.warning("Marca al menos un documento en la barra lateral para buscar.")
        return
    # Se recuerdan los mensajes anteriores para entender seguimientos como "¿y eso qué significa?"
    historial = [{"rol": m["rol"], "texto": m["texto"]} for m in st.session_state.mensajes[-6:]]

    texto_de_imagenes = ""
    nota_imagen = ""
    if imagenes:
        with st.spinner("Leyendo la imagen..."):
            leidas = leer_imagenes(imagenes)
        texto_de_imagenes = " ".join(t for _, t in leidas)[:500]
        nota_imagen = "📎 " + ", ".join(n for n, _ in leidas)
        pregunta = pregunta or "Busca en los documentos lo que dice esta imagen."
    indice_usuario = len(st.session_state.mensajes)
    if imagenes:  # las miniaturas se guardan solo en esta sesión (no en la base de datos)
        st.session_state.setdefault("miniaturas", {})[indice_usuario] = [i.getvalue() for i in imagenes]
    mensaje_usuario = {"rol": "user", "texto": f"{pregunta}\n\n{nota_imagen}" if nota_imagen else pregunta}
    if imagenes:
        mensaje_usuario["imagen"] = {"texto": texto_de_imagenes}
    agregar_mensaje(mensaje_usuario)
    with st.chat_message("user"):
        st.write(pregunta)
        mostrar_imagen_adjunta(indice_usuario, mensaje_usuario)
    # La pregunta que se envía incluye lo leído en la imagen; el usuario ve su pregunta original
    pregunta_enviada = f"{pregunta} [Texto de la imagen adjunta: {texto_de_imagenes}]" \
        if texto_de_imagenes else pregunta
    with st.chat_message("assistant"):
        resultado = {"fuentes": []}
        cuerpo = {"pregunta": pregunta_enviada, "documentos": elegidos, "historial": historial}
        with st.spinner("Pensando..."):
            # st.write_stream muestra el texto a medida que llega
            texto = st.write_stream(leer_stream("/chat_stream", resultado, cuerpo))
        if resultado.get("error"):
            st.error(resultado["error"])
        else:
            mostrar_interpretada(pregunta, resultado.get("pregunta_interpretada"))
            mostrar_imagenes_encontradas(resultado.get("imagenes"))
            mostrar_citas(resultado["fuentes"], resultado.get("citas"),
                          resultado.get("citas_detalle"), len(st.session_state.mensajes),
                          resultado.get("total_buscado"))
            mostrar_tiempos(resultado.get("tiempos"), resultado.get("modelo"))
            if resultado.get("citas_detalle"):
                # El visor de la izquierda se abre solo en el pasaje de donde salió la respuesta
                st.session_state.resaltar_pendiente = resultado["citas_detalle"][0]
                st.session_state.refrescar = True
            agregar_mensaje({
                "rol": "assistant", "texto": texto, "fuentes": resultado["fuentes"],
                "citas": resultado.get("citas"), "citas_detalle": resultado.get("citas_detalle"),
                "total_buscado": resultado.get("total_buscado"),
                "tiempos": resultado.get("tiempos"),
                "modelo": resultado.get("modelo"),
                "interpretada": resultado.get("pregunta_interpretada"),
                "imagenes": resultado.get("imagenes"),
                "pregunta": pregunta,
            })


def panel_chat():
    st.subheader("💬 Chat")
    caja = st.container(height=ALTURA)  # caja con scroll para los mensajes
    with caja:
        for indice, m in enumerate(st.session_state.mensajes):
            with st.chat_message(m["rol"]):
                st.write(m["texto"])
                mostrar_imagen_adjunta(indice, m)
                mostrar_interpretada(m.get("pregunta", ""), m.get("interpretada"))
                mostrar_imagenes_encontradas(m.get("imagenes"))
                mostrar_citas(m.get("fuentes"), m.get("citas"), m.get("citas_detalle"), indice,
                              m.get("total_buscado"))
                mostrar_tiempos(m.get("tiempos"), m.get("modelo"))
                mostrar_sugerencias(indice, m.get("sugerencias"))

        # Resumen, puntos clave y glosario: se escriben aquí, en la conversación
        while st.session_state.pendientes:
            tipo, nombre = st.session_state.pendientes.pop(0)
            if tipo == "resumen":
                resumir_en_chat(nombre)
            else:
                generar_en_chat(tipo, nombre)

    # El chat acepta texto y también imágenes (PNG o JPG): se lee el texto que contienen
    valor = st.chat_input("Escribe tu pregunta (puedes adjuntar una imagen)", accept_file=True,
                          file_type=["png", "jpg", "jpeg"])
    pregunta, imagenes = "", []
    if valor:
        pregunta = (valor.get("text") or "").strip()
        imagenes = valor.get("files") or []
    if not pregunta and not imagenes:  # o la que el usuario eligió entre las sugeridas
        pregunta = st.session_state.pop("pregunta_pendiente", None) or ""
    if pregunta or imagenes:
        with caja:
            responder(pregunta, imagenes)


# ---------- Página: documento a la izquierda, chat a la derecha ----------
st.title("Chat con tus PDFs")
col_doc, col_chat = st.columns(2, gap="large")
with col_doc:
    panel_documento()
with col_chat:
    panel_chat()

# Si se creó una conversación nueva, se recarga para que aparezca en la barra lateral
if st.session_state.pop("refrescar", False):
    st.rerun()
