"""Interfaz Streamlit de «Base de Conocimiento IA»: documento (izquierda) y chat (derecha) al mismo tiempo."""
import hashlib
from contextlib import nullcontext
import json
import queue
import re
import threading
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

import httpx
import pymupdf
import streamlit as st

from app import config
from app import segundo_plano as sp

API = config.API_URL


@st.cache_resource
def _cliente_api() -> httpx.Client:
    """Un solo cliente HTTP para toda la sesión (crear uno por llamada cuesta ~0,4 s)."""
    return httpx.Client(trust_env=False, verify=False)


CLIENTE = _cliente_api()


@st.cache_resource
def _gestor() -> sp.Gestor:
    """Las tareas largas (preguntas, subidas, resúmenes) viven aquí y siguen aunque la pantalla se recargue."""
    return sp.Gestor()


GESTOR = _gestor()
TIPOS_CHAT = ("pregunta", "resumen", "puntos_clave", "glosario", "subida")  # tareas que escriben en una conversación
ALTURA = 700  # alto (px) del visor de PDF y de la caja del chat
USUARIO_POR_DEFECTO = "invitado"

st.set_page_config(page_title="Base de Conocimiento IA", page_icon=":material/forum:", layout="wide")
# Estilo propio (colores, tarjetas, burbujas de chat...): ver estilo.css y .streamlit/config.toml
st.html("<style>" + Path(__file__).with_name("estilo.css").read_text(encoding="utf-8") + "</style>")


# ---------------------------------------------------------------- usuario
def usuario_actual() -> str:
    return (st.session_state.get("usuario_txt") or "").strip() or USUARIO_POR_DEFECTO


def cabeceras(usuario: str | None = None) -> dict:
    """Cabecera con el usuario, que la API usa para separar los datos de cada persona."""
    return {"X-Usuario": quote(usuario or usuario_actual())}


# El usuario inicial puede venir en la dirección (?usuario=Maria); se fija antes de la primera ejecución completa
if "usuario_txt" not in st.session_state:
    st.session_state["usuario_txt"] = st.query_params.get("usuario", USUARIO_POR_DEFECTO)

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
        r = CLIENTE.request(metodo, API + ruta, headers=cab or cabeceras(), timeout=300, **kwargs)
    except httpx.HTTPError:
        return None, "No se pudo conectar con la API. ¿Está corriendo uvicorn?"
    if r.status_code >= 400:
        try:
            return None, r.json().get("detail", "Ocurrió un error.")
        except ValueError:
            return None, "Ocurrió un error en el servidor."
    return r.json(), None


def llamar_cache(ruta: str, ttl: int = 30):
    """GET a la API recordando la respuesta `ttl` segundos (ajustes, estado, resumen guardado...).

    Cada pantalla repite estas consultas; así abrir un chat no espera a la API por cosas que no cambiaron.
    Se olvida todo con olvidar_cache() cuando algo cambia (subir un documento, cambiar un modelo...).
    """
    memoria = st.session_state.setdefault("cache_api", {})
    clave = (usuario_actual(), ruta)
    if clave in memoria and time.time() - memoria[clave][0] < ttl:
        return memoria[clave][1], None
    datos, error = llamar("GET", ruta)
    if not error:
        memoria[clave] = (time.time(), datos)
    return datos, error


def olvidar_cache():
    st.session_state.pop("cache_api", None)
    st.session_state.version_docs = st.session_state.get("version_docs", 0) + 1  # los PDF se vuelven a pedir


@st.cache_data(max_entries=6, show_spinner=False)
def _descargar_pdf(usuario: str, nombre: str, version: int) -> bytes:
    try:
        r = CLIENTE.get(f"{API}/documentos/{quote(nombre)}/pdf", headers=cabeceras(usuario), timeout=60)
    except httpx.HTTPError:
        raise RuntimeError("No se pudo conectar con la API. ¿Está corriendo uvicorn?") from None
    if r.status_code >= 400:
        try:
            raise RuntimeError(r.json().get("detail", "No se pudo cargar el PDF."))
        except ValueError:
            raise RuntimeError("No se pudo cargar el PDF.") from None
    return r.content


def obtener_pdf(nombre: str):
    """El PDF original (se recuerda mientras no cambien los documentos). Devuelve (bytes, error)."""
    try:
        return _descargar_pdf(usuario_actual(), nombre, st.session_state.get("version_docs", 0)), None
    except RuntimeError as e:
        return None, str(e)


def leer_stream(ruta: str, resultado: dict, cuerpo: dict | None = None, cab: dict | None = None):
    """Lee una respuesta de la API que llega por pedazos (una línea JSON cada uno).

    Va entregando el texto; lo demás (fuentes, tiempos, error...) queda en `resultado`.
    """
    try:
        with CLIENTE.stream("POST", API + ruta, json=cuerpo, headers=cab or cabeceras(),
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
        reloj.caption(f"{int(time.perf_counter() - inicio)} s")
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
        caja.info(f"{texto}... **{int(time.perf_counter() - inicio)} s**")
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
        r = CLIENTE.post(f"{API}/documentos/{quote(nombre)}/resaltar", headers=cabeceras(usuario),
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
    """Se ejecuta al pulsar 'Ver en el PDF': el visor muestra ese pasaje pintado.

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
    with (st.container(height=ALTURA, key="caja_visor") if caja else nullcontext()):
        if resaltado:
            c1, c2 = st.columns([6, 1], vertical_alignment="center")
            c1.markdown(f"**{resaltado['titulo']} — pág. {resaltado['pagina']}**")
            c2.button(":material/close:", key="quitar_resaltado", on_click=quitar_resaltado,
                      help="Quitar el resaltado")
            if resaltado["con_pasajes"] and not resaltado["encontrado"]:
                st.caption("No pude ubicar el pasaje exacto; esta es la página citada.")
            st.image(resaltado["png"], width="stretch")
            st.divider()
        for n in range(1, visibles + 1):
            giro = (giros.get(str(n), 0) + st.session_state.get(f"giro_{nombre}_{n}", 0)) % 360
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            c1.caption(f"Página {n} de {total}")
            c2.button(":material/refresh:", key=f"girar_{nombre}_{n}", on_click=girar, args=(n,),
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
    st.session_state.seleccion_pendiente = []  # un chat nuevo empieza sin archivos elegidos (pantalla de inicio)
    for clave in ("resaltar", "reproducir"):
        st.session_state.pop(clave, None)


def documentos_de_conversacion(mensajes: list[dict]) -> list[str] | None:
    """Qué documentos se usaron en una conversación guardada (para volver a marcarlos al abrirla).

    Se toma la selección de la última respuesta que la guardó; en conversaciones viejas se deduce de las
    fuentes citadas. None si no hay forma de saberlo (entonces la selección actual no se toca).
    """
    for m in reversed(mensajes):
        if m.get("documentos_usados"):
            return list(m["documentos_usados"])
    usados = {f["documento"] for m in mensajes for f in (m.get("fuentes") or []) if f.get("documento")}
    usados |= {c["documento"] for m in mensajes for c in (m.get("citas_detalle") or []) if c.get("documento")}
    return sorted(usados) or None


def marcados_ahora() -> list[str]:
    """Documentos marcados en la barra lateral en este momento."""
    return [n for n in nombres if st.session_state.get(f"buscar_{n}", False)]


def recargar_mensajes(conv_id: int):
    """Vuelve a leer la conversación guardada (cuando una tarea en segundo plano terminó de escribir en ella)."""
    mensajes, error = llamar("GET", f"/conversaciones/{conv_id}/mensajes")
    if not error:
        st.session_state.mensajes = mensajes


def abrir_conversacion(conv_id: int):
    mensajes, error = llamar("GET", f"/conversaciones/{conv_id}/mensajes")
    if not error:
        st.session_state.update(mensajes=mensajes, conv_id=conv_id, pendientes=[], miniaturas={})
        st.session_state.seleccion_pendiente = documentos_de_conversacion(mensajes)  # se aplica en la barra lateral
        for clave in ("resaltar", "reproducir"):  # nada resaltado de otro chat
            st.session_state.pop(clave, None)


def borrar_conversacion(conv_id: int):
    llamar("DELETE", f"/conversaciones/{conv_id}")
    if st.session_state.conv_id == conv_id:
        nueva_conversacion()


def cambiar_clave(numero: int):
    """Enciende o apaga una clave de Gemini según su interruptor."""
    llamar("POST", f"/gemini/claves/{numero}", json={"activa": bool(st.session_state[f"clave_gemini_{numero}"])})
    olvidar_cache()


def guardar_clave(numero: int):
    """Guarda la clave pegada en el campo y lo vacía (la clave no se vuelve a mostrar)."""
    texto = st.session_state.get(f"clave_txt_{numero}", "")
    _, error = llamar("PUT", f"/gemini/claves/{numero}/clave", json={"clave": texto})
    olvidar_cache()
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
    olvidar_cache()
    if error:
        st.session_state.mensaje_clave = (0, error, True)


NOMBRES_CHAT = {"ollama": "Local · Ollama (qwen2.5:1.5b)", "llamacpp": "Local · llama.cpp (qwen2.5:1.5b)",
                "gemini": "Nube · Gemini (API)"}


def cambiar_chat():
    """Elige quién escribe las respuestas: el modelo local o uno de la nube."""
    _, error = llamar("POST", "/chat/modelo", json={"proveedor": st.session_state.get("chat_modelo")})
    olvidar_cache()
    if error:
        st.session_state.mensaje_clave = (0, error, True)


NOMBRES_MEDIOS = {"gemma2": "EmbeddingGemma 2 (local): imágenes, audio y video",
                  "gemini": "Gemini (nube): solo imágenes"}


def cambiar_medios():
    """Elige con qué modelo se calculan los vectores de imágenes, audios y videos nuevos."""
    _, error = llamar("POST", "/embeddings/medios", json={"proveedor": st.session_state.get("medios_modelo")})
    olvidar_cache()
    if error:
        st.session_state.mensaje_clave = (0, error, True)


TIPOS_DE_ARCHIVO = ["pdf", "mp3", "wav", "m4a", "ogg", "flac", "aac", "opus", "wma", "mp2", "mpga", "aiff", "aif", "amr",
                    "mp4", "m4v", "webm", "mov", "mkv", "avi", "mpeg", "mpg", "wmv", "flv", "3gp", "ogv"]


def api_de_fondo() -> sp.Api:
    """Acceso a la API para los hilos de fondo (se arma aquí porque necesita el usuario de la sesión)."""
    return sp.Api(CLIENTE, API, cabeceras())


def asegurar_conversacion() -> int | None:
    """La conversación abierta; si no hay ninguna, se crea (así el resultado de una tarea tiene dónde guardarse)."""
    if st.session_state.conv_id is None:
        datos, _ = llamar("POST", "/conversaciones")
        if datos:
            st.session_state.conv_id = datos["id"]
    return st.session_state.conv_id


def modelo_actual() -> str:
    return (estado or {}).get("chat", "")


def procesar_subidas(archivos) -> bool:
    """Arranca en segundo plano la subida de los archivos nuevos (PDF, audio o video). True si arrancó alguna.

    Sigue aunque el usuario abra otro chat o marque archivos: el avance se ve en «Actividad».
    """
    nuevos = []
    for archivo in archivos or []:
        clave = (archivo.name, archivo.size)
        if clave in st.session_state.procesados:
            continue
        st.session_state.procesados.add(clave)
        nuevos.append((archivo.name, archivo.getvalue()))
    if not nuevos:
        return False
    conv_id = asegurar_conversacion()  # el resumen de cada archivo queda como mensaje de esta conversación
    api, modelo = api_de_fondo(), modelo_actual()
    GESTOR.iniciar(usuario_actual(), "subida", "Subiendo " + ", ".join(n for n, _ in nuevos),
                   lambda t: sp.hacer_subidas(api, t, nuevos, modelo), conv_id=conv_id, documento=nuevos[0][0])
    st.rerun()
    return True


def iniciar_generacion(tipo: str, nombre: str):
    """Resumen, puntos clave o glosario de un archivo, en segundo plano (el resultado queda en el chat)."""
    conv_id = asegurar_conversacion()
    api, modelo = api_de_fondo(), modelo_actual()
    if tipo == "resumen":
        titulo, trabajo = f"Resumiendo {nombre}", lambda t: sp.hacer_resumen(api, t, nombre, modelo)
    else:
        titulo = f"Generando {sp.ETIQUETAS[tipo]} de {nombre}"
        trabajo = lambda t: sp.hacer_generacion(api, t, tipo, nombre, modelo)  # noqa: E731
    GESTOR.iniciar(usuario_actual(), tipo, titulo, trabajo, conv_id=conv_id, documento=nombre)
    st.rerun()


def resumir_carpeta(datos: dict) -> tuple[str, str]:
    lineas = [f"{r['nombre']}: {r['mensaje']}" for r in datos["resultados"]]
    nivel = "warning" if any(not r["ok"] for r in datos["resultados"]) else "success"
    return ("\n".join(lineas) or "No había archivos nuevos."), nivel


def resumir_indexado(datos: dict) -> tuple[str, str]:
    mensaje = (f"Imágenes indexadas: {datos['indexadas']} de {datos['total']} · audios: {datos.get('audios', 0)} · "
               f"videos: {datos.get('videos', 0)}.")
    if datos.get("error"):
        return f"{mensaje} {datos['error']}", "warning"
    return mensaje, "success"


def burbujas_en_curso():
    """Lo que se está escribiendo ahora en esta conversación (se actualiza solo cada segundo mientras dure)."""
    usuario, conv = usuario_actual(), st.session_state.conv_id
    en_curso = GESTOR.corriendo(usuario, conv, TIPOS_CHAT) if conv is not None else []

    def pintar():
        for t in GESTOR.corriendo(usuario, conv, TIPOS_CHAT):
            with st.chat_message("assistant"):
                if t.tipo in ("resumen", "subida") and t.documento:
                    st.write(f"**{t.documento}** — resumen:")
                if t.texto:
                    st.write(t.texto)
                st.caption(f"{t.titulo}" + (f" · {t.avance}" if t.avance else "") + f" · {t.segundos} s")

    st.fragment(run_every=1.0 if en_curso else None)(pintar)()


def panel_actividad(sufijo: str):
    """Tareas en segundo plano: avance de lo que corre y aviso de lo que terminó (mientras tanto se puede seguir usando todo)."""
    usuario = usuario_actual()
    hay = GESTOR.hay_actividad(usuario)

    def cuerpo():
        nuevas = [t for t in GESTOR.del_usuario(usuario) if t.estado != "corriendo" and not t.visto]
        cambio = False
        for t in nuevas:  # lo que acaba de terminar: se recarga la conversación si el usuario sigue en ella
            t.visto = True
            cambio = True
            if t.tipo in ("subida", "carpeta", "indexar"):
                olvidar_cache()
            if t.conv_id is not None and t.conv_id == st.session_state.conv_id:
                recargar_mensajes(t.conv_id)
                if t.tipo == "pregunta" and t.meta.get("detalle"):
                    st.session_state.resaltar_pendiente = t.meta["detalle"][0]
                if t.tipo == "subida" and t.meta.get("subidos"):  # se abre lo que se acaba de subir
                    st.session_state.seleccion_pendiente = t.meta["subidos"]
                    st.session_state.doc_actual = t.meta["subidos"][0]
                if t.tipo in ("pregunta", "resumen", "puntos_clave", "glosario") and t.nivel != "error":
                    t.cerrada = True  # ya está escrito en el chat: no hace falta otro aviso
        if cambio:
            st.rerun()  # actualiza chats, archivos y chat abierto
        for t in GESTOR.del_usuario(usuario):
            if t.estado == "corriendo":
                st.html(f'<div class="actividad"><span class="punto"></span><div><b>{t.titulo}</b>'
                        f'<small>{t.avance + " · " if t.avance else ""}{t.segundos} s</small></div></div>')
                continue
            {"success": st.success, "warning": st.warning, "error": st.error}.get(t.nivel, st.info)(
                t.mensaje or "Listo.")
            for aviso in t.avisos:
                st.caption(aviso)
            c1, c2 = st.columns(2)
            if t.conv_id is not None and t.conv_id != st.session_state.conv_id and \
                    c1.button("Abrir chat", key=f"act_ver_{sufijo}_{t.id}", width="stretch"):
                abrir_conversacion(t.conv_id)
                st.rerun()
            if c2.button("Cerrar", key=f"act_cerrar_{sufijo}_{t.id}", width="stretch"):
                GESTOR.cerrar(t.id)
                st.rerun()

    st.fragment(run_every=1.0 if hay else None)(cuerpo)()


def titulo_de_conversacion(conv: dict) -> str:
    """Primera línea: el archivo del que se habló; segunda: la primera pregunta. Ej.: «Energia_solar.pdf»."""
    docs = conv.get("documentos") or []
    pregunta = conv["titulo"]
    if not docs:
        return pregunta
    nombre = docs[0] if len(docs[0]) <= 26 else docs[0][:24] + "…"
    extra = f" y {len(docs) - 1} más" if len(docs) > 1 else ""
    return f"**{nombre}**{extra}  \n{pregunta}"


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.html('<div class="marca"><div class="logo">B</div>Base de Conocimiento IA</div>')
    st.button("Nuevo chat", key="nuevo_chat", on_click=nueva_conversacion, width="stretch")
    c_actividad = st.container()
    c_chats = st.container()
    c_docs = st.container()
    c_ajustes = st.expander("Ajustes", icon=":material/tune:")

    with c_ajustes:
        st.text_input("Usuario", key="usuario_txt",
                      help="Cada usuario ve solo sus documentos y conversaciones. "
                           "No usa contraseña: solo separa la información de cada nombre.")

        estado, error = llamar_cache("/estado", 20)
        if error:
            st.error(error)
        else:
            st.caption(f"Modo {estado['modo'].upper()} · chat `{estado['chat']}` · embeddings `{estado['embeddings']}`")
            if estado.get("gemini"):
                st.caption("Gemini activo: se pueden buscar imágenes dentro de los PDFs.")
            else:
                st.caption("Gemini sin clave activa: revisa las claves de abajo o pega `GEMINI_API_KEY` en el `.env`.")
            datos_claves, _ = llamar_cache("/gemini/claves", 20)
            with st.expander("Claves de Gemini"):
                for c in (datos_claves or {}).get("claves", []):
                    st.toggle(f"Clave {c['numero']}: {c['texto']}",
                              value=c["activa"], disabled=not c["configurada"], key=f"clave_gemini_{c['numero']}",
                              on_change=cambiar_clave, args=(c["numero"],))
                st.caption("Se usan en orden 1, 2, 3. Si una llega a su límite o falla, pasa sola a la siguiente.")
                st.markdown("**Agregar o cambiar una clave**")
                numero_clave = st.selectbox("Clave", [1, 2, 3], key="clave_elegida", label_visibility="collapsed",
                                            format_func=lambda n: f"Clave {n}")
                st.text_input("Pega la clave", type="password", key=f"clave_txt_{numero_clave}",
                              placeholder="Pega aquí la clave de Gemini", label_visibility="collapsed")
                st.button("Guardar clave", on_click=guardar_clave, args=(numero_clave,), width="stretch")
                aviso = st.session_state.pop("mensaje_clave", None)
                if aviso:
                    (st.error if aviso[2] else st.success)(aviso[1])
                st.caption("Se guardan en la base de datos de este computador y nunca se muestran. "
                           "Para borrar una, guarda el campo vacío.")
            modelo_emb, _ = llamar_cache("/embeddings", 30)
            actual = (modelo_emb or {}).get("proveedor", "ollama")
            st.session_state["embeddings_modelo"] = actual  # siempre refleja lo guardado (p. ej. si fue rechazado)
            st.selectbox("Embeddings al subir PDFs", list(NOMBRES_EMBEDDINGS), key="embeddings_modelo",
                         format_func=NOMBRES_EMBEDDINGS.get, on_change=cambiar_embeddings,
                         help="Con qué modelo se calculan los vectores de los PDFs que subas desde ahora. "
                              "Los PDFs ya subidos conservan el suyo. Gemini (nube) es mucho más rápido, pero "
                              "el texto de los PDFs se envía a Google. La respuesta del chat siempre es local.")
            if actual == "gemini":
                st.caption("El texto de los PDFs nuevos se envía a Google. La respuesta del chat sigue siendo local.")
            chat_emb, _ = llamar_cache("/chat/modelo", 30)
            st.session_state["chat_modelo"] = (chat_emb or {}).get("proveedor", "ollama")
            st.selectbox("Quién escribe las respuestas", list(NOMBRES_CHAT), key="chat_modelo",
                         format_func=lambda k: (f"Nube · Gemini ({(chat_emb or {}).get('modelo_nube', 'API')})"
                                                if k == "gemini" else
                                                NOMBRES_CHAT[k].replace("qwen2.5:1.5b", (chat_emb or {}).get("modelo_local", ""))),
                         on_change=cambiar_chat,
                         help="Local: el modelo corre en tu computador, sin internet. Nube: Gemini por API (con tus "
                              "claves); también es el que puede VER las imágenes para describirlas.")
            if (chat_emb or {}).get("proveedor") == "gemini":
                st.caption("Tus preguntas y los fragmentos encontrados se envían a Google para escribir la respuesta.")
            medios_emb, _ = llamar_cache("/embeddings/medios", 30)
            st.session_state["medios_modelo"] = (medios_emb or {}).get("proveedor", "gemma2")
            st.selectbox("Embeddings de imágenes, audio y video", list(NOMBRES_MEDIOS), key="medios_modelo",
                         format_func=NOMBRES_MEDIOS.get, on_change=cambiar_medios,
                         help="EmbeddingGemma 2 entiende texto, imágenes, audio y video en un mismo espacio y corre "
                              "en tu computador. Gemini solo sirve para imágenes y las envía a Google.")
            if estado["ollama_activo"] is False:
                st.error("Ollama no responde. ¿Está instalado y corriendo?")
            for faltante in estado["faltan"]:
                st.warning(f"El modelo `{faltante}` no está instalado. Ejecuta: `ollama pull {faltante}`")
            if not estado["base_de_datos"]:
                st.error("La base de datos no responde. ¿Está corriendo Docker?")
        st.page_link("pages/1_Base_de_datos.py", label="Ver la base de datos", icon=":material/database:")

    with c_actividad:
        panel_actividad("lateral")

    with c_docs:
        st.html('<div class="etiqueta-seccion">Archivos</div>')
        # La clave incluye al usuario: al cambiar de usuario el cargador queda vacío
        archivos = st.file_uploader("Sube PDFs, audios o videos cortos",
                                    type=TIPOS_DE_ARCHIVO,
                                    accept_multiple_files=True,
                                    key=f"subir_{usuario_actual()}")
        procesar_subidas(archivos)

        carpeta, _ = llamar_cache("/carpeta", 20)
        if carpeta and carpeta["existe"]:
            faltan = [a for a in carpeta["archivos"] if not a["indexado"]]
            if st.button(f"Indexar carpeta de conocimiento ({len(faltan)} nuevos)", width="stretch",
                         disabled=not faltan,
                         help=f"Indexa los PDFs, audios y videos que haya en {carpeta['carpeta']}"):
                api = api_de_fondo()
                GESTOR.iniciar(usuario_actual(), "carpeta", "Indexando la carpeta de conocimiento",
                               lambda t: sp.hacer_llamada(api, t, "POST", "/carpeta/indexar", resumir_carpeta))
                st.rerun()

        docs, error = llamar("GET", "/documentos")
        nombres = [d["nombre"] for d in docs] if docs else []
        tipo_de = {d["nombre"]: d.get("tipo", "pdf") for d in docs or []}
        pendiente = st.session_state.pop("seleccion_pendiente", None)
        if pendiente is not None:  # se abrió un chat o se subió algo: se marcan solo esos archivos
            for n in nombres:
                st.session_state[f"buscar_{n}"] = n in pendiente
        if docs:
            st.caption("Marca en cuáles archivos buscar:")
            for d in docs:
                col1, col2 = st.columns([6, 1], vertical_alignment="center")
                fecha = (d.get("creado") or "")[:10]
                nube = {"gemini": " · Gemini", "gemma": " · Gemma", "gemma2": " · Gemma 2"}.get(d.get("embeddings"), "")
                icono = {"audio": "[audio] ", "video": "[video] "}.get(d.get("tipo"), "")
                col1.checkbox(f"{icono}{d['nombre']} ({d['fragmentos']}){nube}", value=False,
                              key=f"buscar_{d['nombre']}", help=f"Subido el {fecha}" if fecha else None)
                if col2.button(":material/delete:", key=f"del_{d['nombre']}", help="Eliminar"):
                    _, error = llamar("DELETE", f"/documentos/{quote(d['nombre'])}")
                    olvidar_cache()
                    if error:
                        st.error(error)
                    # Permite volver a subir el mismo archivo después de borrarlo
                    st.session_state.procesados -= {c for c in st.session_state.procesados
                                                    if c[0] == d["nombre"]}
                    st.rerun()
        elif docs is not None:
            st.caption("Aún no hay documentos.")

    with c_chats:
        st.html('<div class="etiqueta-seccion">Chats</div>')
        conversaciones, _ = llamar("GET", "/conversaciones")
        for conv in conversaciones or []:
            col1, col2 = st.columns([6, 1], vertical_alignment="center")
            activa = conv["id"] == st.session_state.conv_id
            col1.button(titulo_de_conversacion(conv), key=f"conv_{conv['id']}", on_click=abrir_conversacion,
                        args=(conv["id"],), type="primary" if activa else "secondary", width="stretch",
                        help=conv["titulo"])
            col2.button(":material/delete:", key=f"delconv_{conv['id']}", on_click=borrar_conversacion,
                        args=(conv["id"],), help="Eliminar esta conversación")


def documentos_elegidos() -> list[str] | None:
    """Documentos marcados para buscar. None = todos (no hace falta filtrar)."""
    elegidos = [n for n in nombres if st.session_state.get(f"buscar_{n}", False)]
    return None if len(elegidos) == len(nombres) else elegidos


# ------------------------------------------------------ panel del documento (izquierda)
def descargar(ruta: str, usuario: str, params: dict | None = None):
    """Función que trae el archivo a exportar. Se ejecuta solo al pulsar el botón de descarga."""
    def traer() -> bytes:
        r = CLIENTE.get(API + ruta, params=params, headers=cabeceras(usuario), timeout=120)
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
    _tipos = {tipo_de.get(n, "pdf") for n in marcados_ahora()}
    _titulo = "Archivos" if len(marcados_ahora()) > 1 and len(_tipos) > 1 else \
        {"pdf": "Documento", "audio": "Audio", "video": "Video"}[next(iter(_tipos), "pdf")]
    st.html(f'<div class="panel-titulo">{_titulo}</div>')
    if not nombres:
        st.info("Sube un PDF desde la barra lateral para verlo aquí.")
        return
    # Se muestran solo los documentos marcados en la barra lateral
    marcados = [n for n in nombres if st.session_state.get(f"buscar_{n}", False)]
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
        nombre = st.selectbox("Archivo para resumen, puntos clave y exportar", marcados, key="doc_actual")
    else:
        nombre = marcados[0]
        st.session_state.doc_actual = nombre
        st.caption(f"{nombre}")

    multiples = len(marcados) > 1
    with (st.container(height=ALTURA, key="caja_visor") if multiples else nullcontext()):  # varios: un solo scroll
        for doc in marcados:
            if multiples:
                st.markdown(f"#### {doc}")
            if tipo_de.get(doc, "pdf") != "pdf":  # audio o video suelto: se reproduce
                datos_medio = obtener_medio_suelto(usuario_actual(), doc)
                if datos_medio is None:
                    st.warning("No pude cargar ese archivo.")
                else:
                    desde = int(escuchar["inicio"]) if escuchar and escuchar["documento"] == doc else 0
                    if desde:
                        st.caption(f"Reproduciendo desde {_minutos(desde)}")
                    (st.audio if tipo_de[doc] == "audio" else st.video)(datos_medio, start_time=desde)
                continue
            pdf, error = obtener_pdf(doc)
            if error:
                st.warning(error)
                continue
            datos, _ = llamar_cache(f"/documentos/{quote(doc)}/giros", 120)
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
    if b1.button("Resumen", width="stretch"):
        iniciar_generacion("resumen", nombre)
    if b2.button("Puntos clave", width="stretch"):
        iniciar_generacion("puntos_clave", nombre)
    if b3.button("Glosario", width="stretch"):
        iniciar_generacion("glosario", nombre)

    if tipo_de.get(nombre, "pdf") == "pdf" and st.button("Indexar imágenes, audio y video", width="stretch",
                 help="Calcula los vectores de las imágenes de este PDF y de los audios y videos que traiga "
                      "adjuntos, para poder buscarlos con una frase. Con EmbeddingGemma 2 todo queda en tu "
                      "computador; con Gemini las imágenes se envían a Google."):
        api = api_de_fondo()
        GESTOR.iniciar(usuario_actual(), "indexar", f"Indexando imágenes, audio y video de {nombre}",
                       lambda t: sp.hacer_llamada(api, t, "POST", f"/documentos/{quote(nombre)}/indexar_imagenes",
                                                  resumir_indexado), documento=nombre)
        st.rerun()

    guardado, _ = llamar_cache(f"/documentos/{quote(nombre)}/resumen", 60)
    with st.expander("Exportar a PDF o Word"):
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
    vistos, unicos = set(), []  # dos fragmentos de la misma página no necesitan dos botones iguales
    for d in detalle:
        clave = (d.get("documento"), d.get("pagina"), d.get("inicio"))
        if clave not in vistos:
            vistos.add(clave)
            unicos.append(d)
    detalle = unicos
    if len(detalle) > 4:  # p. ej. el resumen: una fuente por página, en un desplegable y con botones cortos
        with st.expander(f"Ver en el PDF de dónde salió esto ({len(detalle)} páginas)"):
            columnas = st.columns(3)
            for i, d in enumerate(detalle):
                columnas[i % 3].button(f"pág. {d['pagina']}", key=f"ver_{indice}_{d['n']}",
                                       on_click=ver_en_pdf, args=(d,), width="stretch")
    else:
        for d in detalle:
            if d.get("inicio") is not None:
                etiqueta = f"Reproducir desde {_minutos(d['inicio'])}: {d['documento']}"
            else:
                etiqueta = f"Ver en el PDF: {d['documento']}, pág. {d['pagina']}"
            st.button(etiqueta, key=f"ver_{indice}_{d['n']}", on_click=ver_en_pdf, args=(d,))
    if not fuentes:
        return
    if not detalle:
        usadas = [f for f in fuentes if f.get("n") in citas]
        if usadas:
            st.caption("**Fuente:** " + " · ".join(
                f"{f['documento']}, " + (f"minuto {_minutos(f['inicio'])}" if f.get("inicio") is not None
                                         else f"pág. {f['pagina']}") for f in usadas))
    de_total = f" de {total_buscado} en los archivos buscados" if total_buscado else ""
    with st.expander(f"Fragmentos consultados: {len(fuentes)}{de_total}"):
        for f in fuentes:
            marca = " usado" if f.get("n") in citas else ""
            tipo = f.get("tipo", "pdf")
            donde = f"minuto {_minutos(f['inicio'])}" if f.get("inicio") is not None else f"pág. {f['pagina']}"
            clase = {"audio": " (audio)", "video": " (video)"}.get(tipo, "")
            st.markdown(f"**[{f.get('n', '·')}] {f['documento']}{clase} — {donde}**{marca}")
            tecnico = detalle_tecnico(f)
            if tecnico:
                st.caption(tecnico)
            if f.get("fragmento"):
                st.markdown("> " + f["fragmento"].replace("\n", " "))


def mostrar_interpretada(pregunta: str, interpretada: str | None):
    """Si la pregunta era un seguimiento, muestra cómo la entendió el sistema."""
    if interpretada and _normalizada(interpretada) != _normalizada(pregunta):
        st.caption(f"Entendí tu pregunta como: «{interpretada}»")


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
        st.caption(f"{modelo or '—'} · " + " · ".join(etapas))


def preguntar(texto: str):
    """Se ejecuta al pulsar una pregunta sugerida: la deja lista para enviarla al chat."""
    st.session_state.pregunta_pendiente = texto


def mostrar_sugerencias(indice: int, sugerencias: list[str] | None):
    """Muestra las preguntas sugeridas como botones. `indice` hace únicas sus claves."""
    if not sugerencias:
        return
    st.caption("Preguntas que puedes hacer:")
    for i, sugerencia in enumerate(sugerencias):
        st.button(sugerencia, key=f"sug_{indice}_{i}", on_click=preguntar, args=(sugerencia,))


def modelo_en_uso() -> str:
    return (estado or {}).get("chat", "—")


@st.cache_data(show_spinner=False, max_entries=100)
def obtener_miniatura(usuario: str, imagen_id: int) -> bytes | None:
    """Descarga una imagen indexada de los documentos (para mostrarla en el chat)."""
    try:
        r = CLIENTE.get(f"{API}/imagenes/{imagen_id}", headers=cabeceras(usuario), timeout=30)
    except httpx.HTTPError:
        return None
    return r.content if r.status_code == 200 else None


@st.cache_data(show_spinner=False, max_entries=20)
def obtener_archivo_de_medio(usuario: str, imagen_id: int) -> bytes | None:
    """El audio o video original al que pertenece un resultado."""
    try:
        r = CLIENTE.get(f"{API}/medios/{imagen_id}/archivo", headers=cabeceras(usuario), timeout=120)
    except httpx.HTTPError:
        return None
    return r.content if r.status_code == 200 else None


@st.cache_data(show_spinner=False, max_entries=10)
def obtener_medio_suelto(usuario: str, nombre: str) -> bytes | None:
    """El audio o video de un documento suelto, para el visor."""
    try:
        r = CLIENTE.get(f"{API}/documentos/{quote(nombre)}/archivo_medio", headers=cabeceras(usuario), timeout=120)
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
                    st.caption(f"{im.get('nombre') or im['documento']} · {desde} · {parecido}")
                    if archivo:
                        st.audio(archivo, start_time=int(im.get("inicio") or 0))
                else:
                    st.caption(f"{im.get('nombre') or im['documento']} · {desde} · {parecido}")
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
        st.caption(f"Texto que leí en la imagen: «{imagen['texto']}»")
    else:
        st.caption("No encontré letras en la imagen. Todavía no puedo describir lo que muestra: "
                   "solo leo el texto que contiene.")


def leer_imagenes(imagenes: list) -> list[tuple[str, str]]:
    """Lee el texto de cada imagen adjunta. Devuelve [(nombre, texto)]; texto vacío si no hay letras."""
    leidas = []
    for imagen in imagenes:
        try:
            r = CLIENTE.post(f"{API}/imagen/leer", headers=cabeceras(), timeout=120,
                           files={"archivo": (imagen.name, imagen.getvalue())})
            texto = r.json().get("texto", "") if r.status_code < 400 else ""
        except httpx.HTTPError:
            texto = ""
        leidas.append((imagen.name, texto))
    return leidas


def iniciar_pregunta(pregunta: str, imagenes: list | None = None):
    """Guarda la pregunta y la responde EN SEGUNDO PLANO (se puede seguir usando el sistema mientras tanto).

    Si se adjuntaron imágenes, se lee su texto y se agrega a la pregunta para buscar con él.
    """
    elegidos = documentos_elegidos()
    docs_de_esta_pregunta = marcados_ahora()  # se guarda con la respuesta para volver a marcarlos al reabrir el chat
    if elegidos == []:
        st.warning("Marca al menos un archivo en la barra lateral para buscar.")
        return
    conv_id = asegurar_conversacion()
    # Se recuerdan los mensajes anteriores para entender seguimientos como "¿y eso qué significa?"
    historial = [{"rol": m["rol"], "texto": m["texto"]} for m in st.session_state.mensajes[-6:]]

    texto_de_imagenes = ""
    nota_imagen = ""
    if imagenes:
        with st.spinner("Leyendo la imagen..."):
            leidas = leer_imagenes(imagenes)
        texto_de_imagenes = " ".join(t for _, t in leidas)[:500]
        nota_imagen = "Imagen adjunta: " + ", ".join(n for n, _ in leidas)
        pregunta = pregunta or "Busca en los documentos lo que dice esta imagen."
    indice_usuario = len(st.session_state.mensajes)
    if imagenes:  # las miniaturas se guardan solo en esta sesión (no en la base de datos)
        st.session_state.setdefault("miniaturas", {})[indice_usuario] = [i.getvalue() for i in imagenes]
    mensaje_usuario = {"rol": "user", "texto": f"{pregunta}\n\n{nota_imagen}" if nota_imagen else pregunta}
    if imagenes:
        mensaje_usuario["imagen"] = {"texto": texto_de_imagenes}
    agregar_mensaje(mensaje_usuario)  # la pregunta queda guardada de inmediato
    # La pregunta que se envía incluye lo leído en la imagen; el usuario ve su pregunta original
    pregunta_enviada = f"{pregunta} [Texto de la imagen adjunta: {texto_de_imagenes}]" \
        if texto_de_imagenes else pregunta
    api = api_de_fondo()
    GESTOR.iniciar(
        usuario_actual(), "pregunta", "Respondiendo",
        lambda t: sp.hacer_pregunta(api, t, pregunta=pregunta, pregunta_enviada=pregunta_enviada,
                                    documentos=elegidos, historial=historial,
                                    documentos_usados=docs_de_esta_pregunta),
        conv_id=conv_id)
    st.rerun()


def panel_chat():
    st.html('<div class="panel-titulo">Chat</div>')
    caja = st.container(height=ALTURA, key="caja_chat")  # caja con scroll para los mensajes (alto: ver estilo.css)
    with caja:
        espacio_vacio = st.empty()  # bloque de bienvenida: se quita en cuanto se empieza a preguntar
        en_curso = st.session_state.conv_id is not None and bool(
            GESTOR.corriendo(usuario_actual(), st.session_state.conv_id, TIPOS_CHAT))
        if not st.session_state.mensajes and not en_curso:
            with espacio_vacio.container():
                st.html('<div class="vacio-chat"><b>Pregúntale a tus archivos</b>'
                        '<p>Escribe abajo o prueba con una de estas ideas:</p></div>')
                for i, ejemplo in enumerate(("Resume el documento", "¿Cuáles son los puntos más importantes?",
                                             "¿Qué datos o cifras menciona?")):
                    st.button(ejemplo, key=f"ejemplo_{i}", on_click=preguntar, args=(ejemplo,))
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
        burbujas_en_curso()  # lo que se está escribiendo ahora (en segundo plano)

    # Mientras se responde en este chat no se admite otra pregunta aquí; en otro chat sí se puede preguntar
    respondiendo = st.session_state.conv_id is not None and bool(
        GESTOR.corriendo(usuario_actual(), st.session_state.conv_id, ("pregunta",)))
    # El chat acepta texto y también imágenes (PNG o JPG): se lee el texto que contienen
    valor = st.chat_input("Respondiendo… puedes seguir usando el sistema" if respondiendo
                          else "Escribe tu pregunta (puedes adjuntar una imagen)",
                          accept_file=True, file_type=["png", "jpg", "jpeg"], disabled=respondiendo)
    pregunta, imagenes = "", []
    if valor:
        pregunta = (valor.get("text") or "").strip()
        imagenes = valor.get("files") or []
    if not pregunta and not imagenes and not respondiendo:  # o la que el usuario eligió entre las sugeridas
        pregunta = st.session_state.pop("pregunta_pendiente", None) or ""
    if pregunta or imagenes:
        espacio_vacio.empty()
        iniciar_pregunta(pregunta, imagenes)


# ---------- Página principal ----------
def barra_superior():
    """Logo a la izquierda y, a la derecha, qué modelos están respondiendo ahora."""
    chat = (estado or {}).get("chat", "")
    emb = (estado or {}).get("embeddings", "")
    chips = (f'<span class="chip">{chat}</span><span class="chip gris">{emb}</span>' if chat else "")
    st.html(f'<div class="barra-superior"><div class="marca"><div class="logo">B</div>Base de Conocimiento IA</div>'
            f'<div class="chips">{chips}</div></div>')


def abrir_archivo(nombre: str):
    """Desde la pantalla de inicio: se elige un archivo ya subido (queda solo ese marcado)."""
    st.session_state.seleccion_pendiente = [nombre]
    st.session_state.doc_actual = nombre


def pantalla_inicio():
    """Nada elegido: título grande, tarjeta para soltar archivos y, si ya hay archivos, para abrirlos (como ChatPDF)."""
    if nombres:
        st.html('<div class="hero"><h1><span class="grad">Elige un archivo</span><br>o abre un chat</h1>'
                '<p>Marca un archivo en la barra lateral, abre una conversación anterior o sube uno nuevo.</p></div>')
    else:
        st.html('<div class="hero"><h1><span class="grad">Chatea con tus PDFs,</span><br>audios y videos</h1>'
                '<p>Sube un archivo y pregúntale lo que quieras: encuentra la respuesta, te dice de dónde salió '
                'y reproduce el audio o video desde ese minuto.</p></div>')
    if GESTOR.hay_actividad(usuario_actual()):
        with st.container(key="tarjeta_actividad"):
            panel_actividad("inicio")
    with st.container(key="tarjeta_subida", border=True):
        st.html('<div class="tarjeta-titulo">Suelta aquí tu archivo</div>')
        archivos = st.file_uploader("Sube PDFs, audios o videos cortos", type=TIPOS_DE_ARCHIVO,
                                    accept_multiple_files=True, label_visibility="collapsed",
                                    key=f"subir_inicio_{usuario_actual()}")
        if procesar_subidas(archivos):
            st.rerun()
        st.html('<div class="formatos"><span class="chip">PDF</span><span class="chip">Imágenes</span>'
                '<span class="chip">Audio</span><span class="chip">Video</span></div>'
                '<div class="privacidad">Se procesa en tu computador. Gemini (nube) solo si tú lo activas.</div>')
    if nombres:
        with st.container(key="tarjeta_archivos"):
            st.html('<div class="tarjeta-titulo">Tus archivos</div>')
            for n in nombres:
                tipo = tipo_de.get(n, "pdf")
                etiqueta = n + {"audio": "  ·  audio", "video": "  ·  video"}.get(tipo, "")
                st.button(etiqueta, key=f"abrir_{n}", on_click=abrir_archivo, args=(n,), width="stretch")
    st.html('<div class="pasos"><div class="paso"><div class="n">1</div><b>Sube</b><span>PDFs, audios o videos cortos, '
            'o déjalos en la carpeta de conocimiento.</span></div>'
            '<div class="paso"><div class="n">2</div><b>Pregunta</b><span>Busca por significado, no por palabras '
            'exactas. También puedes preguntar por las imágenes.</span></div>'
            '<div class="paso"><div class="n">3</div><b>Verifica</b><span>Cada respuesta cita su fuente: página '
            'resaltada o minuto del audio.</span></div></div>')


barra_superior()
# Sin archivos elegidos ni conversación abierta se ve la pantalla de inicio; aparece el visor y el chat al subir
# un archivo, elegirlo o abrir una conversación
_hay_trabajo = bool(marcados_ahora()) or bool(st.session_state.mensajes) \
    or bool(st.session_state.get("resaltar") or st.session_state.get("reproducir"))
if not nombres or not _hay_trabajo:
    pantalla_inicio()
else:
    with st.container(key="layout_principal"):  # en pantallas angostas las dos columnas pasan a una sola (estilo.css)
        col_doc, col_chat = st.columns(2, gap="large")
        with col_doc:
            panel_documento()
        with col_chat:
            panel_chat()

# Si se creó una conversación nueva, se recarga para que aparezca en la barra lateral
if st.session_state.pop("refrescar", False):
    st.rerun()
