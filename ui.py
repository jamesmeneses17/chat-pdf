"""Interfaz Streamlit: documento (izquierda) y chat (derecha) al mismo tiempo."""
import hashlib
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

st.set_page_config(page_title="Chat con PDFs", page_icon="📄", layout="wide")


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
    st.session_state.update(mensajes=[], conv_id=None, pendientes=[], procesados=set(),
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
    """Se ejecuta al pulsar '📍 Ver en el PDF': el visor muestra ese pasaje pintado."""
    st.session_state.resaltar = detalle
    st.session_state.doc_actual = detalle["documento"]


def quitar_resaltado():
    st.session_state.pop("resaltar", None)


def visor_pdf(nombre: str, pdf: bytes, giros: dict, resaltado: dict | None = None):
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

    with st.container(height=ALTURA):  # caja con scroll para recorrer todo el PDF
        if resaltado:
            c1, c2 = st.columns([6, 1], vertical_alignment="center")
            c1.markdown(f"📍 **Fuente de la respuesta — pág. {resaltado['pagina']}**")
            c2.button("✖", key="quitar_resaltado", on_click=quitar_resaltado,
                      help="Quitar el resaltado")
            if not resaltado["encontrado"]:
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
    st.session_state.update(mensajes=[], conv_id=None, pendientes=[])


def abrir_conversacion(conv_id: int):
    mensajes, error = llamar("GET", f"/conversaciones/{conv_id}/mensajes")
    if not error:
        st.session_state.update(mensajes=mensajes, conv_id=conv_id, pendientes=[])


def borrar_conversacion(conv_id: int):
    llamar("DELETE", f"/conversaciones/{conv_id}")
    if st.session_state.conv_id == conv_id:
        nueva_conversacion()


# ---------------------------------------------------------------- barra lateral
with st.sidebar:
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
        if estado["ollama_activo"] is False:
            st.error("Ollama no responde. ¿Está instalado y corriendo?")
        for faltante in estado["faltan"]:
            st.warning(f"El modelo `{faltante}` no está instalado. Ejecuta: `ollama pull {faltante}`")
        if not estado["base_de_datos"]:
            st.error("La base de datos no responde. ¿Está corriendo Docker?")

    st.divider()
    st.subheader("📄 Documentos")
    # La clave incluye al usuario: al cambiar de usuario el cargador queda vacío
    archivos = st.file_uploader("Sube uno o más PDFs", type="pdf", accept_multiple_files=True,
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

    docs, error = llamar("GET", "/documentos")
    nombres = [d["nombre"] for d in docs] if docs else []
    if docs:
        st.caption("Marca los documentos en los que quieres buscar:")
        for d in docs:
            col1, col2 = st.columns([6, 1], vertical_alignment="center")
            fecha = (d.get("creado") or "")[:10]
            col1.checkbox(f"{d['nombre']} ({d['fragmentos']})", value=True,
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
    if st.session_state.get("doc_actual") not in nombres:
        st.session_state.doc_actual = nombres[0]
    nombre = st.selectbox("Documento", nombres, key="doc_actual", label_visibility="collapsed")

    pdf, error = obtener_pdf(nombre)
    if error:
        st.warning(error)
    else:
        datos, _ = llamar("GET", f"/documentos/{quote(nombre)}/giros")
        resaltado = None
        pedido = st.session_state.get("resaltar")
        if pedido and pedido["documento"] == nombre:
            giro_extra = st.session_state.get(f"giro_{nombre}_{pedido['pagina']}", 0)
            try:
                with st.spinner("Marcando la fuente en el PDF..."):
                    png, encontrado = imagen_resaltada(usuario_actual(), nombre, pedido["pagina"],
                                                       tuple(pedido["pasajes"]), giro_extra)
                resaltado = {"pagina": pedido["pagina"], "png": png, "encontrado": encontrado}
            except RuntimeError as e:
                st.warning(str(e))
        visor_pdf(nombre, pdf, datos["giros"] if datos else {}, resaltado)

    # El resultado de estos botones se escribe en el chat, no aquí
    b1, b2, b3 = st.columns(3)
    if b1.button("📝 Resumen", width="stretch"):
        st.session_state.pendientes.append(("resumen", nombre))
    if b2.button("🔑 Puntos clave", width="stretch"):
        st.session_state.pendientes.append(("puntos_clave", nombre))
    if b3.button("📚 Glosario", width="stretch"):
        st.session_state.pendientes.append(("glosario", nombre))

    guardado, _ = llamar("GET", f"/documentos/{quote(nombre)}/resumen")
    with st.expander("⬇️ Exportar a PDF o Word"):
        botones_exportar(nombre, bool(guardado and any(guardado.values())))


# ------------------------------------------------------------ panel del chat (derecha)
def _normalizada(texto: str) -> str:
    sin = "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                  if unicodedata.category(c) != "Mn")
    return " ".join(re.findall(r"\w+", sin))


def mostrar_citas(fuentes: list[dict] | None, citas: list[int] | None,
                  detalle: list[dict] | None = None, indice: int = 0):
    """De dónde salió la respuesta: botones para verla pintada en el PDF y los fragmentos."""
    if not fuentes:
        return
    citas = citas or []
    for d in detalle or []:
        st.button(f"📍 Ver en el PDF: {d['documento']}, pág. {d['pagina']}",
                  key=f"ver_{indice}_{d['n']}", on_click=ver_en_pdf, args=(d,))
    if not detalle:
        usadas = [f for f in fuentes if f.get("n") in citas]
        if usadas:
            st.caption("📎 **Fuente:** " + " · ".join(
                f"{f['documento']}, pág. {f['pagina']}" for f in usadas))
    with st.expander(f"Fragmentos consultados ({len(fuentes)})"):
        for f in fuentes:
            marca = " ✅ usado" if f.get("n") in citas else ""
            st.markdown(f"**[{f.get('n', '·')}] {f['documento']} — pág. {f['pagina']}**{marca}")
            if f.get("fragmento"):
                st.caption(f["fragmento"] + "…")


def mostrar_interpretada(pregunta: str, interpretada: str | None):
    """Si la pregunta era un seguimiento, muestra cómo la entendió el sistema."""
    if interpretada and _normalizada(interpretada) != _normalizada(pregunta):
        st.caption(f"🔎 Entendí tu pregunta como: «{interpretada}»")


def mostrar_tiempos(t, modelo: str | None = None):
    """Línea pequeña con el modelo usado y los tiempos de cada etapa (pruebas de velocidad)."""
    if t:
        reformular = f"Reformular {t['reformular_ms']} ms · " if t.get("reformular_ms") else ""
        st.caption(
            f"🧠 {modelo or '—'} · ⏱ {reformular}Embedding {t.get('embedding_ms', 0)} ms · "
            f"Búsqueda BD {t.get('busqueda_db_ms', 0)} ms · "
            f"Ollama: 1ª palabra {t.get('ollama_primer_token_ms', 0)} ms, "
            f"total {t.get('ollama_ms', 0)} ms · "
            f"**Total {t.get('total_ms', 0)} ms**"
        )


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


def resumir_en_chat(nombre: str):
    """Escribe el resumen del PDF dentro de la conversación, palabra por palabra."""
    sugerencias = []
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
            # Después del resumen, 3 preguntas sugeridas que se envían con un clic
            cab = cabeceras()
            (datos, _), _ = con_contador(
                "Preparando preguntas sugeridas",
                lambda: llamar("POST", f"/documentos/{quote(nombre)}/preguntas", cab=cab),
            )
            sugerencias = datos["preguntas"] if datos else []
            mostrar_sugerencias(len(st.session_state.mensajes), sugerencias)
    agregar_mensaje({"rol": "assistant", "texto": f"{encabezado}\n\n{texto}",
                     "sugerencias": sugerencias})


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


def responder(pregunta: str):
    """Envía la pregunta a la API (con la memoria de la conversación) y muestra la respuesta."""
    elegidos = documentos_elegidos()
    if elegidos == []:
        st.warning("Marca al menos un documento en la barra lateral para buscar.")
        return
    # Se recuerdan los mensajes anteriores para entender seguimientos como "¿y eso qué significa?"
    historial = [{"rol": m["rol"], "texto": m["texto"]} for m in st.session_state.mensajes[-6:]]
    agregar_mensaje({"rol": "user", "texto": pregunta})
    with st.chat_message("user"):
        st.write(pregunta)
    with st.chat_message("assistant"):
        resultado = {"fuentes": []}
        cuerpo = {"pregunta": pregunta, "documentos": elegidos, "historial": historial}
        with st.spinner("Pensando..."):
            # st.write_stream muestra el texto a medida que llega
            texto = st.write_stream(leer_stream("/chat_stream", resultado, cuerpo))
        if resultado.get("error"):
            st.error(resultado["error"])
        else:
            mostrar_interpretada(pregunta, resultado.get("pregunta_interpretada"))
            mostrar_citas(resultado["fuentes"], resultado.get("citas"),
                          resultado.get("citas_detalle"), len(st.session_state.mensajes))
            mostrar_tiempos(resultado.get("tiempos"), resultado.get("modelo"))
            if resultado.get("citas_detalle"):
                # El visor de la izquierda se abre solo en el pasaje de donde salió la respuesta
                st.session_state.resaltar_pendiente = resultado["citas_detalle"][0]
                st.session_state.refrescar = True
            agregar_mensaje({
                "rol": "assistant", "texto": texto, "fuentes": resultado["fuentes"],
                "citas": resultado.get("citas"), "citas_detalle": resultado.get("citas_detalle"),
                "tiempos": resultado.get("tiempos"),
                "modelo": resultado.get("modelo"),
                "interpretada": resultado.get("pregunta_interpretada"),
                "pregunta": pregunta,
            })


def panel_chat():
    st.subheader("💬 Chat")
    caja = st.container(height=ALTURA)  # caja con scroll para los mensajes
    with caja:
        for indice, m in enumerate(st.session_state.mensajes):
            with st.chat_message(m["rol"]):
                st.write(m["texto"])
                mostrar_interpretada(m.get("pregunta", ""), m.get("interpretada"))
                mostrar_citas(m.get("fuentes"), m.get("citas"), m.get("citas_detalle"), indice)
                mostrar_tiempos(m.get("tiempos"), m.get("modelo"))
                mostrar_sugerencias(indice, m.get("sugerencias"))

        # Resumen, puntos clave y glosario: se escriben aquí, en la conversación
        while st.session_state.pendientes:
            tipo, nombre = st.session_state.pendientes.pop(0)
            if tipo == "resumen":
                resumir_en_chat(nombre)
            else:
                generar_en_chat(tipo, nombre)

    pregunta = st.chat_input("Escribe tu pregunta sobre los documentos")
    if not pregunta:  # o la que el usuario eligió entre las sugeridas
        pregunta = st.session_state.pop("pregunta_pendiente", None)
    if pregunta:
        with caja:
            responder(pregunta)


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
