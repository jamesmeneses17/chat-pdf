"""API FastAPI: subir PDFs, listarlos, borrarlos, chatear, historial y exportar.

Cada petición lleva el nombre del usuario en la cabecera X-Usuario (codificado con %XX).
No es un inicio de sesión con contraseña: solo separa los documentos y conversaciones de
cada nombre. Pensado para uso local.
"""
import json
import logging
import re
import time
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Annotated
from urllib.parse import quote, unquote

from fastapi import Depends, FastAPI, File, Header, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import (busqueda_imagenes, config, db, embeddings, exportar, gemini, llamacpp, llm, medios,
                 vista_db)
from app.citas import fragmentos_citados
from app.imagen_chat import MAX_BYTES as MAX_BYTES_IMAGEN
from app.imagen_chat import leer_texto_de_imagen
from app.imagenes import (analizar_imagenes, es_pregunta_de_imagenes, pagina_pedida, pide_audio_o_video,
                          pide_contenido_de_imagen, pide_descripcion_de_imagen)
from app.imagenes import describir as describir_imagenes
from app.ingest import IngestaError, ingerir_pdf
from app.llm import (generar_respuesta, generar_respuesta_stream, info_modelos,
                     nombre_modelo_chat)
from app.memoria import (parece_seguimiento, recortar_historial, reformular_pregunta,
                         texto_historial)
from app.resaltar import pasajes_de_respuesta, resaltar_pagina
from app.resumen import (cifras_sin_respaldo, generar_glosario, generar_preguntas,
                         generar_puntos_clave, generar_resumen, generar_resumen_stream,
                         muestra_del_documento)
from app.search import buscar_similares


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Al iniciar la API se crean las tablas si no existen (y se migran las antiguas)
    db.crear_tablas()
    yield
    llamacpp.detener()  # si esta aplicación arrancó llama-server, lo cierra y libera la memoria


app = FastAPI(title="Base de Conocimiento IA", lifespan=lifespan)
log = logging.getLogger("uvicorn.error")  # así los mensajes salen en la consola de uvicorn


def usuario_actual(x_usuario: str | None = Header(default=None)) -> str:
    """Nombre del usuario de la petición (limpio); si no viene, 'invitado'."""
    nombre = re.sub(r"[^\w .\-]", "", unquote(x_usuario or "")).strip()[:40]
    nombre = nombre or db.USUARIO_POR_DEFECTO
    db.asegurar_usuario(nombre)
    return nombre


Usuario = Annotated[str, Depends(usuario_actual)]


class Mensaje(BaseModel):
    rol: str
    texto: str


class PreguntaChat(BaseModel):
    pregunta: str
    documentos: list[str] | None = None  # None = buscar en todos los del usuario
    historial: list[Mensaje] = []  # mensajes anteriores de la conversación


class MensajeNuevo(BaseModel):
    rol: str
    texto: str
    extra: dict = {}


def minutos(segundos) -> str:
    s = int(segundos or 0)
    return f"{s // 60}:{s % 60:02d}"


def donde_esta(f: dict) -> str:
    """'página 3' para un PDF, 'minuto 2:35' para un audio o video."""
    return f"minuto {minutos(f['inicio'])}" if f.get("inicio") is not None else f"página {f['pagina']}"


def construir_prompt(pregunta: str, fragmentos: list[dict], historial: list[dict] | None = None) -> str:
    """Arma el prompt: instrucciones + historial + contexto numerado + pregunta."""
    contexto = "\n\n".join(
        f"[{f['n']}] ({f['documento']}, {donde_esta(f)})\n{f['contenido']}" for f in fragmentos
    )
    bloque_historial = ""
    if historial:
        bloque_historial = ("HISTORIAL DE LA CONVERSACIÓN (solo para entender la pregunta):\n"
                            f"{texto_historial(historial)}\n\n")
    return (
        "Responde en español usando SOLO el contexto dado.\n"
        "- Si el contexto contiene información relacionada con la pregunta, respóndela con "
        "esos datos, aunque sea parcial.\n"
        "- Solo si el contexto no trata del tema, responde exactamente: "
        '"No encontré esa información en los documentos".\n'
        "- Cuando uses información de un fragmento, cita su número entre corchetes al final de "
        "la frase, por ejemplo [1] o [2].\n"
        "- NO uses como dato los números de las preguntas, de las páginas ni de las casillas "
        "(por ejemplo, '32.' al inicio de una pregunta no es una edad).\n"
        "- Los formularios escaneados traen un bloque 'RESPUESTAS MARCADAS EN LA PÁGINA N' con las "
        "opciones que la persona eligió (detección aproximada); también hay opciones con "
        "[MARCADA] en el texto. Si se pregunta por un dato de la persona (edad, sexo, nivel "
        "educativo, vivienda...), responde con esas opciones marcadas e indica la página "
        "(pág. N); si hay varias páginas o personas, di cuál es de cada una.\n"
        "- Los nombres escritos a mano casi no se leen: si preguntan por una persona por su nombre "
        "y no puedes confirmarlo en el texto, NO digas que no encontraste nada; da las respuestas "
        "marcadas de cada página y aclara que no puedes confirmar a quién corresponde cada una.\n"
        "- Si se pregunta qué opciones o rangos existen, enumera todas las opciones del texto, "
        "estén marcadas o no.\n"
        "- Si piden un dato de la persona y no hay ninguna opción marcada de ese tema, responde "
        'exactamente: "No encontré esa información en los documentos".\n'
        "- El contexto viene de PDFs y puede tener errores o símbolos sueltos de escaneo: "
        "ignóralos.\n"
        "- Sé breve y directo: responde en pocas líneas.\n\n"
        f"{bloque_historial}"
        f"CONTEXTO:\n{contexto}\n\n"
        f"PREGUNTA: {pregunta}"
    )


def preparar_consulta(datos: PreguntaChat, usuario: str, tiempos: dict):
    """Entiende la pregunta (con la memoria), busca los fragmentos y arma el prompt.

    Devuelve (pregunta_interpretada, fragmentos numerados, prompt).
    """
    historial = recortar_historial([m.model_dump() for m in datos.historial])
    interpretada = datos.pregunta
    if historial and parece_seguimiento(datos.pregunta):
        t = time.perf_counter()
        interpretada = reformular_pregunta(datos.pregunta, historial)
        tiempos["reformular_ms"] = round((time.perf_counter() - t) * 1000)

    fragmentos = buscar_similares(interpretada, usuario, datos.documentos, tiempos)
    for i, f in enumerate(fragmentos, 1):
        f["n"] = i  # número con el que el modelo cita cada fragmento
    prompt = construir_prompt(interpretada, fragmentos, historial) if fragmentos else ""
    return interpretada, fragmentos, prompt


def modelo_de_respuesta_con_imagenes(pregunta: str) -> str:
    """Qué hizo la respuesta: describir una imagen usa un modelo con visión; lo demás es leer / buscar."""
    if pide_descripcion_de_imagen(pregunta):
        return f"{config.GEMINI_CHAT_MODEL} (nube, visión)"
    return "lectura del PDF (sin IA)"


def respuesta_viendo_imagenes(datos: PreguntaChat, usuario: str, nombres: list[str]):
    """Contesta una pregunta sobre lo que MUESTRA una imagen, usando un modelo con visión (Gemini, en la nube).

    Elige las imágenes de la página que se nombra o, si no se nombra ninguna, las más parecidas a la pregunta.
    """
    pagina = pagina_pedida(datos.pregunta)
    imagenes = db.imagenes_de_documentos(usuario, nombres, pagina, 3)
    if not pagina and len(imagenes) > 1:  # sin página: las más parecidas a lo que se pregunta
        try:
            coinciden, candidatas = busqueda_imagenes.buscar(usuario, datos.pregunta, nombres, ("imagen",))
            ids = [c["id"] for c in (coinciden or candidatas)[:2]]
            imagenes = [i for i in db.imagenes_de_documentos(usuario, nombres, None, 50) if i["id"] in ids] or imagenes
        except RuntimeError:
            pass
    if not imagenes:
        return ("No tengo imágenes indexadas" + (f" en la página {pagina}" if pagina else "") +
                " de los documentos elegidos. Usa el botón «🖼️ Indexar imágenes, audio y video» bajo el visor.", [], [])
    if not gemini.disponible():
        return ("Para decir qué muestra una imagen hace falta un modelo con visión, y no hay ninguna clave de "
                "Gemini disponible. " + gemini.mensaje_sin_claves(), [], [])
    # El texto cercano ayuda al modelo a entender la figura (pie de foto, explicación)
    contexto = ""
    try:
        cercanos = [f for f in buscar_similares(datos.pregunta, usuario, nombres)
                    if not pagina or f["pagina"] == pagina][:2]
        contexto = "\n".join(f"- {f['contenido']}" for f in cercanos)
    except RuntimeError:
        pass
    prompt = ("Responde en español, de forma breve, usando lo que ves en la(s) imagen(es) adjunta(s)"
              + (f" y este texto del documento:\n{contexto}\n" if contexto else ".") +
              f"\n\nPregunta: {datos.pregunta}")
    try:
        respuesta = gemini.generar(prompt, [i["png"] for i in imagenes], maximo=600)
    except RuntimeError as e:
        return f"No pude consultar el modelo con visión: {e}", [], []
    origen = ", ".join(sorted({f"{i['documento']} pág. {i['pagina']}" for i in imagenes}))
    texto = (f"{respuesta}\n\n*Imágenes vistas por {config.GEMINI_CHAT_MODEL} (nube): {origen}. "
             "Se enviaron a Google para poder verlas.*")
    avisos = gemini.pop_avisos()
    if avisos:
        texto += "\n\n⚠️ " + " ".join(avisos)
    detalle = [{"n": n, "documento": i["documento"], "pagina": i["pagina"], "pasajes": [],
                "titulo": "Imagen analizada"} for n, i in enumerate(imagenes, 1)]
    mostradas = [{"id": i["id"], "documento": i["documento"], "pagina": i["pagina"], "similitud": 1.0,
                  "tipo": "imagen"} for i in imagenes]
    return texto, detalle, mostradas


def respuesta_sobre_imagenes(datos: PreguntaChat, usuario: str) -> tuple[str, list[dict], list[dict]]:
    """Responde sobre las imágenes del PDF. Cuenta leyendo el archivo (sin IA) y, si preguntan qué
    muestran ("¿hay un perro?"), busca las más parecidas con Gemini cuando hay clave.

    Devuelve (texto, páginas para señalar en el visor, imágenes parecidas encontradas).
    """
    nombres = datos.documentos if datos.documentos is not None else [
        d["nombre"] for d in db.listar_documentos(usuario)]
    if not nombres:
        return "No hay documentos cargados todavía.", [], []
    if pide_descripcion_de_imagen(datos.pregunta):
        return respuesta_viendo_imagenes(datos, usuario, nombres)
    partes, detalle = [], []
    sueltos = {d["nombre"] for d in db.listar_documentos(usuario) if d["tipo"] != "pdf"}  # audios y videos sueltos
    solo_audio_video = pide_audio_o_video(datos.pregunta) and not re.search(
        r"im[aá]gen|foto|figura|ilustraci|gr[aá]fico", datos.pregunta, re.I)
    for nombre in nombres:
        if nombre in sueltos or solo_audio_video:
            continue  # no son PDFs con imágenes que contar
        pdf = db.obtener_pdf(usuario, nombre)
        if pdf is None:
            partes.append(f"**{nombre}**: no tiene el original guardado (bórralo y vuelve a subirlo).")
            continue
        analisis = analizar_imagenes(pdf)
        partes.append(describir_imagenes(nombre, analisis))
        for pagina, d in analisis["por_pagina"].items():
            if d["figuras"] and len(detalle) < 6:  # botones "📍 Ver en el PDF" para las primeras páginas
                detalle.append({"n": len(detalle) + 1, "documento": nombre, "pagina": pagina,
                                "pasajes": [], "titulo": "Página con imágenes"})
    texto = "\n\n".join(partes)
    encontradas: list[dict] = []
    if pide_contenido_de_imagen(datos.pregunta):
        tema = busqueda_imagenes.tema_de_la_pregunta(datos.pregunta)
        tipos = busqueda_imagenes.tipos_pedidos(datos.pregunta)
        que = " / ".join({"imagen": "imágenes", "audio": "audios", "video": "videos"}[t] for t in tipos)
        try:
            encontradas, candidatas = busqueda_imagenes.buscar(usuario, tema, datos.documentos, tipos)
        except RuntimeError as e:
            candidatas, encontradas = None, []
            texto += f"\n\nNo pude buscar en {que}: {e}"
        if candidatas is not None:
            if not candidatas:
                texto += (f"\n\nNo hay {que} indexados para buscar. Sube el archivo de nuevo o usa el botón "
                          "«🖼️ Indexar imágenes, audio y video» bajo el visor.")
            elif not encontradas:
                mejor = candidatas[0]
                donde = f"pág. {mejor['pagina']} de {mejor['documento']}" if mejor["pagina"] else mejor["documento"]
                minimo = config.UMBRALES.get(mejor["proveedor"], {}).get(mejor["tipo"], (0, 0))[0]
                texto += (f"\n\n**No encontré nada ({que}) que se parezca a «{tema}»** con suficiente "
                          f"seguridad. Lo más cercano fue {mejor['tipo']} de {donde}, con "
                          f"{mejor['similitud'] * 100:.0f} % de parecido (el mínimo para aceptarlo es "
                          f"{minimo * 100:.1f} %).")
            else:
                texto += (f"\n\nEncontré {len(encontradas)} resultado(s) que se parecen a «{tema}». El parecido "
                          "es una estimación, confírmalo viéndolo o escuchándolo. Para preguntar qué dicen, "
                          "hazlo sin pedir «un audio…» (por ejemplo: «¿qué dice el audio sobre perros?»).")
                detalle = [{"n": i + 1, "documento": e["documento"], "pagina": e["pagina"],
                            "pasajes": [], "titulo": f"Imagen parecida a «{tema}»"}
                           for i, e in enumerate(encontradas) if e["tipo"] == "imagen" and e["pagina"]]
    avisos = gemini.pop_avisos()  # si hubo que cambiar de clave o una llegó a su límite
    if avisos:
        texto += "\n\n⚠️ " + " ".join(avisos)
    return texto, detalle, encontradas


def sin_etiqueta_marcada(trozos):
    """Quita '[MARCADA]' de la respuesta mientras llega por pedazos.

    La etiqueta es una ayuda interna para el modelo; a veces la repite en su respuesta.
    Como puede llegar partida en varios pedazos ('[MAR', 'CADA]'), se retiene el final
    mientras pueda ser el comienzo de la etiqueta.
    """
    etiqueta, pendiente = "[MARCADA]", ""
    for trozo in trozos:
        pendiente = (pendiente + trozo).replace(etiqueta, "")
        # cuántos caracteres del final podrían ser el comienzo de la etiqueta (nunca más que los que hay)
        retener = next((k for k in range(min(len(etiqueta) - 1, len(pendiente)), 0, -1)
                        if etiqueta.startswith(pendiente[-k:])), 0)
        salida, pendiente = pendiente[:len(pendiente) - retener], pendiente[len(pendiente) - retener:]
        if salida:
            yield salida
    if pendiente:
        yield pendiente


def detalle_de_citas(respuesta: str, pregunta: str, fragmentos: list[dict],
                     citas: list[int]) -> list[dict]:
    """Por cada fragmento citado: documento, página y los pasajes exactos que hay que resaltar."""
    return [{"n": f["n"], "documento": f["documento"], "pagina": f["pagina"], "inicio": f.get("inicio"),
             "pasajes": pasajes_de_respuesta(respuesta, pregunta, f["contenido"])}
            for f in fragmentos if f["n"] in citas]


def lista_de_fuentes(fragmentos: list[dict]) -> list[dict]:
    """Fuentes para mostrar: número, documento, página, texto completo y por qué se eligió."""
    return [{
        "n": f["n"], "documento": f["documento"], "pagina": f["pagina"], "inicio": f.get("inicio"),
        "fragmento": f["contenido"],  # el texto completo del fragmento
        "id": f.get("id"), "posicion": f.get("posicion"), "total_documento": f.get("total_documento"),
        "similitud": round(f.get("similitud", 0), 3),  # parecido por significado (vectores)
        "coincidencia": round(f.get("coincidencia", 0), 3),  # palabras de la pregunta que contiene
        "puntaje": round(f.get("puntaje", 0), 3), "hallado_por": f.get("hallado_por"),
    } for f in fragmentos]


# ---------------------------------------------------------------- documentos
@app.post("/subir")
async def subir(usuario: Usuario, archivo: UploadFile = File(...)):
    contenido = await archivo.read()
    return procesar_subida(usuario, archivo.filename, contenido)


def procesar_subida(usuario: str, nombre: str, contenido: bytes) -> dict:
    """Indexa un archivo (PDF, audio o video). Lo usan la subida manual y la carpeta de conocimiento."""
    inicio = time.perf_counter()
    tipo_medio = medios.tipo_de_archivo(nombre)
    if tipo_medio:  # un audio o video suelto: se indexa por segmentos con EmbeddingGemma 2
        try:
            r = busqueda_imagenes.indexar_archivo_suelto(usuario, nombre, contenido)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"No se pudo indexar el {tipo_medio}: {e}")
        mensaje = (f"'{nombre}' indexado: {r['segmentos']} segmento(s) de {tipo_medio} (EmbeddingGemma 2); "
                   f"se transcribieron {r['palabras']} palabras en {r['fragmentos']} fragmento(s).")
        avisos = [r["aviso_embeddings"]] if r.get("aviso_embeddings") else []
        if not r["fragmentos"]:
            avisos.append("No se oyó ninguna voz en el archivo: se puede buscar por su sonido o imagen, "
                          "pero no hay texto para conversar con él.")
        return {"mensaje": mensaje, "advertencias": avisos}
    try:
        r = ingerir_pdf(usuario, nombre, contenido)
    except IngestaError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:  # errores de Ollama
        raise HTTPException(status_code=503, detail=str(e))

    mensaje = f"'{nombre}' indexado: {r['fragmentos']} fragmentos (embeddings: {r['proveedor']})."
    advertencias = []
    if r.get("aviso_embeddings"):
        advertencias.append(r["aviso_embeddings"])
    if r["ocr"]:
        mensaje += f" Se usó OCR en {len(r['ocr'])} página(s)."
    if r.get("marcadas"):
        mensaje += f" Se detectaron {r['marcadas']} casillas marcadas (detección aproximada)."
    if True:  # imágenes (y audios / videos adjuntos) del PDF: se indexan para poder buscarlos con una frase
        try:
            im = busqueda_imagenes.indexar(usuario, nombre, contenido)
        except Exception as e:
            im = {"indexadas": 0, "total": 0, "audios": 0, "videos": 0, "proveedor": "?",
                  "error": f"No se pudieron indexar las imágenes ({e})."}
        modelo = {"gemma2": "EmbeddingGemma 2", "gemini": "Gemini"}.get(im["proveedor"], im["proveedor"])
        if im["indexadas"]:
            mensaje += f" Se indexaron {im['indexadas']} imágenes con {modelo}."
        if im.get("audios") or im.get("videos"):
            mensaje += f" Audios adjuntos: {im['audios']}, videos adjuntos: {im['videos']}."
        if im["error"]:
            advertencias.append(f"Imágenes: {im['error']} (indexadas {im['indexadas']} de {im['total']}).")
        advertencias += [f"Gemini: {a}" for a in gemini.pop_avisos()]  # cambios de clave, límites...
    if r["baja_calidad"]:
        paginas = ", ".join(str(p) for p in r["baja_calidad"])
        advertencias.append(
            f"El texto de la(s) página(s) {paginas} salió poco legible (escaneo borroso o "
            "con mucho ruido). Las respuestas sobre esas páginas pueden ser imprecisas."
        )
    log.info("SUBIDA '%s' (usuario %s): %d fragmentos, OCR en %d pág., en %d ms",
             nombre, usuario, r["fragmentos"], len(r["ocr"]),
             round((time.perf_counter() - inicio) * 1000))
    return {"mensaje": mensaje, "advertencias": advertencias}


@app.get("/documentos")
def documentos(usuario: Usuario):
    return db.listar_documentos(usuario)


@app.get("/documentos/{nombre}/archivo_medio")
def ver_medio_suelto(nombre: str, usuario: Usuario):
    """El audio o video de un documento suelto, para reproducirlo en el visor."""
    medio = db.obtener_adjunto_de_documento(usuario, nombre)
    if medio is None:
        raise HTTPException(status_code=404, detail="Ese documento no es un audio ni un video.")
    return Response(content=medio[2], media_type=medio[1])


# ------------------------------------------------ carpeta de conocimiento (PDFs, audios y videos)
def _carpeta() -> Path:
    ruta = Path(config.CARPETA_CONOCIMIENTO)
    return ruta if ruta.is_absolute() else Path(__file__).resolve().parent.parent / ruta


def _archivos_de_la_carpeta() -> list[Path]:
    carpeta = _carpeta()
    if not carpeta.is_dir():
        return []
    return sorted(p for p in carpeta.iterdir()
                  if p.is_file() and (p.suffix.lower() == ".pdf" or medios.tipo_de_archivo(p.name)))


@app.get("/carpeta")
def estado_de_la_carpeta(usuario: Usuario):
    """Qué archivos hay en la carpeta de conocimiento y cuáles faltan por indexar."""
    ya = {d["nombre"] for d in db.listar_documentos(usuario)}
    archivos = [{"nombre": p.name, "indexado": p.name in ya, "kb": p.stat().st_size // 1024}
                for p in _archivos_de_la_carpeta()]
    return {"carpeta": str(_carpeta()), "existe": _carpeta().is_dir(), "archivos": archivos}


@app.post("/carpeta/indexar")
def indexar_la_carpeta(usuario: Usuario):
    """Indexa los PDFs, audios y videos de la carpeta de conocimiento que el usuario aún no tiene."""
    ya = {d["nombre"] for d in db.listar_documentos(usuario)}
    resultados = []
    for ruta in _archivos_de_la_carpeta():
        if ruta.name in ya:
            continue
        try:
            r = procesar_subida(usuario, ruta.name, ruta.read_bytes())
            resultados.append({"nombre": ruta.name, "ok": True, "mensaje": r["mensaje"],
                               "advertencias": r["advertencias"]})
        except HTTPException as e:
            resultados.append({"nombre": ruta.name, "ok": False, "mensaje": str(e.detail), "advertencias": []})
    return {"carpeta": str(_carpeta()), "resultados": resultados}


@app.get("/documentos/{nombre}/pdf")
def ver_pdf(nombre: str, usuario: Usuario):
    """Devuelve el PDF original, para mostrarlo en la interfaz."""
    contenido = db.obtener_pdf(usuario, nombre)
    if contenido is None:
        raise HTTPException(
            status_code=404,
            detail="El PDF original no está guardado. Bórralo y vuelve a subirlo.",
        )
    return Response(content=contenido, media_type="application/pdf")


class PedidoResaltado(BaseModel):
    pagina: int
    pasajes: list[str]
    giro_extra: int = 0  # giro que el usuario le dio a la página en el visor


@app.post("/documentos/{nombre}/resaltar")
def resaltar(nombre: str, pedido: PedidoResaltado, usuario: Usuario):
    """Imagen (PNG) de una página con los pasajes indicados pintados de amarillo.

    La cabecera X-Encontrado dice si se pudo ubicar algún pasaje en la página.
    """
    contenido = db.obtener_pdf(usuario, nombre)
    if contenido is None:
        raise HTTPException(status_code=404, detail="El PDF original no está guardado.")
    try:
        giro_auto = int(db.obtener_giros(usuario, nombre).get(str(pedido.pagina), 0))
        inicio = time.perf_counter()
        png, encontrado = resaltar_pagina(contenido, pedido.pagina, pedido.pasajes,
                                          giro_auto, pedido.giro_extra)
    except IndexError:
        raise HTTPException(status_code=404, detail="Esa página no existe en el documento.")
    except Exception as e:  # p. ej. OCR no disponible: la interfaz muestra la página sin pintar
        raise HTTPException(status_code=503, detail=f"No se pudo marcar el pasaje: {e}")
    log.info("RESALTADO de '%s' pág. %d en %d ms (encontrado=%s)", nombre, pedido.pagina,
             round((time.perf_counter() - inicio) * 1000), encontrado)
    return Response(content=png, media_type="image/png",
                    headers={"X-Encontrado": "1" if encontrado else "0"})


@app.post("/documentos/{nombre}/indexar_imagenes")
def indexar_imagenes_del_documento(nombre: str, usuario: Usuario):
    """Indexa las imágenes (y audios / videos adjuntos) de un documento ya subido para poder buscarlos."""
    if embeddings.proveedor_medios() == "gemini" and not gemini.disponible():
        raise HTTPException(status_code=400, detail=gemini.mensaje_sin_claves())
    contenido = db.obtener_pdf(usuario, nombre)
    if contenido is None:
        raise HTTPException(status_code=404, detail="El PDF original no está guardado.")
    try:
        resultado = busqueda_imagenes.indexar(usuario, nombre, contenido)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"No se pudieron indexar las imágenes: {e}")
    resultado["avisos"] = gemini.pop_avisos()
    return resultado


class EstadoClave(BaseModel):
    activa: bool


class ClaveNueva(BaseModel):
    clave: str


class ProveedorEmbeddings(BaseModel):
    proveedor: str


@app.get("/gemini/claves")
def claves_de_gemini():
    """Estado de las 3 claves de Gemini (sin mostrar las claves)."""
    return {"claves": gemini.estado_claves()}


@app.put("/gemini/claves/{numero}/clave")
def guardar_clave_de_gemini(numero: int, nueva: ClaveNueva):
    """Guarda la clave pegada en la interfaz (vacía = borrarla). Nunca se devuelve la clave."""
    try:
        gemini.guardar_clave(numero, nueva.clave)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"claves": gemini.estado_claves()}


@app.get("/embeddings")
def modelo_de_embeddings():
    """Con qué modelo se calculan los vectores de los PDFs que se suban ahora."""
    return {"proveedor": embeddings.proveedor_actual(), "gemini_disponible": gemini.disponible()}


@app.post("/embeddings")
def elegir_modelo_de_embeddings(eleccion: ProveedorEmbeddings):
    if eleccion.proveedor not in embeddings.PROVEEDORES:
        raise HTTPException(status_code=400, detail=f"El proveedor debe ser uno de: {', '.join(embeddings.PROVEEDORES)}.")
    if eleccion.proveedor == "gemini" and not gemini.disponible():
        raise HTTPException(status_code=400, detail=gemini.mensaje_sin_claves())
    embeddings.fijar_proveedor(eleccion.proveedor)
    return {"proveedor": eleccion.proveedor, "gemini_disponible": gemini.disponible()}


@app.post("/gemini/claves/{numero}")
def encender_o_apagar_clave(numero: int, estado: EstadoClave):
    """Enciende o apaga una clave de Gemini."""
    if numero not in (1, 2, 3):
        raise HTTPException(status_code=404, detail="Las claves son la 1, 2 y 3.")
    gemini.fijar_activa(numero, estado.activa)
    return {"claves": gemini.estado_claves()}


@app.get("/imagenes/{imagen_id}")
def ver_imagen(imagen_id: int, usuario: Usuario):
    """Una imagen indexada de los documentos del usuario (para mostrarla en el chat)."""
    png = db.obtener_imagen(usuario, imagen_id)
    if png is None:
        raise HTTPException(status_code=404, detail="Imagen no encontrada.")
    return Response(content=png, media_type="image/png")


@app.get("/medios/{imagen_id}/archivo")
def ver_archivo_de_medio(imagen_id: int, usuario: Usuario):
    """El audio o video original al que pertenece un segmento encontrado (para reproducirlo en el chat)."""
    adjunto = db.obtener_adjunto(usuario, imagen_id)
    if adjunto is None:
        raise HTTPException(status_code=404, detail="Archivo no encontrado.")
    return Response(content=adjunto[1], media_type=adjunto[0])


@app.get("/chat/modelo")
def modelo_de_chat():
    """Quién escribe las respuestas: el modelo local (Ollama) o uno de la nube (Gemini)."""
    return {"proveedor": llm.proveedor_chat(), "opciones": list(llm.PROVEEDORES_CHAT),
            "gemini_disponible": gemini.disponible(), "modelo_nube": config.GEMINI_CHAT_MODEL,
            "modelo_local": config.CHAT_MODEL, "llamacpp_listo": llamacpp.ruta_servidor().is_file()
            and llamacpp.ruta_modelo() is not None, "llamacpp_activo": llamacpp.servidor_activo()}


@app.post("/chat/modelo")
def elegir_modelo_de_chat(eleccion: ProveedorEmbeddings):
    if eleccion.proveedor == "gemini" and not gemini.disponible():
        raise HTTPException(status_code=400, detail=gemini.mensaje_sin_claves())
    if eleccion.proveedor == "llamacpp":
        try:
            llamacpp.iniciar()  # arranca llama-server si hace falta (tarda unos segundos en cargar el modelo)
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
    try:
        llm.fijar_proveedor_chat(eleccion.proveedor)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return modelo_de_chat()


@app.get("/embeddings/medios")
def modelo_de_medios():
    """Con qué modelo se calculan los vectores de imágenes, audios y videos nuevos."""
    return {"proveedor": embeddings.proveedor_medios(), "opciones": list(embeddings.MEDIOS)}


@app.post("/embeddings/medios")
def elegir_modelo_de_medios(eleccion: ProveedorEmbeddings):
    try:
        embeddings.fijar_proveedor_medios(eleccion.proveedor)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"proveedor": eleccion.proveedor, "opciones": list(embeddings.MEDIOS)}


@app.get("/documentos/{nombre}/giros")
def ver_giros(nombre: str, usuario: Usuario):
    """Grados que hay que girar cada página escaneada para verla derecha."""
    return {"giros": db.obtener_giros(usuario, nombre)}


@app.get("/documentos/{nombre}/resumen")
def leer_resumen(nombre: str, usuario: Usuario):
    """Devuelve lo guardado del documento: resumen, puntos clave y glosario (null si no existen)."""
    return {campo: db.obtener_campo(usuario, nombre, campo)
            for campo in ("resumen", "puntos_clave", "glosario")}


def _generar(nombre: str, usuario: str, funcion, clave: str, etiqueta: str):
    """Ejecuta una generación con el LLM y traduce los errores a respuestas HTTP."""
    inicio = time.perf_counter()
    try:
        resultado = funcion(usuario, nombre)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:  # errores de Ollama o de formato
        raise HTTPException(status_code=503, detail=str(e))
    log.info("%s de '%s' generado en %d ms", etiqueta, nombre,
             round((time.perf_counter() - inicio) * 1000))
    return {clave: resultado}


@app.post("/documentos/{nombre}/resumen")
def crear_resumen(nombre: str, usuario: Usuario):
    """Genera (o regenera) el resumen del documento y lo guarda."""
    return _generar(nombre, usuario, generar_resumen, "resumen", "RESUMEN")


@app.post("/documentos/{nombre}/preguntas")
def sugerir_preguntas(nombre: str, usuario: Usuario):
    """Genera 3 preguntas que se pueden hacer sobre el documento."""
    return _generar(nombre, usuario, generar_preguntas, "preguntas", "PREGUNTAS sugeridas")


@app.post("/documentos/{nombre}/puntos_clave")
def crear_puntos_clave(nombre: str, usuario: Usuario):
    """Genera los puntos clave del documento y los guarda."""
    return _generar(nombre, usuario, generar_puntos_clave, "puntos_clave", "PUNTOS CLAVE")


@app.post("/documentos/{nombre}/glosario")
def crear_glosario(nombre: str, usuario: Usuario):
    """Genera el glosario del documento y lo guarda."""
    return _generar(nombre, usuario, generar_glosario, "glosario", "GLOSARIO")


@app.post("/documentos/{nombre}/resumen_stream")
def crear_resumen_stream(nombre: str, usuario: Usuario):
    """Igual que POST .../resumen, pero envía el texto por pedazos (una línea JSON cada uno)."""

    def generar():
        inicio = time.perf_counter()
        try:
            partes = []
            for trozo in generar_resumen_stream(usuario, nombre):
                partes.append(trozo)
                yield json.dumps({"texto": trozo}) + "\n"
            # Cuánto del documento vio el modelo (un fragmento por página) y si inventó cifras
            texto_leido, cobertura = muestra_del_documento(usuario, nombre)
            resumen = "".join(partes)
            cobertura["cifras_sin_respaldo"] = cifras_sin_respaldo(resumen, texto_leido)
            textos = cobertura.pop("textos")
            # Por cada página leída, los pasajes que respaldan lo escrito (para pintarlos de amarillo en el PDF)
            detalle = [{"n": i, "documento": nombre, "pagina": pagina, "titulo": "Fuente del resumen",
                        "pasajes": pasajes_de_respuesta(resumen, "", texto)}
                       for i, (pagina, texto) in enumerate(sorted(textos.items()), 1)]
            cobertura["detalle"] = [d for d in detalle if d["pasajes"]]
            yield json.dumps({"cobertura": cobertura}) + "\n"
            log.info("RESUMEN (stream) de '%s' generado en %d ms",
                     nombre, round((time.perf_counter() - inicio) * 1000))
        except ValueError as e:
            yield json.dumps({"error": str(e)}) + "\n"
        except RuntimeError as e:  # errores de Ollama
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(generar(), media_type="application/x-ndjson")


@app.delete("/documentos/{nombre}")
def eliminar(nombre: str, usuario: Usuario):
    if db.borrar_documento(usuario, nombre) == 0:
        raise HTTPException(status_code=404, detail="Documento no encontrado.")
    return {"mensaje": f"'{nombre}' eliminado."}


# ---------------------------------------------------------------------- chat
def registrar_tiempos(endpoint: str, tiempos: dict):
    """Imprime los tiempos en la consola de uvicorn, para las pruebas de velocidad."""
    detalle = " | ".join(f"{nombre}={valor}" for nombre, valor in tiempos.items())
    log.info("TIEMPOS %s (modelo=%s, top_k=%s) -> %s",
             endpoint, nombre_modelo_chat(), config.TOP_K, detalle)


@app.post("/chat")
def chat(datos: PreguntaChat, usuario: Usuario):
    if not datos.pregunta.strip():
        raise HTTPException(status_code=400, detail="Escribe una pregunta.")
    tiempos = {}
    inicio = time.perf_counter()
    if es_pregunta_de_imagenes(datos.pregunta):  # se contesta leyendo el PDF, sin modelo de chat
        texto, detalle, imagenes = respuesta_sobre_imagenes(datos, usuario)
        return {"respuesta": texto, "fuentes": [], "modelo": modelo_de_respuesta_con_imagenes(datos.pregunta),
                "citas": [], "citas_detalle": detalle, "imagenes": imagenes,
                "pregunta_interpretada": datos.pregunta}
    try:
        interpretada, fragmentos, prompt = preparar_consulta(datos, usuario, tiempos)
        if not fragmentos:
            return {"respuesta": "No hay documentos cargados todavía.", "fuentes": []}
        t = time.perf_counter()
        respuesta = generar_respuesta(prompt).replace("[MARCADA]", "")
        tiempos["ollama_ms"] = round((time.perf_counter() - t) * 1000)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))

    tiempos["total_ms"] = round((time.perf_counter() - inicio) * 1000)
    registrar_tiempos("/chat", tiempos)
    citas = fragmentos_citados(respuesta, fragmentos)
    return {"respuesta": respuesta, "fuentes": lista_de_fuentes(fragmentos), "tiempos": tiempos,
            "total_buscado": fragmentos[0].get("total_buscado"),
            "modelo": nombre_modelo_chat(), "pregunta_interpretada": interpretada,
            "citas": citas, "citas_detalle": detalle_de_citas(respuesta, interpretada, fragmentos, citas)}


@app.post("/chat_stream")
def chat_stream(datos: PreguntaChat, usuario: Usuario):
    """Igual que /chat, pero envía la respuesta por pedazos (una línea JSON cada uno).

    Primera línea: {"fuentes": [...], "pregunta_interpretada": "..."}. Luego {"texto": "..."}
    por cada pedazo. Última línea: {"tiempos": {...}, "modelo": "..."}. Si algo falla:
    {"error": "..."}.
    """
    if not datos.pregunta.strip():
        raise HTTPException(status_code=400, detail="Escribe una pregunta.")

    def generar():
        tiempos = {}
        inicio = time.perf_counter()
        try:
            if es_pregunta_de_imagenes(datos.pregunta):  # se contesta leyendo el PDF, sin modelo de chat
                texto, detalle, imagenes = respuesta_sobre_imagenes(datos, usuario)
                yield json.dumps({"fuentes": [], "pregunta_interpretada": datos.pregunta}) + "\n"
                yield json.dumps({"texto": texto}) + "\n"
                tiempos["total_ms"] = round((time.perf_counter() - inicio) * 1000)
                yield json.dumps({"tiempos": tiempos, "modelo": modelo_de_respuesta_con_imagenes(datos.pregunta),
                                  "citas": [], "citas_detalle": detalle, "imagenes": imagenes}) + "\n"
                return
            interpretada, fragmentos, prompt = preparar_consulta(datos, usuario, tiempos)
            if not fragmentos:
                yield json.dumps({"fuentes": []}) + "\n"
                yield json.dumps({"texto": "No hay documentos cargados todavía."}) + "\n"
                return
            yield json.dumps({"fuentes": lista_de_fuentes(fragmentos),
                              "total_buscado": fragmentos[0].get("total_buscado"),
                              "pregunta_interpretada": interpretada}) + "\n"

            t = time.perf_counter()
            primer_token = True
            partes = []
            for trozo in sin_etiqueta_marcada(generar_respuesta_stream(prompt)):
                if primer_token:  # lo que tarda Ollama en empezar a responder
                    tiempos["ollama_primer_token_ms"] = round((time.perf_counter() - t) * 1000)
                    primer_token = False
                partes.append(trozo)
                yield json.dumps({"texto": trozo}) + "\n"
            tiempos["ollama_ms"] = round((time.perf_counter() - t) * 1000)

            tiempos["total_ms"] = round((time.perf_counter() - inicio) * 1000)
            registrar_tiempos("/chat_stream", tiempos)
            respuesta = "".join(partes)
            citas = fragmentos_citados(respuesta, fragmentos)
            yield json.dumps({"tiempos": tiempos, "modelo": nombre_modelo_chat(), "citas": citas,
                              "citas_detalle": detalle_de_citas(respuesta, interpretada,
                                                                fragmentos, citas)}) + "\n"
        except RuntimeError as e:
            yield json.dumps({"error": str(e)}) + "\n"

    return StreamingResponse(generar(), media_type="application/x-ndjson")


# ----------------------------------------------------------- imagen en el chat
@app.post("/imagen/leer")
async def leer_imagen(usuario: Usuario, archivo: UploadFile = File(...)):
    """Lee el texto de una imagen adjunta en el chat (OCR). No describe lo que la imagen muestra."""
    contenido = await archivo.read()
    if len(contenido) > MAX_BYTES_IMAGEN:
        raise HTTPException(status_code=413, detail="La imagen pesa más de 10 MB.")
    try:
        texto = leer_texto_de_imagen(contenido, archivo.filename or "imagen.png")
    except Exception:
        raise HTTPException(status_code=400, detail="No se pudo abrir la imagen. Usa PNG o JPG.")
    return {"texto": texto}


# ------------------------------------------------------------- conversaciones
def _conversacion_propia(usuario: str, conversacion_id: int):
    if not db.conversacion_es_del_usuario(usuario, conversacion_id):
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")


@app.get("/conversaciones")
def conversaciones(usuario: Usuario):
    return db.listar_conversaciones(usuario)


@app.post("/conversaciones")
def nueva_conversacion(usuario: Usuario):
    return {"id": db.crear_conversacion(usuario)}


@app.get("/conversaciones/{conversacion_id}/mensajes")
def mensajes_de(conversacion_id: int, usuario: Usuario):
    _conversacion_propia(usuario, conversacion_id)
    return db.obtener_mensajes(conversacion_id)


@app.post("/conversaciones/{conversacion_id}/mensajes")
def agregar_mensaje(conversacion_id: int, mensaje: MensajeNuevo, usuario: Usuario):
    _conversacion_propia(usuario, conversacion_id)
    db.guardar_mensaje(conversacion_id, mensaje.rol, mensaje.texto, mensaje.extra)
    return {"ok": True}


@app.delete("/conversaciones/{conversacion_id}")
def eliminar_conversacion(conversacion_id: int, usuario: Usuario):
    if not db.borrar_conversacion(usuario, conversacion_id):
        raise HTTPException(status_code=404, detail="Conversación no encontrada.")
    return {"ok": True}


# ----------------------------------------------------------------- exportar
def _archivo(nombre: str, bloques: list[dict], formato: str) -> Response:
    try:
        contenido = exportar.exportar(bloques, formato)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    nombre_archivo = quote(f"{nombre}.{formato}")
    return Response(
        content=contenido,
        media_type=exportar.FORMATOS[formato],
        headers={"Content-Disposition": f"attachment; filename*=UTF-8''{nombre_archivo}"},
    )


@app.get("/exportar/resumen/{nombre}")
def exportar_resumen(nombre: str, usuario: Usuario, formato: str = "pdf"):
    """Resumen, puntos clave y glosario de un documento, en PDF o Word."""
    try:
        archivo, bloques = exportar.bloques_resumen(usuario, nombre)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _archivo(archivo, bloques, formato)


@app.get("/exportar/chat/{conversacion_id}")
def exportar_chat(conversacion_id: int, usuario: Usuario, formato: str = "pdf"):
    """Una conversación completa, en PDF o Word."""
    try:
        archivo, bloques = exportar.bloques_chat(usuario, conversacion_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    return _archivo(archivo, bloques, formato)


# ------------------------------------------------- ver la base de datos (solo lectura)
class ConsultaSQL(BaseModel):
    sql: str


@app.get("/db/resumen")
def db_resumen(usuario: Usuario):
    """Versión de PostgreSQL, tablas con su tamaño, índices y los documentos del usuario."""
    return vista_db.resumen(usuario)


@app.get("/db/muestra/{tabla}")
def db_muestra(tabla: str, usuario: Usuario, n: int = 10):
    try:
        return vista_db.muestra(tabla, n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/db/consulta")
def db_consulta(consulta: ConsultaSQL, usuario: Usuario):
    """Ejecuta una consulta de solo lectura (SELECT / EXPLAIN)."""
    try:
        return vista_db.consulta(consulta.sql)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # error de SQL: se muestra el mensaje de PostgreSQL
        raise HTTPException(status_code=400, detail=f"PostgreSQL: {str(e).splitlines()[0]}")


@app.get("/estado")
def estado():
    return {"modo": config.AI_MODE, "base_de_datos": db.base_responde(), "gemini": gemini.disponible(),
            **info_modelos()}
