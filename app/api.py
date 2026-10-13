"""API FastAPI: subir PDFs, listarlos, borrarlos, chatear, historial y exportar.

Cada petición lleva el nombre del usuario en la cabecera X-Usuario (codificado con %XX).
No es un inicio de sesión con contraseña: solo separa los documentos y conversaciones de
cada nombre. Pensado para uso local.
"""
import json
import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Annotated
from urllib.parse import quote, unquote

from fastapi import Depends, FastAPI, File, Header, HTTPException, Response, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app import config, db, exportar
from app.citas import fragmentos_citados
from app.ingest import IngestaError, ingerir_pdf
from app.llm import (generar_respuesta, generar_respuesta_stream, info_modelos,
                     nombre_modelo_chat)
from app.memoria import (parece_seguimiento, recortar_historial, reformular_pregunta,
                         texto_historial)
from app.resaltar import pasajes_de_respuesta, resaltar_pagina
from app.resumen import (generar_glosario, generar_preguntas, generar_puntos_clave,
                         generar_resumen, generar_resumen_stream)
from app.search import buscar_similares


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Al iniciar la API se crean las tablas si no existen (y se migran las antiguas)
    db.crear_tablas()
    yield


app = FastAPI(title="Chat con PDFs", lifespan=lifespan)
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


def construir_prompt(pregunta: str, fragmentos: list[dict], historial: list[dict] | None = None) -> str:
    """Arma el prompt: instrucciones + historial + contexto numerado + pregunta."""
    contexto = "\n\n".join(
        f"[{f['n']}] ({f['documento']}, página {f['pagina']})\n{f['contenido']}" for f in fragmentos
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
    return [{"n": f["n"], "documento": f["documento"], "pagina": f["pagina"],
             "pasajes": pasajes_de_respuesta(respuesta, pregunta, f["contenido"])}
            for f in fragmentos if f["n"] in citas]


def lista_de_fuentes(fragmentos: list[dict]) -> list[dict]:
    """Fuentes para mostrar: número, documento, página y un pedazo del texto."""
    return [{"n": f["n"], "documento": f["documento"], "pagina": f["pagina"],
             "fragmento": f["contenido"][:240]} for f in fragmentos]


# ---------------------------------------------------------------- documentos
@app.post("/subir")
async def subir(usuario: Usuario, archivo: UploadFile = File(...)):
    contenido = await archivo.read()
    inicio = time.perf_counter()
    try:
        r = ingerir_pdf(usuario, archivo.filename, contenido)
    except IngestaError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:  # errores de Ollama
        raise HTTPException(status_code=503, detail=str(e))

    mensaje = f"'{archivo.filename}' indexado: {r['fragmentos']} fragmentos."
    advertencias = []
    if r["ocr"]:
        mensaje += f" Se usó OCR en {len(r['ocr'])} página(s)."
    if r.get("marcadas"):
        mensaje += f" Se detectaron {r['marcadas']} casillas marcadas (detección aproximada)."
    if r["baja_calidad"]:
        paginas = ", ".join(str(p) for p in r["baja_calidad"])
        advertencias.append(
            f"El texto de la(s) página(s) {paginas} salió poco legible (escaneo borroso o "
            "con mucho ruido). Las respuestas sobre esas páginas pueden ser imprecisas."
        )
    log.info("SUBIDA '%s' (usuario %s): %d fragmentos, OCR en %d pág., en %d ms",
             archivo.filename, usuario, r["fragmentos"], len(r["ocr"]),
             round((time.perf_counter() - inicio) * 1000))
    return {"mensaje": mensaje, "advertencias": advertencias}


@app.get("/documentos")
def documentos(usuario: Usuario):
    return db.listar_documentos(usuario)


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
            for trozo in generar_resumen_stream(usuario, nombre):
                yield json.dumps({"texto": trozo}) + "\n"
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
            interpretada, fragmentos, prompt = preparar_consulta(datos, usuario, tiempos)
            if not fragmentos:
                yield json.dumps({"fuentes": []}) + "\n"
                yield json.dumps({"texto": "No hay documentos cargados todavía."}) + "\n"
                return
            yield json.dumps({"fuentes": lista_de_fuentes(fragmentos),
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


@app.get("/estado")
def estado():
    return {"modo": config.AI_MODE, "base_de_datos": db.base_responde(), **info_modelos()}
