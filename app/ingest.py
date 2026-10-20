"""Ingesta: PDF -> páginas -> fragmentos -> vectores -> base de datos."""
from app import db
from app.chunker import dividir_texto
from app.embeddings import obtener_embeddings, proveedor_actual
from app.pdf_loader import OcrNoDisponible, extraer_paginas


class IngestaError(Exception):
    """Error esperable (PDF escaneado, duplicado...) con mensaje para el usuario."""


def _vectores(textos: list[str]) -> tuple[list, str, str | None]:
    """Vectores de los fragmentos con el modelo elegido en la interfaz. Devuelve (vectores, modelo, aviso)."""
    proveedor, aviso = proveedor_actual(), None
    if proveedor == "gemini":
        try:
            return obtener_embeddings(textos, proveedor="gemini"), proveedor, aviso
        except RuntimeError as e:  # sin clave, sin internet o límite de uso: se usa Ollama
            proveedor, aviso = "ollama", f"Gemini no pudo calcular los embeddings ({e}) Se usó Ollama."
    return obtener_embeddings(textos, proveedor=proveedor), proveedor, aviso


def ingerir_transcripcion(usuario: str, nombre: str, segmentos: list[dict]) -> dict:
    """Guarda como fragmentos de texto lo que se dice en un audio o video.

    `segmentos`: [{"numero": n, "inicio": segundos, "texto": "..."}]. El «número» hace de página (para citar
    y para el resumen) y `inicio` permite reproducir desde ese minuto.
    """
    fragmentos = [(s["numero"], s["inicio"], f) for s in segmentos for f in dividir_texto(s["texto"])]
    if not fragmentos:
        return {"fragmentos": 0, "proveedor": None, "aviso_embeddings": None}
    vectores, proveedor, aviso = _vectores([f for _, _, f in fragmentos])
    filas = [(usuario, nombre, n, f, v, proveedor, ini) for (n, ini, f), v in zip(fragmentos, vectores)]
    with db.conectar() as conn:
        with conn.transaction():
            conn.execute("DELETE FROM chunks WHERE usuario = %s AND documento = %s", (usuario, nombre))
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (usuario, documento, pagina, contenido, embedding, proveedor, inicio) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)", filas)
            db.guardar_pdf(conn, usuario, nombre, b"", {})  # sin PDF, pero guarda su resumen, glosario...
    return {"fragmentos": len(filas), "proveedor": proveedor, "aviso_embeddings": aviso}


def ingerir_pdf(usuario: str, nombre: str, contenido_pdf: bytes) -> dict:
    """Indexa un PDF del usuario. Devuelve {"fragmentos": n, "ocr": [...], "baja_calidad": [...]}."""
    if db.documento_existe(usuario, nombre):
        raise IngestaError(f"El documento '{nombre}' ya fue subido. No se duplicó.")

    # 1. Texto por página (con OCR en las páginas escaneadas)
    try:
        paginas, info = extraer_paginas(contenido_pdf)
    except OcrNoDisponible as e:
        raise IngestaError(str(e)) from e
    except Exception as e:
        raise IngestaError("No se pudo leer el archivo. ¿Es un PDF válido?") from e
    if not paginas:
        raise IngestaError(
            "No se encontró texto en este PDF, ni siquiera con OCR. "
            "¿Está vacío o la imagen es muy borrosa?"
        )

    # 2 y 3. Fragmentos + embeddings (todo antes de tocar la BD)
    fragmentos = [(numero, f) for numero, texto in paginas for f in dividir_texto(texto)]
    vectores, proveedor, aviso = _vectores([f for _, f in fragmentos])
    filas = [(usuario, nombre, numero, f, v, proveedor) for (numero, f), v in zip(fragmentos, vectores)]

    # 4. Guardar todo en una sola transacción (o se guarda todo o nada)
    with db.conectar() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (usuario, documento, pagina, contenido, embedding, proveedor) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    filas,
                )
            db.guardar_pdf(conn, usuario, nombre, contenido_pdf, info["giros"])  # para el visor
    return {"fragmentos": len(filas), "proveedor": proveedor, "aviso_embeddings": aviso, **info}
