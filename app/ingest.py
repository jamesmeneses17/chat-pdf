"""Ingesta: PDF -> páginas -> fragmentos -> vectores -> base de datos."""
from app import db
from app.chunker import dividir_texto
from app.embeddings import obtener_embeddings
from app.pdf_loader import OcrNoDisponible, extraer_paginas


class IngestaError(Exception):
    """Error esperable (PDF escaneado, duplicado...) con mensaje para el usuario."""


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
    vectores = obtener_embeddings([f for _, f in fragmentos])  # en lotes
    filas = [(usuario, nombre, numero, f, v) for (numero, f), v in zip(fragmentos, vectores)]

    # 4. Guardar todo en una sola transacción (o se guarda todo o nada)
    with db.conectar() as conn:
        with conn.transaction():
            with conn.cursor() as cur:
                cur.executemany(
                    "INSERT INTO chunks (usuario, documento, pagina, contenido, embedding) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    filas,
                )
            db.guardar_pdf(conn, usuario, nombre, contenido_pdf, info["giros"])  # para el visor
    return {"fragmentos": len(filas), **info}
