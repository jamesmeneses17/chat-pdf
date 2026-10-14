"""Buscar dentro de las imágenes de los PDFs con una frase ("¿hay un perro?").

Al subir un PDF se recortan sus imágenes, se convierten en vectores con Gemini y se guardan. Al
preguntar, la frase también se convierte en vector y se muestran las imágenes más parecidas.

Importante: el parecido es una ESTIMACIÓN. Siempre habrá imágenes "más parecidas", aunque ninguna
muestre lo que se busca; por eso se muestran con su porcentaje y hay que mirarlas.
"""
import re

import numpy as np

from app import config, db, gemini
from app.imagenes import extraer_imagenes

_ARTICULO = r"(?:un|una|unos|unas|el|la|los|las)"
_TEMA = re.compile(
    rf"(?:im[aá]gen(?:es)?|fotos?|fotograf[ií]as?|figuras?|ilustraci[oó]n(?:es)?|gr[aá]ficos?)\s+"
    rf"(?:de|del|con)\s+(?:{_ARTICULO}\s+)?(?P<tema>[^?¿!.]+)", re.I)


def tema_de_la_pregunta(pregunta: str) -> str:
    """Lo que se busca: 'hay una imagen de un perro?' -> 'perro'. Si no se reconoce, la pregunta entera."""
    m = _TEMA.search(pregunta)
    tema = m.group("tema").strip() if m else re.sub(r"[¿?¡!]", "", pregunta).strip()
    return tema


def indexar(usuario: str, nombre: str, contenido_pdf: bytes) -> dict:
    """Envía las imágenes del PDF a Gemini y guarda sus vectores.

    Devuelve {"indexadas": n, "total": imágenes encontradas, "error": mensaje o None, "activo": bool}.
    Si falla a mitad (límite de uso, sin internet), se guardan las que ya se hicieron.
    """
    if not gemini.disponible():
        return {"indexadas": 0, "total": 0, "error": None, "activo": False}
    imagenes = extraer_imagenes(contenido_pdf, config.IMAGENES_INCLUIR_ESCANEADAS,
                                config.IMAGENES_MAX_POR_DOC)
    filas, error = [], None
    for imagen in imagenes:
        try:
            vector = gemini.embedding_imagen(imagen["png"])
        except RuntimeError as e:
            error = str(e)
            break
        filas.append((imagen["pagina"], imagen["ancho"], imagen["alto"], imagen["png"], np.array(vector)))
    db.guardar_imagenes(usuario, nombre, filas)
    return {"indexadas": len(filas), "total": len(imagenes), "error": error, "activo": True}


def buscar(usuario: str, consulta: str, documentos: list[str] | None = None) -> tuple[list[dict], list[dict]]:
    """Busca las imágenes que coinciden con la frase. Devuelve (coinciden, candidatas).

    `candidatas` son las más parecidas, de mayor a menor, con {id, documento, pagina, similitud}.
    `coinciden` son solo las que pasan los dos filtros:
      1. parecido al menos IMAGENES_MIN_SIM (si ninguna llega, no hay coincidencia), y
      2. estar a menos de IMAGENES_MARGEN de la mejor (si el gato gana por mucho, no salen el perro
         ni la lombriz; si la pregunta es general, "un animal", salen todos los animales).
    """
    vector = np.array(gemini.embedding_consulta(consulta))
    candidatas = db.buscar_imagenes(usuario, vector, documentos, 20)
    if not candidatas:
        return [], []
    mejor = candidatas[0]["similitud"]
    coinciden = [c for c in candidatas
                 if c["similitud"] >= config.IMAGENES_MIN_SIM and c["similitud"] >= mejor - config.IMAGENES_MARGEN]
    return coinciden[:config.IMAGENES_TOP], candidatas
