"""Extrae el texto de un PDF, página por página, usando PyMuPDF.

Si una página no tiene texto (es una imagen escaneada) se aplica OCR con Tesseract.
Las hojas escaneadas de lado se detectan probando giros hasta que el texto sea legible.
"""
import re

import pymupdf

from app import config

CALIDAD_BUENA = 0.45  # con esta calidad el OCR se da por bueno y no se prueban más giros
CALIDAD_MINIMA = 0.30  # por debajo de esto se avisa que el texto es poco legible
GIROS = (0, 270, 90, 180)  # grados que se prueban, en este orden

_PALABRA = re.compile(r"^[A-Za-zÁÉÍÓÚÜáéíóúüñÑ]{2,}[.,;:?!]?$")
_VOCAL = re.compile(r"[aeiouáéíóúAEIOU]")


class OcrNoDisponible(Exception):
    """Tesseract no está instalado o no se encontró el idioma."""


def calidad_texto(texto: str) -> float:
    """Qué tan legible es el texto del OCR: porcentaje de 'palabras' que parecen palabras.

    Texto bien leído da ~0.6; texto basura (símbolos, letras sueltas) da menos de 0.2.
    """
    palabras = texto.split()
    if not palabras:
        return 0.0
    buenas = sum(1 for p in palabras if _PALABRA.match(p) and _VOCAL.search(p))
    return buenas / len(palabras)


# ---------------------------------------------------------- casillas marcadas
# El OCR lee casi siempre una casilla VACÍA como 'O', 'DO', 'BD', 'DD'... y una casilla MARCADA
# (con una X a mano) como cualquier otra cosa: 'N', 'X]', 'XX', o pegada al texto ('ASecundaria').
# Con eso se detectan las opciones elegidas en formularios escaneados. Es una heurística:
# acierta con las marcas típicas, pero no es infalible.
MARCA = "[MARCADA]"
_CASILLA_VACIA = re.compile(r"^[OD0BQ]{1,2}$")
_MARCA_SUELTA = re.compile(r"[XKNMWHAV\]\[¡¿×]")
_MARCA_PEGADA = re.compile(r"^[¿¡\[\(]?[A-Z]{1,2}(?=[A-ZÁÉÍÓÚ][a-záéíóúñ]|\d)")  # 'ASecundaria', 'X7a10'
_SOLO_NUMERO = re.compile(r"^\d+[.)]?$")


def _lineas_ocr(palabras: list) -> list[list[str]]:
    """Las palabras del OCR agrupadas por línea de texto, en orden de lectura."""
    grupos: dict = {}
    for w in sorted(palabras, key=lambda w: (w[5], w[6], w[7])):  # (bloque, línea, palabra)
        grupos.setdefault((w[5], w[6]), []).append(w[4])
    return list(grupos.values())


def _prefijo_pegado(token: str):
    """Marca pegada al texto ('ASecundaria'). Una 'O' o 'D' pegada es una casilla vacía, no una marca."""
    m = _MARCA_PEGADA.match(token)
    if not m:
        return None
    letras = re.sub(r"[^A-Z]", "", m.group(0))
    return None if _CASILLA_VACIA.match(letras) else m


def _parece_marca(token: str) -> bool:
    """¿La primera palabra de la línea parece una casilla marcada leída por el OCR?"""
    if _prefijo_pegado(token):
        return True
    return (len(token) <= 3 and token == token.upper() and bool(_MARCA_SUELTA.search(token))
            and not _SOLO_NUMERO.match(token))


def marcar_casillas(lineas: list[list[str]]) -> tuple[list[str], list[tuple[str, str]]]:
    """Antepone [MARCADA] a las opciones que parecen elegidas.

    Devuelve (líneas, pares) donde cada par es (pregunta, opción marcada).
    Solo se mira dentro de listas de opciones (líneas vecinas que también parecen opciones,
    con al menos una casilla vacía cerca), para no confundir con el texto normal.
    """
    vacia = [bool(_CASILLA_VACIA.match(l[0])) for l in lineas]
    opcion = [vacia[i] or _parece_marca(l[0]) for i, l in enumerate(lineas)]
    textos, pares = [], []
    pregunta, distancia = "", 99  # última pregunta numerada vista ("1. Edad:") y hace cuántas líneas
    for i, palabras in enumerate(lineas):
        primera = palabras[0]
        if _SOLO_NUMERO.match(primera) and len(palabras) > 1:
            pregunta, distancia = " ".join(palabras)[:110], 0
        else:
            distancia += 1
        if not vacia[i] and _parece_marca(primera):
            vecinas = sum(opcion[max(0, i - 8):i]) + sum(opcion[i + 1:i + 9])
            vacias_cerca = sum(vacia[max(0, i - 8):i]) + sum(vacia[i + 1:i + 9])
            # Una marca sola en la línea, sin texto de opción, no dice nada: se deja como está
            tiene_texto = len(palabras) > 1 or bool(_prefijo_pegado(primera))
            if vecinas >= 2 and vacias_cerca >= 1 and tiene_texto:
                if _prefijo_pegado(primera):  # 'ASecundaria' -> '[MARCADA] Secundaria'
                    palabras = [_MARCA_PEGADA.sub(MARCA + " ", primera, count=1)] + palabras[1:]
                else:  # 'N', 'X]', 'XX'
                    palabras = [MARCA] + palabras[1:]
                opcion_marcada = " ".join(palabras).replace(MARCA, "").strip()
                pares.append((pregunta if distancia <= 14 else "", opcion_marcada))
        textos.append(" ".join(palabras))
    return textos, pares


def bloque_de_respuestas(pagina_n: int, pares: list[tuple[str, str]]) -> str:
    """Resumen compacto de las opciones marcadas de una página, con su pregunta.

    Va aparte del texto de la página para que la pregunta y su respuesta marcada queden en el
    mismo fragmento aunque en la página estén lejos (o cortadas por el límite de un fragmento).
    """
    if not pares:
        return ""
    por_pregunta: dict = {}
    for pregunta, opcion in pares:
        por_pregunta.setdefault(pregunta or "(opción sin pregunta identificada)", []).append(opcion)
    lineas = [f"RESPUESTAS MARCADAS EN LA PÁGINA {pagina_n} (casillas marcadas, detección aproximada):"]
    lineas += [f"- {pregunta} → {'; '.join(opciones)}" for pregunta, opciones in por_pregunta.items()]
    return "\n".join(lineas)


def _ocr(pagina) -> tuple[str, str, list]:
    """OCR de la página. Devuelve (texto con casillas marcadas, texto plano, pares marcados)."""
    ocr = pagina.get_textpage_ocr(
        language=config.OCR_LANG,
        dpi=config.OCR_DPI,
        full=True,  # hace OCR de toda la página
        tessdata=config.TESSDATA_DIR or None,
    )
    lineas = _lineas_ocr(pagina.get_text("words", textpage=ocr))
    plano = "\n".join(" ".join(l) for l in lineas).strip()
    con_marcas, pares = marcar_casillas(lineas)
    texto = "\n".join(con_marcas).strip() if config.MARCAS_EN_TEXTO else plano
    return texto, plano, pares


def _texto_con_ocr(pagina) -> tuple[str, float, int, list]:
    """Lee una página-imagen con OCR. Devuelve (texto, calidad, giro, pares marcados).

    Si el texto sale ilegible, gira la página (270°, 90°, 180°) y se queda con la
    versión más legible. Así se leen las hojas escaneadas de lado o al revés.
    `giro` son los grados que hubo que girarla para que quede derecha.
    """
    original = pagina.rotation
    mejor = ("", -1.0, 0, [])
    try:
        for giro in GIROS:
            pagina.set_rotation((original + giro) % 360)
            texto, plano, pares = _ocr(pagina)
            calidad = calidad_texto(plano)  # la calidad se mide sin las etiquetas [MARCADA]
            if calidad > mejor[1]:
                mejor = (texto, calidad, giro, pares)
            if calidad >= CALIDAD_BUENA:
                break
    except Exception as e:
        raise OcrNoDisponible(
            "Este PDF es escaneado y no se pudo aplicar OCR. "
            "¿Está instalado Tesseract con el idioma español? "
            "Revisa OCR_LANG y TESSDATA_DIR en el .env."
        ) from e
    finally:
        pagina.set_rotation(original)
    return mejor[0], max(mejor[1], 0.0), mejor[2], mejor[3]


def extraer_paginas(contenido_pdf: bytes) -> tuple[list[tuple[int, str]], dict]:
    """Devuelve (paginas, info).

    paginas: lista de (numero_de_pagina, texto); las páginas vacías se omiten.
    info: {"ocr": [páginas leídas con OCR], "baja_calidad": [páginas poco legibles],
           "giros": {página: grados para enderezarla}}
    """
    paginas = []
    info = {"ocr": [], "baja_calidad": [], "giros": {}, "marcadas": 0}
    with pymupdf.open(stream=contenido_pdf, filetype="pdf") as doc:
        for i, pagina in enumerate(doc, start=1):  # las páginas empiezan en 1
            texto = pagina.get_text().strip()
            bloque = ""
            if not texto and (pagina.get_images() or pagina.get_drawings()):
                texto, calidad, giro, pares = _texto_con_ocr(pagina)  # escaneada: usar OCR
                info["ocr"].append(i)
                info["marcadas"] += len(pares)
                bloque = bloque_de_respuestas(i, pares)
                if texto and calidad < CALIDAD_MINIMA:
                    info["baja_calidad"].append(i)
                elif giro:
                    info["giros"][i] = giro  # el visor la mostrará derecha
            if texto:
                paginas.append((i, texto))
            if bloque:  # texto aparte de la misma página (misma cita y mismo resaltado)
                paginas.append((i, bloque))
    return paginas, info
