"""Señalar en el PDF el pasaje del que salió una respuesta: pintarlo de amarillo sobre la página.

Pasos:
1. `pasajes_de_respuesta`: de un fragmento de ~600 caracteres se elige la frase (unas 20 palabras)
   que más palabras comparte con la respuesta y la pregunta.
2. `resaltar_pagina`: se buscan esas palabras en la página (con el texto del PDF o, si es un
   escaneo, con OCR), se localizan sus posiciones y se pintan encima de la imagen de la página.

La búsqueda en la página es por palabras (no por texto exacto) para tolerar los errores del OCR.
"""
import hashlib
import re
import unicodedata
from collections import OrderedDict

import numpy as np
import pymupdf

from app import config

DPI = 110  # igual que el visor de la interfaz
PALABRAS_POR_PASAJE = 20
_VACIAS = {
    "segun", "contexto", "informacion", "proporcionada", "documento", "documentos", "tambien",
    "ademas", "entre", "para", "desde", "hasta", "esta", "este", "estos", "estas", "tiene",
    "tienen", "puede", "pueden", "sobre", "como", "cual", "cuales", "donde", "cuando", "porque",
    "pero", "esos", "esas", "quien", "quienes", "cuanto", "cuantos", "cuanta", "cuantas",
}


def _norm(texto: str) -> str:
    """Minúsculas, sin tildes y solo letras/números (para comparar palabras)."""
    sin_tildes = "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                         if unicodedata.category(c) != "Mn")
    return re.sub(r"\W", "", sin_tildes)


def _interesantes(texto: str) -> set[str]:
    """Palabras de la respuesta/pregunta que sirven para ubicar el pasaje."""
    normalizadas = (_norm(p) for p in texto.split())
    return {p for p in normalizadas if (len(p) >= 4 or p.isdigit()) and p not in _VACIAS}


# ------------------------------------------------------------ elegir el pasaje
def _pasajes_de_bloque(interes: set[str], fragmento: str) -> list[str]:
    """Para un bloque de 'respuestas marcadas': el texto de la pregunta y el de la opción marcada.

    En la página están en líneas separadas (la pregunta, y más abajo la casilla marcada), por
    eso se devuelven como dos pasajes para pintar las dos cosas.
    """
    # Al guardarse, los saltos de línea del bloque se vuelven espacios: los ítems quedan separados
    # por " - " ("... aproximada): - 1. Edad: → 46 a 55 años - 2. Sexo: → Hombre ...")
    cuerpo = fragmento.split("):", 1)[-1]
    mejor, puntaje_mejor = None, 0
    for item in re.split(r"\s-\s(?=\S)", " " + cuerpo):
        if "→" not in item:
            continue
        pregunta, _, opciones = item.partition("→")
        puntaje = len(_interesantes(pregunta + " " + opciones) & interes)
        if puntaje > puntaje_mejor:
            mejor, puntaje_mejor = (pregunta, opciones), puntaje
    if not mejor:
        return []
    pregunta, opciones = mejor
    pregunta = re.sub(r"^\s*\d+[.)]?\s*", "", pregunta).strip()  # sin el número ('1. Edad:' -> 'Edad:')
    if pregunta.startswith("("):  # '(opción sin pregunta identificada)': no está escrita en la página
        pregunta = ""
    primera_opcion = opciones.split(";")[0].strip()
    return [p for p in (pregunta, primera_opcion) if p]


def pasajes_de_respuesta(respuesta: str, pregunta: str, fragmento: str) -> list[str]:
    """Frases del fragmento que respaldan la respuesta (como máximo 2)."""
    interes = _interesantes(respuesta) | _interesantes(pregunta)
    if fragmento.startswith("RESPUESTAS MARCADAS"):
        return _pasajes_de_bloque(interes, fragmento)
    palabras = fragmento.split()
    if not interes or not palabras:
        return []
    ancho = min(PALABRAS_POR_PASAJE, len(palabras))
    puntajes = []
    for i in range(len(palabras) - ancho + 1):
        presentes = {_norm(p) for p in palabras[i:i + ancho]}
        puntajes.append(len(presentes & interes))
    mejor = max(puntajes)
    if mejor == 0:
        return []
    elegidos = []
    for i in sorted(range(len(puntajes)), key=lambda k: puntajes[k], reverse=True):
        if puntajes[i] < max(2, mejor * 0.6) and elegidos:
            break
        if all(abs(i - j) >= ancho for j in elegidos):  # que no se solapen
            elegidos.append(i)
        if len(elegidos) == 2:
            break
    return [" ".join(palabras[i:i + ancho]) for i in sorted(elegidos)]


# ------------------------------------------------------ palabras de la página
_CACHE_PALABRAS: OrderedDict = OrderedDict()  # el OCR de una página tarda; se recuerda
_MAX_CACHE = 64


def _palabras_de_pagina(pagina, clave: str) -> list:
    """Palabras de la página con su posición. Si no hay texto (escaneo), se usa OCR.

    Las posiciones están en el sistema de coordenadas de la página sin girar.
    """
    if clave in _CACHE_PALABRAS:
        _CACHE_PALABRAS.move_to_end(clave)
        return _CACHE_PALABRAS[clave]
    palabras = pagina.get_text("words")
    if len(palabras) < 3 and (pagina.get_images() or pagina.get_drawings()):
        ocr = pagina.get_textpage_ocr(language=config.OCR_LANG, dpi=config.OCR_DPI, full=True,
                                      tessdata=config.TESSDATA_DIR or None)
        palabras = pagina.get_text("words", textpage=ocr)
    _CACHE_PALABRAS[clave] = palabras
    if len(_CACHE_PALABRAS) > _MAX_CACHE:
        _CACHE_PALABRAS.popitem(last=False)
    return palabras


def _ubicar(pasaje: str, palabras: list, tokens: list[str]) -> list:
    """Palabras de la página que mejor coinciden con el pasaje (una ventana de palabras seguidas)."""
    # La etiqueta [MARCADA] la agrega el sistema: no está escrita en la página
    buscadas = [t for t in (_norm(p) for p in pasaje.split()) if t and t != "marcada"]
    if not buscadas:
        return []
    conjunto, largo = set(buscadas), len(buscadas)
    # El OCR mete "palabras" de ruido (casillas, símbolos): la ventana es un poco más larga
    ventana = int(largo * 1.4) + 1
    mejor, inicio = -1, 0
    for i in range(max(1, len(tokens) - ventana + 1)):
        puntaje = sum(1 for t in tokens[i:i + ventana] if t in conjunto)
        if puntaje > mejor:
            mejor, inicio = puntaje, i
    if mejor < largo * 0.5:  # no se encontró con suficiente seguridad
        return []
    # Se recorta la ventana desde la primera hasta la última palabra que sí coincide
    coinciden = [i for i in range(inicio, min(inicio + ventana, len(tokens))) if tokens[i] in conjunto]
    # Las letras sueltas del OCR (casillas leídas como 'O' o 'X') no cuentan en los extremos
    while coinciden and len(tokens[coinciden[0]]) <= 1:
        coinciden.pop(0)
    while coinciden and len(tokens[coinciden[-1]]) <= 1:
        coinciden.pop()
    if not coinciden:
        return []
    # Si la primera y la última palabra del pasaje están en la ventana, se ajusta a ellas
    # (evita arrastrar una palabra suelta de la línea anterior)
    primero = next((i for i in coinciden if tokens[i] == buscadas[0]), None)
    ultimo = next((i for i in reversed(coinciden) if tokens[i] == buscadas[-1]), None)
    if primero is not None and ultimo is not None and primero <= ultimo:
        coinciden = [i for i in coinciden if primero <= i <= ultimo]
    return palabras[coinciden[0]:coinciden[-1] + 1]


# --------------------------------------------------------------------- pintar
def resaltar_pagina(contenido_pdf: bytes, pagina_n: int, pasajes: list[str],
                    giro_auto: int = 0, giro_extra: int = 0) -> tuple[bytes, bool]:
    """Imagen PNG de la página con los pasajes pintados de amarillo.

    `giro_auto`: giro con el que se leyó la página (hojas escaneadas de lado).
    `giro_extra`: giro adicional que el usuario le dio en el visor.
    Devuelve (png, encontrado). Si no se pudo ubicar ningún pasaje, devuelve la página sin pintar.
    """
    with pymupdf.open(stream=contenido_pdf, filetype="pdf") as doc:
        pagina = doc[pagina_n - 1]
        original = pagina.rotation

        # Las palabras se leen con el giro que deja el texto derecho (el OCR lo necesita)
        pagina.set_rotation((original + giro_auto) % 360)
        clave = f"{hashlib.md5(contenido_pdf).hexdigest()}:{pagina_n}:{giro_auto}"
        palabras = _palabras_de_pagina(pagina, clave)
        tokens = [_norm(p[4]) for p in palabras]

        # Se dibuja con el giro que ve el usuario en el visor
        pagina.set_rotation((original + giro_auto + giro_extra) % 360)
        pix = pagina.get_pixmap(dpi=DPI)
        imagen = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n).copy()

        encontrado = False
        zoom = pymupdf.Matrix(DPI / 72, DPI / 72)
        for pasaje in pasajes:
            lineas = {}  # un rectángulo por línea de texto
            for p in _ubicar(pasaje, palabras, tokens):
                rect = pymupdf.Rect(p[:4])
                clave_linea = (p[5], p[6])  # (bloque, línea)
                lineas[clave_linea] = lineas[clave_linea] | rect if clave_linea in lineas else rect
            for rect in lineas.values():
                # coordenadas sin girar -> píxeles de la imagen girada
                r = (rect * pagina.rotation_matrix) * zoom
                x0, y0 = max(0, int(r.x0) - 2), max(0, int(r.y0) - 2)
                x1, y1 = min(pix.width, int(r.x1) + 2), min(pix.height, int(r.y1) + 2)
                if x1 > x0 and y1 > y0:
                    # Se multiplica el azul: el fondo queda amarillo y el texto sigue legible
                    imagen[y0:y1, x0:x1, 2] = (imagen[y0:y1, x0:x1, 2] * 0.35).astype(np.uint8)
                    encontrado = True

        salida = pymupdf.Pixmap(pix.colorspace, pix.width, pix.height, imagen.tobytes(), False)
        return salida.tobytes("png"), encontrado
