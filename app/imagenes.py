"""Contar y ubicar las imágenes de un PDF.

No usa IA: lee la estructura del archivo, así que el conteo es exacto. Lo que NO puede decir es
qué muestra cada imagen (para eso haría falta un modelo que "vea").
"""
import hashlib
import re

import pymupdf

_CACHE: dict = {}
MIN_LADO = 30  # puntos: más chico que esto son líneas, viñetas o adornos, no imágenes
PAGINA_COMPLETA = 0.85  # una imagen que cubre tanto de la hoja es un escaneo, no una figura


def analizar_imagenes(contenido_pdf: bytes) -> dict:
    """Cuenta las imágenes de un PDF.

    Devuelve {"paginas": total de páginas,
              "figuras": imágenes dentro del contenido (colocadas en las páginas),
              "figuras_distintas": cuántas de ellas son archivos de imagen diferentes,
              "paginas_escaneadas": páginas que son una sola imagen a pantalla completa,
              "por_pagina": {página: {"figuras": n, "escaneada": bool}}}
    """
    clave = hashlib.md5(contenido_pdf).hexdigest()  # el análisis tarda unos segundos: se recuerda
    if clave in _CACHE:
        return _CACHE[clave]
    por_pagina, figuras, escaneadas, distintas = {}, 0, 0, set()
    with pymupdf.open(stream=contenido_pdf, filetype="pdf") as doc:
        total = len(doc)
        for numero, pagina in enumerate(doc, start=1):
            area = max(1.0, pagina.rect.width * pagina.rect.height)
            sin_texto = len(pagina.get_text().strip()) < 20  # un escaneo no trae texto real
            n_figuras, escaneada = 0, False
            for info in pagina.get_image_info(xrefs=True):
                caja = pymupdf.Rect(info["bbox"])
                if caja.width < MIN_LADO or caja.height < MIN_LADO:
                    continue
                cobertura = caja.get_area() / area
                # Escaneo: cubre casi toda la hoja, o grande en una hoja sin texto (hoja girada dentro de otra)
                if cobertura >= PAGINA_COMPLETA or (sin_texto and cobertura >= 0.4):
                    escaneada = True
                else:
                    n_figuras += 1
                    distintas.add(info.get("xref") or (numero, tuple(round(x) for x in caja)))
            if n_figuras or escaneada:
                por_pagina[numero] = {"figuras": n_figuras, "escaneada": escaneada}
            figuras += n_figuras
            escaneadas += escaneada
    resultado = {"paginas": total, "figuras": figuras, "figuras_distintas": len(distintas),
                 "paginas_escaneadas": escaneadas, "por_pagina": por_pagina}
    if len(_CACHE) >= 32:
        _CACHE.pop(next(iter(_CACHE)))
    _CACHE[clave] = resultado
    return resultado


def extraer_imagenes(contenido_pdf: bytes, incluir_escaneadas: bool = False,
                     maximo: int = 40) -> list[dict]:
    """Recorta las imágenes de un PDF como PNG (de hasta 1024 px) para indexarlas.

    Una imagen que se repite (un logo en todas las hojas) se toma una sola vez. Por defecto no se
    incluyen las páginas que son un escaneo completo: suelen ser formularios con datos personales.
    Devuelve [{"pagina", "ancho", "alto", "png"}].
    """
    salida, vistas = [], set()
    with pymupdf.open(stream=contenido_pdf, filetype="pdf") as doc:
        for numero, pagina in enumerate(doc, start=1):
            area = max(1.0, pagina.rect.width * pagina.rect.height)
            sin_texto = len(pagina.get_text().strip()) < 20
            for info in pagina.get_image_info(xrefs=True):
                caja = pymupdf.Rect(info["bbox"]) & pagina.rect
                if caja.is_empty or caja.width < MIN_LADO or caja.height < MIN_LADO:
                    continue
                cobertura = caja.get_area() / area
                if (cobertura >= PAGINA_COMPLETA or (sin_texto and cobertura >= 0.4)) and not incluir_escaneadas:
                    continue
                xref = info.get("xref") or 0
                if xref and xref in vistas:
                    continue
                if xref:
                    vistas.add(xref)
                zoom = min(2.0, 1024 / max(caja.width, caja.height))
                pix = pagina.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=caja)
                salida.append({"pagina": numero, "ancho": pix.width, "alto": pix.height,
                               "png": pix.tobytes("png")})
                if len(salida) >= maximo:
                    return salida
    return salida


def _lista_de_paginas(paginas: list[int]) -> str:
    return ", ".join(str(p) for p in paginas) if paginas else "ninguna"


def describir(nombre: str, datos: dict) -> str:
    """Texto en español con el resultado del conteo, para mostrar en el chat."""
    con_figuras = [p for p, d in datos["por_pagina"].items() if d["figuras"]]
    escaneadas = [p for p, d in datos["por_pagina"].items() if d["escaneada"]]
    lineas = [f"**{nombre}** ({datos['paginas']} páginas):"]
    if datos["figuras"]:
        repetidas = datos["figuras"] - datos["figuras_distintas"]
        distintas = datos["figuras_distintas"]
        nota = (f", de las cuales {distintas} {'es distinta' if distintas == 1 else 'son distintas'} "
                f"(el resto se repite)") if repetidas else ""
        palabra = "imagen" if datos["figuras"] == 1 else "imágenes"
        detalle = ", ".join(f"pág. {p}: {datos['por_pagina'][p]['figuras']}" for p in con_figuras[:15])
        mas = " …" if len(con_figuras) > 15 else ""
        lineas.append(f"- {datos['figuras']} {palabra} dentro del contenido{nota}. ({detalle}{mas})")
    else:
        lineas.append("- No tiene imágenes dentro del contenido (fotos, figuras o logos).")
    if escaneadas:
        if len(escaneadas) == 1:
            lineas.append(f"- 1 página es un escaneo completo (una sola imagen): pág. {escaneadas[0]}.")
        else:
            lineas.append(f"- {len(escaneadas)} páginas son un escaneo completo (una sola imagen): "
                          f"págs. {_lista_de_paginas(escaneadas)}.")
    return "\n".join(lineas)


# Preguntas sobre las imágenes del PDF ("¿cuántas imágenes tiene?", "¿hay fotos?")
_IMAGEN = r"(?:im[aá]gen(?:es)?|fotos?|fotograf[ií]as?|figuras?|ilustraci[oó]n(?:es)?|gr[aá]ficos?)"
_PIDE_CONTEO = re.compile(
    rf"(cu[aá]nt[ao]s?\s+{_IMAGEN}|"
    rf"\b(tiene|trae|hay|contiene|incluye|existen?|aparece[n]?)\b.{{0,25}}\b{_IMAGEN}\b)", re.I)
# Preguntas sobre QUÉ muestra una imagen ("¿hay una foto de un perro?"): eso no se puede contestar sin visión.
# No cuenta "la imagen de la página 3" (eso señala dónde está, no qué muestra).
_PIDE_CONTENIDO = re.compile(
    rf"\b{_IMAGEN}\s+(?:de|del|con)\s+(?:un|una|unos|unas|el|la|los|las)\s+(?!p[aá]g|hoja|secci[oó]n|cap[ií]tulo)"
    rf"|\b{_IMAGEN}\s+(?:de|del|con)\s+(?!(?:un|una|unos|unas|el|la|los|las)\b)\w+", re.I)


# Preguntas sobre audios y videos ("¿hay un audio donde hablen de energía solar?", "video de un perro")
_MEDIO = r"(?:audios?|v[ií]deos?|grabaci[oó]n(?:es)?|sonidos?)"
_PIDE_MEDIO = re.compile(
    # buscar: "¿hay un audio de…?", "muéstrame el video donde…", "busca una grabación sobre…"
    rf"\b(?:hay|existe|existen|busca\w*|encuentra\w*|mu[eé]stra\w*|ens[eé][ñn]a\w*|d[oó]nde|tienes|tiene|tienen)\b.{{0,30}}\b{_MEDIO}\b"
    # o "un audio donde / en el que / que hable de…"
    rf"|\b{_MEDIO}\s+(?:donde|en\s+(?:el|la|los|las)\s+que|que\s+(?:hable|hablen|trate|mencione|diga|muestre|suene|contenga))\b", re.I)


# Preguntas sobre QUÉ MUESTRA una imagen: "¿qué hay en la imagen de la página 3?", "describe la figura 2"
_FIGURA = r"(?:im[aá]gen(?:es)?|fotos?|fotograf[ií]as?|figuras?|ilustraci[oó]n(?:es)?|gr[aá]ficos?|diagramas?)"
_PIDE_DESCRIPCION = re.compile(
    rf"\b(?:qu[eé]|cu[aá]l(?:es)?)\s+(?:se\s+)?(?:muestra\w*|hay|aparece\w*|ve|representa\w*|significa\w*|contiene\w*|dice\w*|son)\b.{{0,40}}\b{_FIGURA}\b"
    rf"|\b(?:describ\w+|explic\w+|anali[cz]\w+|interpret\w+)\b.{{0,40}}\b{_FIGURA}\b"
    rf"|\bde\s+qu[eé]\s+(?:trata|habla|va)\b.{{0,30}}\b{_FIGURA}\b", re.I)


def pide_descripcion_de_imagen(pregunta: str) -> bool:
    return bool(_PIDE_DESCRIPCION.search(pregunta))


def pagina_pedida(pregunta: str) -> int | None:
    m = re.search(r"p[aá]g(?:ina)?\.?\s*(\d+)", pregunta, re.I)
    return int(m.group(1)) if m else None


def pide_audio_o_video(pregunta: str) -> bool:
    return bool(_PIDE_MEDIO.search(pregunta))


def es_pregunta_de_imagenes(pregunta: str) -> bool:
    return bool(_PIDE_CONTEO.search(pregunta) or _PIDE_CONTENIDO.search(pregunta) or _PIDE_MEDIO.search(pregunta)
                or _PIDE_DESCRIPCION.search(pregunta))


def pide_contenido_de_imagen(pregunta: str) -> bool:
    return bool(_PIDE_CONTENIDO.search(pregunta) or _PIDE_MEDIO.search(pregunta))
