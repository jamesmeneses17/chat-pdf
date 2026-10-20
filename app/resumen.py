"""Genera el resumen, las preguntas sugeridas, los puntos clave y el glosario de un documento."""
import re
import unicodedata

from app import config, db
from app.llm import generar_respuesta, generar_respuesta_stream


MIN_CARACTERES_POR_PAGINA = 150  # por debajo de esto un fragmento ya no dice casi nada


def _recortar(texto: str, limite: int) -> str:
    """Corta el texto en `limite` caracteres sin partir una palabra."""
    if len(texto) <= limite:
        return texto
    return texto[:limite].rsplit(" ", 1)[0] + "…"


def muestra_del_documento(usuario: str, nombre: str) -> tuple[str, dict]:
    """Un fragmento por página, juntos en orden. Devuelve (texto, cobertura).

    De cada página se toma su fragmento más largo (el que más contenido tiene). El texto total
    cabe en RESUMEN_CHARS: cada página recibe una parte igual del espacio. Si hay tantas páginas
    que cada una tendría muy poco, se eligen páginas repartidas de principio a fin.
    `cobertura` dice qué parte del documento se usó, para mostrarlo junto al resultado.
    """
    filas = db.obtener_fragmentos_con_pagina(usuario, nombre)
    if not filas:
        raise ValueError("Documento no encontrado.")
    # Los bloques de "respuestas marcadas" son un apoyo para el chat: no van en resúmenes ni glosarios
    del_texto = [(p, c) for p, c in filas if not c.startswith("RESPUESTAS MARCADAS") and "→" not in c]
    del_texto = del_texto or filas

    mejor_de_pagina: dict[int, str] = {}
    for pagina, contenido in del_texto:
        if len(contenido) > len(mejor_de_pagina.get(pagina, "")):
            mejor_de_pagina[pagina] = contenido
    paginas = sorted(mejor_de_pagina)

    maximo = max(1, config.RESUMEN_CHARS // MIN_CARACTERES_POR_PAGINA)
    elegidas = paginas
    if len(paginas) > maximo:  # documento muy largo: páginas repartidas
        elegidas = [paginas[int(i * len(paginas) / maximo)] for i in range(maximo)]
    presupuesto = config.RESUMEN_CHARS // len(elegidas)
    partes = [f"[Página {p}] {_recortar(mejor_de_pagina[p], presupuesto)}" for p in elegidas]

    texto_total = sum(len(c) for _, c in del_texto)
    texto_usado = sum(len(_recortar(mejor_de_pagina[p], presupuesto)) for p in elegidas)
    cobertura = {
        "fragmentos_usados": len(elegidas),
        "fragmentos_total": len(del_texto),
        "paginas_usadas": elegidas,
        "paginas_total": len(paginas),
        "porcentaje": round(100 * texto_usado / max(1, texto_total)),
        # el texto exacto que el modelo leyó de cada página (para ubicar de dónde salió lo que escribió)
        "textos": {p: _recortar(mejor_de_pagina[p], presupuesto) for p in elegidas},
    }
    return "\n\n".join(partes), cobertura


def cifras_sin_respaldo(resumen: str, texto_leido: str) -> list[str]:
    """Cifras que el resumen menciona pero que no están en el texto que el modelo leyó.

    Los modelos pequeños a veces inventan números ("20 preguntas"). No se puede corregir, pero sí avisar.
    """
    def cifras(texto: str) -> set[str]:
        sin_etiquetas = re.sub(r"\[Página \d+\]", " ", texto)  # los números de página no cuentan
        return {c.strip(".,") for c in re.findall(r"\d[\d.,]*", sin_etiquetas)}
    return sorted(cifras(resumen) - cifras(texto_leido), key=lambda c: (len(c), c))


def _texto_del_documento(usuario: str, nombre: str) -> str:
    """Texto representativo del documento (para el resumen, las preguntas, etc.)."""
    return muestra_del_documento(usuario, nombre)[0]


def _armar_prompt(usuario: str, nombre: str) -> str:
    texto = _texto_del_documento(usuario, nombre)
    return (
        "Resume en español el documento, usando SOLO el texto dado.\n"
        "Escribe ÚNICAMENTE un párrafo de máximo 100 palabras, sin títulos, sin viñetas y "
        "sin repetir estas instrucciones.\n"
        "- Empieza diciendo qué tipo de documento es: si tiene preguntas con casillas u "
        "opciones para marcar, es un formulario o cuestionario; si no, puede ser carta, "
        "contrato, informe, etc.\n"
        "- Di de qué temas trata.\n"
        "- NO inventes cifras, porcentajes ni datos que no estén escritos en el texto.\n"
        "- Si es un formulario, describe sus secciones y preguntas; no afirmes cuáles "
        "respuestas se marcaron.\n"
        "- El texto puede venir de un escaneo (OCR) con errores o símbolos sueltos: ignóralos.\n"
        "- El contexto reúne un fragmento de cada página del documento (marcados [Página N]): "
        "resume el documento completo, no solo el inicio.\n\n"
        f"CONTEXTO:\n{texto}\n\n"
        "PREGUNTA: Resume el documento."
    )


def generar_resumen(usuario: str, nombre: str) -> str:
    """Resume el documento con el LLM, lo guarda en la base y lo devuelve."""
    resumen = generar_respuesta(_armar_prompt(usuario, nombre)).strip()
    db.guardar_campo(usuario, nombre, "resumen", resumen)
    return resumen


def generar_resumen_stream(usuario: str, nombre: str):
    """Igual que generar_resumen, pero entrega el texto por pedazos mientras se escribe."""
    partes = []
    for trozo in generar_respuesta_stream(_armar_prompt(usuario, nombre)):
        partes.append(trozo)
        yield trozo
    db.guardar_campo(usuario, nombre, "resumen", "".join(partes).strip())


# ------------------------------------------------------- preguntas sugeridas
# Si el modelo no devuelve 3 preguntas válidas, se completan con estas
PREGUNTAS_GENERICAS = [
    "¿De qué trata este documento?",
    "¿Cuáles son los puntos más importantes?",
    "¿Qué datos o cifras menciona?",
]


def _limpiar_preguntas(texto: str) -> list[str]:
    """Saca las preguntas de la respuesta del modelo (una por línea, sin numeración)."""
    preguntas = []
    for linea in texto.splitlines():
        linea = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s*", "", linea).strip().strip('"“”«»*').strip()
        if "?" not in linea or len(linea) < 12:
            continue
        linea = linea[: linea.index("?") + 1]  # corta lo que el modelo escriba después
        if not linea.startswith("¿"):
            linea = "¿" + linea.lstrip("¿")
        if linea not in preguntas:
            preguntas.append(linea)
    return preguntas


def _grupos_de_palabras(texto: str, n: int = 4) -> set[tuple]:
    """Conjunto de todas las secuencias de `n` palabras seguidas del texto (sin tildes)."""
    sin_tildes = "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                         if unicodedata.category(c) != "Mn")
    palabras = re.findall(r"\w+", sin_tildes)
    return {tuple(palabras[i:i + n]) for i in range(len(palabras) - n + 1)}


def _es_copia_del_texto(pregunta: str, grupos_texto: set[tuple]) -> bool:
    """True si la pregunta repite frases del propio documento (p. ej. preguntas de un formulario)."""
    grupos = _grupos_de_palabras(pregunta)
    if not grupos:
        return False
    return len(grupos & grupos_texto) / len(grupos) >= 0.35


def generar_preguntas(usuario: str, nombre: str) -> list[str]:
    """Devuelve 3 preguntas que se pueden hacer sobre el documento."""
    texto = _texto_del_documento(usuario, nombre)
    prompt = (
        "A partir del texto, escribe EXACTAMENTE 5 preguntas distintas que una persona "
        "podría hacerle a este documento y que se puedan responder con su contenido.\n"
        "- Las preguntas son SOBRE el documento, hechas por quien lo lee. Ejemplos de estilo: "
        "¿Cuál es el objetivo del documento? ¿Qué secciones o temas tiene? ¿Qué opciones se "
        "ofrecen para tal tema? ¿Quién firma o dirige el documento?\n"
        "- NO copies las preguntas que aparezcan dentro del texto (por ejemplo, las de un "
        "formulario o encuesta): esas son para quien lo llena, no para quien lo lee.\n"
        "- Una pregunta por línea, sin numeración, sin viñetas, sin respuestas ni explicaciones.\n"
        "- En español, cortas (máximo 15 palabras) y sobre datos concretos del texto.\n"
        "- El texto puede venir de un escaneo (OCR) con errores o símbolos sueltos: ignóralos.\n\n"
        f"CONTEXTO:\n{texto}\n\n"
        "PREGUNTA: Escribe las 5 preguntas."
    )
    # Se piden 5 y se descartan las que copian frases del documento; se usan las 3 primeras
    grupos_texto = _grupos_de_palabras(texto)
    candidatas = _limpiar_preguntas(generar_respuesta(prompt))
    preguntas = [p for p in candidatas if not _es_copia_del_texto(p, grupos_texto)][:3]
    for generica in PREGUNTAS_GENERICAS:  # completar si faltaron
        if len(preguntas) >= 3:
            break
        if generica not in preguntas:
            preguntas.append(generica)
    return preguntas


# ------------------------------------------------- puntos clave y glosario
def _lineas_con_vinetas(texto: str, maximo: int) -> list[str]:
    """Limpia una lista escrita por el modelo: una línea por punto, sin numeración."""
    puntos = []
    for linea in texto.splitlines():
        linea = re.sub(r"^\s*(?:[-•*]|\d+[.)])\s*", "", linea).strip()
        if len(linea) >= 8 and linea not in puntos:
            puntos.append(linea)
    return puntos[:maximo]


def generar_puntos_clave(usuario: str, nombre: str) -> str:
    """Genera de 5 a 7 puntos clave del documento, los guarda y los devuelve en texto."""
    texto = _texto_del_documento(usuario, nombre)
    prompt = (
        "Escribe en español de 5 a 7 PUNTOS CLAVE del documento, usando SOLO el texto dado.\n"
        "- Una línea por punto, cada una empezando con '- ', de máximo 20 palabras.\n"
        "- Sin títulos ni introducción, sin repetir estas instrucciones.\n"
        "- NO inventes cifras ni datos que no estén en el texto. Si es un formulario, los puntos "
        "describen sus secciones y qué información pide.\n"
        "- El texto puede venir de un escaneo (OCR) con errores o símbolos sueltos: ignóralos.\n\n"
        f"CONTEXTO:\n{texto}\n\n"
        "PREGUNTA: Escribe los puntos clave."
    )
    puntos = _lineas_con_vinetas(generar_respuesta(prompt), 7)
    if not puntos:
        raise RuntimeError("El modelo no devolvió puntos clave. Inténtalo de nuevo.")
    resultado = "\n".join(f"- {p}" for p in puntos)
    db.guardar_campo(usuario, nombre, "puntos_clave", resultado)
    return resultado


_TERMINOS_COMUNES = {
    "mision", "vision", "presupuesto", "fundacion", "informacion", "objetivo", "objetivos",
    "documento", "organizacion", "empresa", "fecha", "nombre", "pesos", "dinero", "persona",
    "personas", "datos", "proyecto", "programa",
}


def _sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                   if unicodedata.category(c) != "Mn")


def generar_glosario(usuario: str, nombre: str) -> str:
    """Genera un glosario de 5 a 8 términos del documento, lo guarda y lo devuelve en texto."""
    texto = _texto_del_documento(usuario, nombre)
    prompt = (
        "Escribe en español un GLOSARIO de 5 a 8 términos importantes que aparezcan en el texto "
        "y que convenga explicar.\n"
        "- Una línea por término con el formato exacto: Término: definición breve "
        "(máximo 20 palabras).\n"
        "- Elige términos técnicos, siglas, nombres de programas o conceptos específicos del "
        "documento. NO incluyas palabras comunes (como misión, presupuesto, fundación, "
        "información) ni el nombre de la propia organización.\n"
        "- Solo términos que estén escritos en el texto; define según el contexto del documento "
        "y, si el texto no basta para definirlo con seguridad, omítelo.\n"
        "- Sin títulos ni introducción, sin repetir estas instrucciones, sin inventar.\n"
        "- El texto puede venir de un escaneo (OCR) con errores o símbolos sueltos: ignóralos.\n\n"
        f"CONTEXTO:\n{texto}\n\n"
        "PREGUNTA: Escribe el glosario."
    )
    entradas = []
    for linea in _lineas_con_vinetas(generar_respuesta(prompt), 10):
        termino, _, definicion = linea.partition(":")
        termino, definicion = termino.strip("*_ \"“”"), definicion.strip()
        # El modelo pequeño a veces ignora la regla de las palabras comunes: se filtran aquí
        if _sin_tildes(termino) in _TERMINOS_COMUNES:
            continue
        if termino and definicion and len(termino) <= 60:
            entradas.append(f"- **{termino}**: {definicion}")
    if not entradas:
        raise RuntimeError("El modelo no devolvió un glosario. Inténtalo de nuevo.")
    resultado = "\n".join(entradas)
    db.guardar_campo(usuario, nombre, "glosario", resultado)
    return resultado
