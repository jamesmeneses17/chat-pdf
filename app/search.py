"""Búsqueda híbrida de fragmentos: vectores + palabras clave, con reordenamiento final.

1. Por significado: los fragmentos más cercanos a la pregunta en pgvector.
2. Por palabras: búsqueda de texto completo de PostgreSQL (en español) con las palabras
   importantes de la pregunta. Encuentra cosas que los vectores confunden (nombres, cifras,
   términos exactos), sobre todo en documentos donde todas las páginas "suenan parecido".
3. Se juntan los candidatos de las dos búsquedas y se reordenan con un puntaje final:
   similitud de vectores + palabras de la pregunta que contiene el fragmento
   + posición en cada búsqueda (fusión RRF).

Solo se busca entre los documentos del usuario (y, si se indica, entre los elegidos).
"""
import re
import time
import unicodedata

import numpy as np

from app import config, db
from app.embeddings import obtener_embedding

# Palabras que no ayudan a distinguir un fragmento de otro
_PALABRAS_VACIAS = {
    "cual", "cuales", "cuanto", "cuantos", "cuanta", "cuantas", "como", "donde", "cuando",
    "quien", "quienes", "porque", "para", "sobre", "entre", "segun", "desde", "hasta",
    "tiene", "tienen", "hay", "puede", "pueden", "este", "esta", "estos", "estas", "ese",
    "esa", "ofrecen", "elegir", "preguntan", "documento", "ficha", "archivo", "pdf",
    "dice", "habla", "trata", "menciona", "mencionan", "existen", "existe", "siguientes",
}
_K_RRF = 60  # constante de la fusión RRF (valor estándar)

# La gente pregunta con otras palabras que las del documento ("cuántos años tiene" vs "Edad:").
# Para los datos típicos de formularios y fichas se agregan las palabras del documento.
_SINONIMOS = [
    (r"cu[aá]ntos?\s+a[ñn]os|qu[eé]\s+edad|\bedad\b", "edad"),
    (r"cu[aá]nto\s+(gana|cobra|recibe)|sueldo|salario|ingresos?", "ingresos"),
    (r"ingresos?\s+mensual\w*|mensual\w*", "mes dinero obtiene"),
    (r"d[oó]nde\s+(vive|reside|queda)|direcci[oó]n|domicilio", "vivienda residencia direccion lugar"),
    (r"c[oó]mo\s+se\s+llama|cu[aá]l\s+es\s+su\s+nombre|nombre\s+completo", "nombre"),
    (r"a\s+qu[eé]\s+se\s+dedica|en\s+qu[eé]\s+trabaja|ocupaci[oó]n|profesi[oó]n", "actividad trabajo ocupacion"),
    (r"estado\s+civil|casad[oa]|solter[oa]", "estado civil"),
    (r"g[eé]nero|\bsexo\b|hombre\s+o\s+mujer", "sexo"),
    (r"cu[aá]ntos\s+hijos|n[uú]mero\s+de\s+hijos", "hijos"),
    (r"tel[eé]fono|celular|contacto", "telefono celular"),
    (r"cu[aá]nto\s+tiempo\s+(lleva|vive|trabaja)", "tiempo"),
    (r"nivel\s+educativo|estudios|escolaridad", "nivel educativo"),
]


# Preguntas sobre qué opciones existen ("¿qué rangos se ofrecen?"), no sobre lo que eligió alguien
_PREGUNTA_DE_OPCIONES = re.compile(
    r"\b(opciones|rangos?|posibles|ofrec\w+|alternativas|categor[ií]as|se\s+pregunta\w*|"
    r"qu[eé]\s+tipos?)\b", re.I)


def es_pregunta_de_opciones(pregunta: str) -> bool:
    return bool(_PREGUNTA_DE_OPCIONES.search(pregunta))


def expandir_pregunta(pregunta: str) -> str:
    """La pregunta más las palabras del documento que suelen nombrar lo que se pregunta."""
    ya_dichas = set(re.findall(r"\w+", _sin_tildes(pregunta)))
    extra = []
    for patron, palabras in _SINONIMOS:
        if re.search(patron, pregunta, re.I):
            # solo las palabras que la pregunta todavía no tiene (repetirlas distorsiona la búsqueda)
            extra += [p for p in palabras.split() if _sin_tildes(p) not in ya_dichas and p not in extra]
    return f"{pregunta} {' '.join(extra)}" if extra else pregunta


def _sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                   if unicodedata.category(c) != "Mn")


def _palabras_clave(pregunta: str) -> list[str]:
    palabras = re.findall(r"\w+", _sin_tildes(pregunta))
    return [p for p in palabras if len(p) >= 4 and p not in _PALABRAS_VACIAS]


def _coincidencia(palabras: list[str], contenido: str) -> float:
    """Fracción de palabras clave de la pregunta que aparecen en el fragmento (0 a 1).

    Se comparan solo las primeras 5 letras, para tolerar plurales y errores de OCR al final.
    """
    if not palabras:
        return 0.0
    texto = _sin_tildes(contenido)
    return sum(1 for p in palabras if p[:5] in texto) / len(palabras)


def _rrf(posicion: int) -> float:
    """Puntos por quedar en cierta posición (1 = primero) de una búsqueda."""
    return 1 / (_K_RRF + posicion)


def buscar_similares(pregunta: str, usuario: str, documentos: list[str] | None = None,
                     tiempos: dict | None = None) -> list[dict]:
    """Devuelve los TOP_K fragmentos más relevantes para la pregunta.

    `documentos`: nombres entre los que buscar (None = todos los del usuario).
    Si se pasa un diccionario `tiempos`, se llena con los milisegundos de cada etapa.
    """
    if documentos is not None and not documentos:
        return []  # el usuario no eligió ningún documento

    solo_texto = es_pregunta_de_opciones(pregunta)  # sin los bloques de "respuestas marcadas"
    pregunta = expandir_pregunta(pregunta)  # solo para buscar; el usuario ve su pregunta original
    t0 = time.perf_counter()
    vector = np.array(obtener_embedding(pregunta))
    t1 = time.perf_counter()

    palabras = _palabras_clave(pregunta)
    consulta_texto = " | ".join(palabras)  # cualquiera de las palabras (OR)
    n = max(config.BUSQUEDA_CANDIDATOS, config.TOP_K)
    filtro, extra = "usuario = %s", [usuario]
    if solo_texto:
        filtro += " AND contenido NOT LIKE 'RESPUESTAS MARCADAS%%' AND contenido NOT LIKE '%%→%%'"
    if documentos:
        filtro += " AND documento = ANY(%s)"
        extra.append(documentos)

    with db.conectar() as conn:
        # 1. Por significado. <=> es la distancia coseno: menor distancia = más parecido
        por_vectores = conn.execute(
            f"""
            SELECT id, documento, pagina, contenido, embedding <=> %s AS distancia
            FROM chunks WHERE {filtro}
            ORDER BY embedding <=> %s LIMIT %s
            """,
            (vector, *extra, vector, n),
        ).fetchall()

        # 2. Por palabras (texto completo en español)
        por_palabras = []
        if consulta_texto:
            por_palabras = conn.execute(
                f"""
                SELECT id, documento, pagina, contenido,
                       ts_rank_cd(to_tsvector('spanish', contenido), q) AS relevancia
                FROM chunks, to_tsquery('spanish', %s) q
                WHERE {filtro} AND to_tsvector('spanish', contenido) @@ q
                ORDER BY relevancia DESC LIMIT %s
                """,
                (consulta_texto, *extra, n),
            ).fetchall()

        # Los hallados solo por palabras necesitan su distancia de vectores para el puntaje
        distancias = {f[0]: float(f[4]) for f in por_vectores}
        faltan = [f[0] for f in por_palabras if f[0] not in distancias]
        if faltan:
            for id_, distancia in conn.execute(
                    "SELECT id, embedding <=> %s FROM chunks WHERE id = ANY(%s)", (vector, faltan)):
                distancias[id_] = float(distancia)

    # 3. Reordenamiento final de todos los candidatos
    candidatos = {f[0]: f[1:4] for f in [*por_vectores, *por_palabras]}  # id -> (doc, pág, texto)
    rrf = {}
    for lista in (por_vectores, por_palabras):
        for posicion, fila in enumerate(lista, 1):
            rrf[fila[0]] = rrf.get(fila[0], 0) + _rrf(posicion)
    relevancia = {f[0]: float(f[4]) for f in por_palabras}
    max_relevancia = max(relevancia.values(), default=1) or 1
    max_rrf = max(rrf.values(), default=1)

    ids_por_vectores = {f[0] for f in por_vectores}
    ids_por_palabras = {f[0] for f in por_palabras}
    fragmentos = []
    for id_, (documento, pagina, contenido) in candidatos.items():
        coincidencia = _coincidencia(palabras, contenido)
        relevancia_texto = relevancia.get(id_, 0) / max_relevancia
        puntaje = (
            (1 - distancias[id_])  # similitud por significado
            + config.PESO_PALABRAS * coincidencia
            + config.PESO_FTS * relevancia_texto
            + config.PESO_RRF * rrf[id_] / max_rrf
        )
        hallado_por = ("significado y palabras" if id_ in ids_por_vectores and id_ in ids_por_palabras
                       else "significado" if id_ in ids_por_vectores else "palabras")
        fragmentos.append({"id": id_, "documento": documento, "pagina": pagina, "contenido": contenido,
                           "distancia": distancias[id_], "similitud": 1 - distancias[id_],
                           "coincidencia": coincidencia, "relevancia_texto": relevancia_texto,
                           "hallado_por": hallado_por, "puntaje": puntaje})
    fragmentos.sort(key=lambda f: f["puntaje"], reverse=True)
    elegidos = fragmentos[:config.TOP_K]

    # Dónde está cada fragmento elegido dentro de su documento ("fragmento 12 de 42") y cuántos
    # fragmentos había en total entre los documentos donde se buscó
    if elegidos:
        with db.conectar() as conn:
            posiciones = {i: (pos, total) for i, pos, total in conn.execute(
                """
                SELECT id, pos, total FROM (
                    SELECT id, ROW_NUMBER() OVER (PARTITION BY documento ORDER BY id) AS pos,
                           COUNT(*) OVER (PARTITION BY documento) AS total
                    FROM chunks WHERE usuario = %s
                ) t WHERE id = ANY(%s)
                """,
                (usuario, [f["id"] for f in elegidos]))}
            total_buscado = conn.execute(f"SELECT COUNT(*) FROM chunks WHERE {filtro}", extra).fetchone()[0]
        for f in elegidos:
            f["posicion"], f["total_documento"] = posiciones.get(f["id"], (None, None))
            f["total_buscado"] = total_buscado
    t2 = time.perf_counter()

    if tiempos is not None:
        tiempos["embedding_ms"] = round((t1 - t0) * 1000)  # pregunta -> vector (Ollama)
        tiempos["busqueda_db_ms"] = round((t2 - t1) * 1000)  # búsquedas + reordenamiento
    return elegidos
