"""Convierte texto en un vector (embedding). Modo mock u ollama.

Este archivo (junto con llm.py) es el ÚNICO que lee AI_MODE.
"""
import hashlib
import math
import re
import unicodedata

import httpx

from app import clientes, config, db, gemini, gemma2

MENSAJE_SIN_OLLAMA = "No se pudo conectar con Ollama. ¿Está instalado y corriendo?"


def _sin_tildes(texto: str) -> str:
    descompuesto = unicodedata.normalize("NFD", texto)
    return "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")


def _embedding_mock(texto: str) -> list[float]:
    """'Hashing trick': cada palabra suma 1 en una posición fija del vector.

    Textos con palabras en común quedan cercanos, sin necesidad de IA real.
    """
    vector = [0.0] * config.EMBED_DIM
    palabras = re.findall(r"\w+", _sin_tildes(texto.lower()))
    for palabra in palabras:
        # md5 es estable entre ejecuciones (el hash() de Python no lo es)
        h = int(hashlib.md5(palabra.encode("utf-8")).hexdigest(), 16)
        vector[h % config.EMBED_DIM] += 1.0

    norma = math.sqrt(sum(x * x for x in vector))
    if norma > 0:
        vector = [x / norma for x in vector]
    return vector


def _embeddings_ollama(textos: list[str], modelo: str | None = None) -> list[list[float]]:
    """Ollama acepta varios textos en una sola llamada (más rápido que uno por uno)."""
    try:
        r = clientes.LOCAL.post(
            f"{config.OLLAMA_URL}/api/embed",
            json={"model": modelo or config.EMBED_MODEL, "input": textos,
                  "keep_alive": config.KEEP_ALIVE},
            timeout=120,
        )
        r.raise_for_status()
    except httpx.ConnectError as e:
        raise RuntimeError(MENSAJE_SIN_OLLAMA) from e
    except httpx.HTTPError as e:
        raise RuntimeError(f"Ollama devolvió un error: {e}") from e

    vectores = r.json()["embeddings"]
    for vector in vectores:
        if len(vector) != config.EMBED_DIM:
            raise RuntimeError(
                f"La dimensión del modelo ({len(vector)}) no coincide con EMBED_DIM "
                f"({config.EMBED_DIM}) de la base de datos. Ajusta EMBED_DIM en el .env "
                "y recrea la tabla."
            )
    return vectores


# ollama = nomic-embed-text (local) · gemma = EmbeddingGemma 1 (local, por Ollama)
# gemma2 = EmbeddingGemma 2 (local, multimodal) · gemini = nube
PROVEEDORES = ("ollama", "gemma", "gemma2", "gemini")


def proveedor_actual() -> str:
    """Con qué modelo se calculan los vectores de los PDFs nuevos (se elige en la interfaz)."""
    try:
        elegido = db.obtener_ajuste("proveedor_embeddings", "ollama")
    except Exception:
        elegido = "ollama"
    return elegido if elegido in PROVEEDORES else "ollama"


def fijar_proveedor(proveedor: str):
    if proveedor not in PROVEEDORES:
        raise ValueError(f"El proveedor debe ser uno de: {', '.join(PROVEEDORES)}.")
    db.guardar_ajuste("proveedor_embeddings", proveedor)


MEDIOS = ("gemma2", "gemini")  # gemma2 = imágenes, audio y video (local) · gemini = solo imágenes (nube)


def proveedor_medios() -> str:
    """Con qué modelo se calculan los vectores de imágenes, audios y videos nuevos."""
    try:
        elegido = db.obtener_ajuste("proveedor_medios", "gemma2")
    except Exception:
        elegido = "gemma2"
    return elegido if elegido in MEDIOS else "gemma2"


def fijar_proveedor_medios(proveedor: str):
    if proveedor not in MEDIOS:
        raise ValueError(f"El proveedor de imágenes, audio y video debe ser uno de: {', '.join(MEDIOS)}.")
    db.guardar_ajuste("proveedor_medios", proveedor)


def obtener_embedding(texto: str, proveedor: str = "ollama") -> list[float]:
    """Vector de una pregunta. Debe calcularse con el mismo modelo que los fragmentos donde se busca."""
    if proveedor == "gemini":
        return gemini.embedding_consulta(texto)
    if proveedor == "gemma2":
        return gemma2.embedding_consulta(texto)
    if proveedor == "gemma":
        return _embeddings_ollama([f"task: search result | query: {texto}"], config.EMBED_GEMMA_MODEL)[0]
    return obtener_embeddings([texto])[0]


def obtener_embeddings(textos: list[str], lote: int = 16, proveedor: str = "ollama") -> list[list[float]]:
    """Embeddings de varios fragmentos. En modo ollama los envía en lotes."""
    if proveedor == "gemini":
        return gemini.embeddings_documentos(textos)
    if proveedor == "gemma2":
        return gemma2.embeddings_documentos(textos)
    if proveedor == "gemma":  # EmbeddingGemma pide este formato para los fragmentos de documento
        vectores = []
        for i in range(0, len(textos), lote):
            vectores.extend(_embeddings_ollama([f"title: none | text: {t}" for t in textos[i:i + lote]],
                                               config.EMBED_GEMMA_MODEL))
        return vectores
    if config.AI_MODE != "ollama":
        return [_embedding_mock(t) for t in textos]
    vectores = []
    for i in range(0, len(textos), lote):
        vectores.extend(_embeddings_ollama(textos[i:i + lote]))
    return vectores
