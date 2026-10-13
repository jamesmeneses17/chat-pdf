"""Convierte texto en un vector (embedding). Modo mock u ollama.

Este archivo (junto con llm.py) es el ÚNICO que lee AI_MODE.
"""
import hashlib
import math
import re
import unicodedata

import httpx

from app import config

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


def _embeddings_ollama(textos: list[str]) -> list[list[float]]:
    """Ollama acepta varios textos en una sola llamada (más rápido que uno por uno)."""
    try:
        r = httpx.post(
            f"{config.OLLAMA_URL}/api/embed",
            json={"model": config.EMBED_MODEL, "input": textos,
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


def obtener_embedding(texto: str) -> list[float]:
    """Devuelve siempre una lista de EMBED_DIM números."""
    return obtener_embeddings([texto])[0]


def obtener_embeddings(textos: list[str], lote: int = 16) -> list[list[float]]:
    """Embeddings de varios textos. En modo ollama los envía en lotes."""
    if config.AI_MODE != "ollama":
        return [_embedding_mock(t) for t in textos]
    vectores = []
    for i in range(0, len(textos), lote):
        vectores.extend(_embeddings_ollama(textos[i:i + lote]))
    return vectores
