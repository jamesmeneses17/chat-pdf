"""EmbeddingGemma 2 (local): texto, código, imágenes, video y audio en un mismo espacio de 768 números.

Se usa como indica la guía "EmbeddingGemma 2: The Developer Guide": con sentence-transformers (v6.1 o
más nuevo) y el modelo `google/embeddinggemma-2`, que se descarga una sola vez desde Hugging Face
(Apache 2.0, sin clave). Corre en el computador, sin internet ni APIs.

Tiene dos formas de cargarse, que comparten el mismo espacio de vectores (así una pregunta calculada con
la versión de solo texto se compara bien con imágenes calculadas con la versión completa):
  - solo texto (270M parámetros): la que se usa para los fragmentos de los PDFs y las preguntas.
  - completa (740M): imágenes, video y audio. Solo se carga cuando hace falta (gasta más memoria).

Los modelos se cargan en float32: en CPUs sin soporte de bfloat16 (como un Core i5 de 8.ª gen) el
formato por defecto era ~8 veces más lento.
"""
import io
import threading

import numpy as np

from app import config

_modelos: dict[str, object] = {}
_candado = threading.Lock()


def _cargar(completo: bool):
    """Carga (una sola vez) la versión de solo texto o la completa."""
    clave = "completo" if completo else "texto"
    with _candado:
        if clave not in _modelos:
            try:
                import torch
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise RuntimeError("Falta instalar sentence-transformers y torch "
                                   "(pip install -r requirements.txt).") from e
            try:
                kwargs = {} if completo else {"config_kwargs": {"vision_config": None, "audio_config": None}}
                modelo = SentenceTransformer(config.GEMMA2_MODEL, **kwargs)
            except Exception as e:
                raise RuntimeError("No se pudo cargar EmbeddingGemma 2. La primera vez necesita internet "
                                   "para descargar el modelo (unos GB).") from e
            modelo.to(torch.float32)
            _modelos[clave] = modelo
    return _modelos[clave]


def _lista(vectores) -> list:
    return np.asarray(vectores, dtype=float).tolist()


def embedding_consulta(texto: str) -> list[float]:
    """Vector de una pregunta (se compara con fragmentos, imágenes, audios y videos)."""
    return _lista(_cargar(False).encode(texto, prompt_name="SearchQuery", normalize_embeddings=True))


def embeddings_documentos(textos: list[str], lote: int = 8) -> list[list[float]]:
    """Vectores de los fragmentos de un documento."""
    if not textos:
        return []
    return _lista(_cargar(False).encode(textos, prompt_name="Document", batch_size=lote,
                                        normalize_embeddings=True))


def embedding_imagen(png: bytes) -> list[float]:
    """Vector de una imagen (PNG/JPG en bytes). Usa la versión completa del modelo."""
    from PIL import Image
    imagen = Image.open(io.BytesIO(png)).convert("RGB")
    return _lista(_cargar(True).encode({"image": imagen}, normalize_embeddings=True))


def embedding_audio(ruta: str) -> list[float]:
    """Vector de un audio (archivo WAV/MP3...; la guía recomienda 16 kHz mono)."""
    return _lista(_cargar(True).encode({"audio": ruta}, normalize_embeddings=True))


def embedding_video(fotogramas: list) -> list[float]:
    """Vector de un clip de video: lista de fotogramas (arreglos RGB), 1 por segundo.

    La guía pasa un archivo MP4, pero decodificarlo requiere torchcodec + ffmpeg (difícil en Windows);
    PyAV ya saca los fotogramas, y el modelo también acepta una lista de imágenes.
    """
    from PIL import Image
    return _lista(_cargar(True).encode({"video": [Image.fromarray(f) for f in fotogramas]},
                                       normalize_embeddings=True))
