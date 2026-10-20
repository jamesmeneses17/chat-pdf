"""Audios y videos: sacarlos de los PDFs (archivos adjuntos) y partirlos en segmentos cortos para los embeddings.

EmbeddingGemma 2 convierte audio y video directamente en vectores (sin transcribir), pero cada entrada
tiene un tope de tamaño; por eso cada archivo se parte en segmentos de SEGUNDOS_POR_SEGMENTO. Así también se
puede decir en qué segundo empieza lo que se encontró.

Según la guía del modelo, el audio debe ir a 16 kHz y mono, y el video se muestrea a 1 fotograma por segundo.
Todo se hace con PyAV (ya incluye los códecs): no hace falta instalar ffmpeg.
"""
import io
import os
import tempfile
import wave

import numpy as np
import pymupdf

from app import config

EXT_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".opus"}
EXT_VIDEO = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v"}
MIMES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".flac": "audio/flac",
         ".aac": "audio/aac", ".wma": "audio/x-ms-wma", ".opus": "audio/ogg", ".mp4": "video/mp4",
         ".webm": "video/webm", ".mov": "video/quicktime", ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
         ".m4v": "video/mp4"}
TASA_AUDIO = 16000  # Hz, lo que pide el modelo
ANCHO_VIDEO = 320  # px: los fotogramas se reducen (el modelo no necesita más y el cálculo es mucho más rápido)


def tipo_de_archivo(nombre: str) -> str | None:
    """'audio', 'video' o None según la extensión."""
    ext = os.path.splitext(nombre.lower())[1]
    return "audio" if ext in EXT_AUDIO else "video" if ext in EXT_VIDEO else None


def mime_de(nombre: str) -> str:
    return MIMES.get(os.path.splitext(nombre.lower())[1], "application/octet-stream")


def adjuntos_del_pdf(contenido_pdf: bytes) -> list[dict]:
    """Archivos de audio y video adjuntos dentro de un PDF: [{nombre, tipo, mime, datos}]."""
    salida = []
    try:
        with pymupdf.open(stream=contenido_pdf, filetype="pdf") as doc:
            for nombre in doc.embfile_names():
                info = doc.embfile_info(nombre)
                archivo = info.get("filename") or info.get("name") or nombre
                tipo = tipo_de_archivo(archivo)
                if tipo:
                    salida.append({"nombre": archivo, "tipo": tipo, "mime": mime_de(archivo),
                                   "datos": doc.embfile_get(nombre)})
    except Exception:
        return []  # un PDF raro sin adjuntos legibles: simplemente no hay medios
    return salida


# ------------------------------------------------------------------------------ audio
def segmentos_de_audio(datos: bytes, directorio: str, segundos: int | None = None) -> list[dict]:
    """Parte un audio en WAV de `segundos` (16 kHz, mono). Devuelve [{inicio, fin, ruta}]."""
    import av
    segundos = segundos or config.SEGUNDOS_POR_SEGMENTO
    muestras = []
    with av.open(io.BytesIO(datos)) as contenedor:
        flujo = next((f for f in contenedor.streams if f.type == "audio"), None)
        if flujo is None:
            return []
        remuestreo = av.AudioResampler(format="s16", layout="mono", rate=TASA_AUDIO)
        for trama in contenedor.decode(flujo):
            for r in remuestreo.resample(trama):
                muestras.append(r.to_ndarray().reshape(-1))
        for r in remuestreo.resample(None) or []:
            muestras.append(r.to_ndarray().reshape(-1))
    if not muestras:
        return []
    señal = np.concatenate(muestras).astype(np.int16)
    por_segmento = segundos * TASA_AUDIO
    salida = []
    for n, inicio in enumerate(range(0, len(señal), por_segmento)):
        if n >= config.MEDIOS_MAX_SEGMENTOS:
            break
        trozo = señal[inicio:inicio + por_segmento]
        if len(trozo) < TASA_AUDIO // 2:  # menos de medio segundo: no dice nada
            continue
        ruta = os.path.join(directorio, f"audio_{n}.wav")
        with wave.open(ruta, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(TASA_AUDIO)
            w.writeframes(trozo.tobytes())
        salida.append({"inicio": inicio / TASA_AUDIO, "fin": (inicio + len(trozo)) / TASA_AUDIO, "ruta": ruta})
    return salida


# ------------------------------------------------------------------------------ video
def segmentos_de_video(datos: bytes, directorio: str, segundos: int | None = None) -> list[dict]:
    """Parte un video en clips de `segundos` con 1 fotograma cada VIDEO_SEGUNDOS_ENTRE_FOTOS (320 px de ancho).

    Devuelve [{inicio, fin, fotos, png, ancho, alto}]: `fotos` son los fotogramas (el modelo los recibe así,
    sin pasar por un archivo) y `png` la miniatura del clip.
    """
    import av
    from PIL import Image
    segundos = segundos or config.SEGUNDOS_POR_SEGMENTO
    paso = max(1, config.VIDEO_SEGUNDOS_ENTRE_FOTOS)
    por_segundo: dict[int, np.ndarray] = {}
    with av.open(io.BytesIO(datos)) as contenedor:
        flujo = next((f for f in contenedor.streams if f.type == "video"), None)
        if flujo is None:
            return []
        for trama in contenedor.decode(flujo):
            if trama.time is None:
                continue
            s = int(trama.time)
            if s in por_segundo or s % paso:
                continue  # solo un fotograma cada `paso` segundos (cada fotograma tarda en el CPU)
            ancho = ANCHO_VIDEO - ANCHO_VIDEO % 2
            alto = int(trama.height * ancho / trama.width)
            alto -= alto % 2
            por_segundo[s] = trama.to_ndarray(width=ancho, height=alto, format="rgb24")
    if not por_segundo:
        return []
    ultimo = max(por_segundo)
    salida = []
    for n, inicio in enumerate(range(0, ultimo + 1, segundos)):
        if n >= config.MEDIOS_MAX_SEGMENTOS:
            break
        fotos = [por_segundo[t] for t in sorted(por_segundo) if inicio <= t < inicio + segundos]
        if not fotos:
            continue
        miniatura = io.BytesIO()
        Image.fromarray(fotos[0]).save(miniatura, format="PNG")
        salida.append({"inicio": float(inicio), "fin": float(min(inicio + segundos, ultimo + paso)), "fotos": fotos,
                       "png": miniatura.getvalue(), "ancho": fotos[0].shape[1], "alto": fotos[0].shape[0]})
    return salida


def con_directorio_temporal():
    """Carpeta temporal para los segmentos (se borra sola al terminar)."""
    return tempfile.TemporaryDirectory(prefix="medios_", ignore_cleanup_errors=True)
