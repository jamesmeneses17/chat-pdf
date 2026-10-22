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
from fractions import Fraction

import numpy as np
import pymupdf

from app import config

EXT_AUDIO = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".wma", ".opus", ".mp2", ".mpga", ".aiff", ".aif", ".amr"}
EXT_VIDEO = {".mp4", ".webm", ".mov", ".mkv", ".avi", ".m4v", ".mpeg", ".mpg", ".wmv", ".flv", ".3gp", ".ogv"}
MIMES = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4", ".ogg": "audio/ogg", ".flac": "audio/flac",
         ".aac": "audio/aac", ".wma": "audio/x-ms-wma", ".opus": "audio/ogg", ".mp2": "audio/mpeg",
         ".mpga": "audio/mpeg", ".aiff": "audio/aiff", ".aif": "audio/aiff", ".amr": "audio/amr",
         ".mp4": "video/mp4", ".webm": "video/webm", ".mov": "video/quicktime", ".mkv": "video/x-matroska",
         ".avi": "video/x-msvideo", ".m4v": "video/mp4", ".mpeg": "video/mpeg", ".mpg": "video/mpeg",
         ".wmv": "video/x-ms-wmv", ".flv": "video/x-flv", ".3gp": "video/3gpp", ".ogv": "video/ogg"}
# Formatos que el navegador reproduce solo. Los demás (MPEG, AVI, WMV...) se leen igual para indexarlos, pero
# se guarda además una copia en MP4 / WAV para poder reproducirlos en el visor.
REPRODUCIBLES = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".aac", ".opus", ".mp4", ".webm", ".m4v", ".mov", ".ogv"}
TASA_AUDIO = 16000  # Hz, lo que pide el modelo
ANCHO_VIDEO = 320  # px: los fotogramas se reducen (el modelo no necesita más y el cálculo es mucho más rápido)


def tipo_de_archivo(nombre: str) -> str | None:
    """'audio', 'video' o None según la extensión."""
    ext = os.path.splitext(nombre.lower())[1]
    return "audio" if ext in EXT_AUDIO else "video" if ext in EXT_VIDEO else None


def mime_de(nombre: str) -> str:
    return MIMES.get(os.path.splitext(nombre.lower())[1], "application/octet-stream")


def version_reproducible(adjunto: dict) -> dict:
    """Si el navegador no puede reproducir el formato (p. ej. MPEG), devuelve el adjunto con una copia en MP4 / WAV.

    El nombre no cambia (así se sigue llamando clip.mpeg); solo cambian los bytes guardados y su tipo MIME.
    Si la conversión falla, se deja el original (se indexa igual, solo que el navegador quizá no lo reproduzca).
    """
    if os.path.splitext(adjunto["nombre"].lower())[1] in REPRODUCIBLES:
        return adjunto
    try:
        if adjunto["tipo"] == "video":
            return {**adjunto, "datos": _video_a_mp4(adjunto["datos"]), "mime": "video/mp4"}
        return {**adjunto, "datos": _audio_a_wav(adjunto["datos"]), "mime": "audio/wav"}
    except Exception:
        return adjunto


def _audio_a_wav(datos: bytes) -> bytes:
    """Todo el audio en un WAV (16 kHz, mono) que cualquier navegador reproduce."""
    import av
    muestras = []
    with av.open(io.BytesIO(datos)) as contenedor:
        flujo = next(f for f in contenedor.streams if f.type == "audio")
        remuestreo = av.AudioResampler(format="s16", layout="mono", rate=TASA_AUDIO)
        for trama in contenedor.decode(flujo):
            for r in remuestreo.resample(trama):
                muestras.append(r.to_ndarray().reshape(-1))
        for r in remuestreo.resample(None) or []:
            muestras.append(r.to_ndarray().reshape(-1))
    salida = io.BytesIO()
    with wave.open(salida, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TASA_AUDIO)
        w.writeframes(np.concatenate(muestras).astype(np.int16).tobytes())
    return salida.getvalue()


def _video_a_mp4(datos: bytes, ancho_max: int = 640) -> bytes:
    """Convierte un video (MPEG, AVI, WMV...) a MP4 con H.264 + AAC, que todos los navegadores reproducen."""
    import av
    destino = io.BytesIO()
    with av.open(io.BytesIO(datos)) as entrada, av.open(destino, "w", format="mp4") as salida:
        v_in = next(f for f in entrada.streams if f.type == "video")
        a_in = next((f for f in entrada.streams if f.type == "audio"), None)
        ancho = min(v_in.codec_context.width, ancho_max)
        ancho -= ancho % 2
        alto = int(v_in.codec_context.height * ancho / v_in.codec_context.width)
        alto -= alto % 2
        fps = Fraction(v_in.average_rate or 25)
        tiempo_por_cuadro = 1 / fps  # tiempo de cada fotograma (el del stream aún no está fijado al empezar)
        v_out = salida.add_stream("libx264", rate=fps)
        v_out.width, v_out.height, v_out.pix_fmt = ancho, alto, "yuv420p"
        v_out.options = {"crf": "28", "preset": "veryfast", "bf": "0"}
        a_out, remuestreo = None, None
        if a_in is not None:
            a_out = salida.add_stream("aac", rate=44100)
            remuestreo = av.AudioResampler(format="fltp", layout="stereo", rate=44100)
        n = 0
        for trama in entrada.decode(*[s for s in (v_in, a_in) if s is not None]):
            if trama.__class__.__name__ == "VideoFrame":
                nueva = trama.reformat(width=ancho, height=alto, format="yuv420p")
                nueva.pts, nueva.time_base = n, tiempo_por_cuadro
                n += 1
                for paquete in v_out.encode(nueva):
                    salida.mux(paquete)
            elif a_out is not None:
                for r in remuestreo.resample(trama):
                    for paquete in a_out.encode(r):
                        salida.mux(paquete)
        for paquete in v_out.encode():
            salida.mux(paquete)
        if a_out is not None:
            for paquete in a_out.encode():
                salida.mux(paquete)
    return destino.getvalue()


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
