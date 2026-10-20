"""Transcripción local de audio con Whisper: convierte lo que se dice en texto.

EmbeddingGemma 2 sirve para ENCONTRAR audios y videos, pero el modelo que escribe las respuestas solo lee
texto. Por eso, al subir un audio (o un video) se transcribe su sonido: ese texto se parte en fragmentos con
su minuto de inicio y entra al mismo flujo que los PDFs (búsqueda, chat con citas, resumen, glosario...).

Whisper corre en el computador (modelo `openai/whisper-base` por defecto: ~0,7 s de cálculo por cada
segundo de audio en un Core i5 de 8.ª gen). `openai/whisper-small` es más exacto pero ~4 veces más lento.
"""
import threading

import numpy as np

from app import config

_asr = None
_candado = threading.Lock()


def _cargar():
    global _asr
    with _candado:
        if _asr is None:
            try:
                import torch
                from transformers import pipeline
            except ImportError as e:
                raise RuntimeError("Falta instalar transformers y torch (pip install -r requirements.txt).") from e
            try:
                _asr = pipeline("automatic-speech-recognition", model=config.WHISPER_MODEL,
                                torch_dtype=torch.float32, device="cpu")
            except Exception as e:
                raise RuntimeError("No se pudo cargar Whisper. La primera vez necesita internet para "
                                   "descargar el modelo.") from e
    return _asr


def texto_de_wav(ruta: str) -> str:
    """Lo que se dice en un WAV (16 kHz mono, de hasta unos 30 s). Devuelve '' si no hay voz."""
    import soundfile as sf
    audio, tasa = sf.read(ruta, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if len(audio) < tasa // 2 or float(np.abs(audio).max()) < 0.01:  # medio segundo o silencio
        return ""
    r = _cargar()({"raw": audio, "sampling_rate": tasa}, return_timestamps=True,
                  generate_kwargs={"language": config.WHISPER_IDIOMA, "task": "transcribe"})
    texto = " ".join(str(r.get("text", "")).split())
    return texto if len(texto) >= 3 else ""
