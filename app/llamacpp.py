"""Chat local con llama.cpp (`llama-server`), sin Ollama de por medio.

llama.cpp es el motor que ejecuta los modelos en formato GGUF; Ollama lo usa por dentro. Aquí se usa
directamente: `llama-server` (carpeta llama.cpp/) abre una API compatible con OpenAI en el puerto
LLAMACPP_PUERTO y esta clase le habla por HTTP.

Para no descargar el modelo dos veces, se usa el archivo GGUF que Ollama ya tiene de CHAT_MODEL
(o el que se indique en LLAMACPP_MODELO). El servidor se arranca solo la primera vez que se necesita.
"""
import json
import os
import subprocess
import threading
import time
from pathlib import Path

import httpx

from app import config

RAIZ = Path(__file__).resolve().parent.parent
_candado = threading.Lock()
_proceso = None


def _url() -> str:
    return f"http://127.0.0.1:{config.LLAMACPP_PUERTO}"


def ruta_servidor() -> Path:
    return RAIZ / config.LLAMACPP_CARPETA / "llama-server.exe"


def ruta_modelo() -> Path | None:
    """El archivo GGUF a usar: el indicado en LLAMACPP_MODELO o el que Ollama ya descargó de CHAT_MODEL."""
    if config.LLAMACPP_MODELO:
        ruta = Path(config.LLAMACPP_MODELO)
        return ruta if ruta.is_file() else None
    nombre, _, etiqueta = config.CHAT_MODEL.partition(":")
    base = Path(os.path.expanduser("~")) / ".ollama" / "models"
    manifiesto = base / "manifests" / "registry.ollama.ai" / "library" / nombre / (etiqueta or "latest")
    if not manifiesto.is_file():
        return None
    for capa in json.loads(manifiesto.read_text(encoding="utf-8")).get("layers", []):
        if capa.get("mediaType", "").endswith("image.model"):
            ruta = base / "blobs" / capa["digest"].replace(":", "-")
            return ruta if ruta.is_file() else None
    return None


def servidor_activo() -> bool:
    try:
        return httpx.get(f"{_url()}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def iniciar(espera: int = 120):
    """Arranca llama-server si no está corriendo y espera a que cargue el modelo."""
    global _proceso
    if servidor_activo():
        return
    with _candado:
        if servidor_activo():
            return
        servidor, modelo = ruta_servidor(), ruta_modelo()
        if not servidor.is_file():
            raise RuntimeError("Falta llama.cpp: descomprime llama-server.exe en la carpeta "
                               f"{config.LLAMACPP_CARPETA}/ (ver README).")
        if modelo is None:
            raise RuntimeError(f"No encuentro el modelo GGUF de «{config.CHAT_MODEL}». Indica uno en "
                               "LLAMACPP_MODELO o instálalo con `ollama pull`.")
        _proceso = subprocess.Popen(
            [str(servidor), "-m", str(modelo), "--host", "127.0.0.1", "--port", str(config.LLAMACPP_PUERTO),
             "-c", str(config.NUM_CTX), "-t", "4"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        limite = time.time() + espera
        while time.time() < limite:
            if servidor_activo():
                return
            if _proceso.poll() is not None:
                raise RuntimeError("llama-server se cerró al arrancar (¿poca memoria o modelo incompatible?).")
            time.sleep(1)
        raise RuntimeError("llama-server tardó demasiado en cargar el modelo.")


def detener():
    """Cierra llama-server si lo arrancó esta aplicación (libera la memoria)."""
    global _proceso
    if _proceso is not None and _proceso.poll() is None:
        _proceso.terminate()
    _proceso = None


def _cuerpo(prompt: str, stream: bool) -> dict:
    return {"messages": [{"role": "user", "content": prompt}], "stream": stream,
            "temperature": config.TEMPERATURE, "max_tokens": config.NUM_PREDICT}


def generar(prompt: str) -> str:
    iniciar()
    try:
        r = httpx.post(f"{_url()}/v1/chat/completions", json=_cuerpo(prompt, False), timeout=300)
        r.raise_for_status()
    except httpx.HTTPError as e:
        raise RuntimeError(f"llama.cpp devolvió un error: {e}") from e
    return r.json()["choices"][0]["message"]["content"]


def generar_stream(prompt: str):
    """Entrega la respuesta por pedazos a medida que llama.cpp la escribe."""
    iniciar()
    try:
        with httpx.stream("POST", f"{_url()}/v1/chat/completions", json=_cuerpo(prompt, True), timeout=300) as r:
            r.raise_for_status()
            for linea in r.iter_lines():
                if not linea.startswith("data:") or linea.strip() == "data: [DONE]":
                    continue
                datos = json.loads(linea[5:])
                trozo = (datos.get("choices") or [{}])[0].get("delta", {}).get("content")
                if trozo:
                    yield trozo
    except httpx.HTTPError as e:
        raise RuntimeError(f"llama.cpp devolvió un error: {e}") from e
