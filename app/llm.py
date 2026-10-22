"""Genera la respuesta del chat. Modo mock u ollama.

Este archivo (junto con embeddings.py) es el ÚNICO que lee AI_MODE.
"""
import json

import httpx

import time

from app import clientes, config, db, gemini, llamacpp

# ollama = modelo local con Ollama · llamacpp = el mismo modelo local con llama.cpp · gemini = nube (API)
PROVEEDORES_CHAT = ("ollama", "llamacpp", "gemini")


def proveedor_chat() -> str:
    """Quién escribe las respuestas del chat (se elige en la interfaz)."""
    try:
        elegido = db.obtener_ajuste("proveedor_chat", "ollama")
    except Exception:
        elegido = "ollama"
    return elegido if elegido in PROVEEDORES_CHAT else "ollama"


def fijar_proveedor_chat(proveedor: str):
    if proveedor not in PROVEEDORES_CHAT:
        raise ValueError(f"El modelo de chat debe ser uno de: {', '.join(PROVEEDORES_CHAT)}.")
    db.guardar_ajuste("proveedor_chat", proveedor)

MENSAJE_SIN_OLLAMA = "No se pudo conectar con Ollama. ¿Está instalado y corriendo?"


def _respuesta_mock(prompt: str) -> str:
    """Simula la IA: muestra el contexto que venía dentro del prompt."""
    contexto = prompt
    if "CONTEXTO:" in prompt:
        contexto = prompt.split("CONTEXTO:", 1)[1].split("PREGUNTA:", 1)[0].strip()
    return (
        "[MODO PRUEBA] No hay IA conectada. "
        "Estos son los fragmentos más relevantes encontrados:\n\n" + contexto
    )


def _cuerpo_ollama(prompt: str, stream: bool) -> dict:
    return {
        "model": config.CHAT_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "keep_alive": config.KEEP_ALIVE,
        "options": {"num_ctx": config.NUM_CTX, "temperature": config.TEMPERATURE,
                    "num_predict": config.NUM_PREDICT,
                    "repeat_penalty": config.REPEAT_PENALTY},  # evita que repita lo mismo en bucle
    }


def _respuesta_ollama(prompt: str) -> str:
    try:
        r = clientes.LOCAL.post(
            f"{config.OLLAMA_URL}/api/chat",
            json=_cuerpo_ollama(prompt, stream=False),
            timeout=120,  # PC lenta: damos tiempo de sobra
        )
        r.raise_for_status()
    except httpx.ConnectError as e:
        raise RuntimeError(MENSAJE_SIN_OLLAMA) from e
    except httpx.HTTPError as e:
        raise RuntimeError(f"Ollama devolvió un error: {e}") from e
    return r.json()["message"]["content"]


def _respuesta_ollama_stream(prompt: str):
    """Va entregando la respuesta por pedazos a medida que Ollama la genera."""
    try:
        with clientes.LOCAL.stream(
            "POST",
            f"{config.OLLAMA_URL}/api/chat",
            json=_cuerpo_ollama(prompt, stream=True),
            timeout=120,
        ) as r:
            r.raise_for_status()
            for linea in r.iter_lines():
                if linea:
                    trozo = json.loads(linea).get("message", {}).get("content", "")
                    if trozo:
                        yield trozo
    except httpx.ConnectError as e:
        raise RuntimeError(MENSAJE_SIN_OLLAMA) from e
    except httpx.HTTPError as e:
        raise RuntimeError(f"Ollama devolvió un error: {e}") from e


def nombre_modelo_chat() -> str:
    """Nombre del modelo que genera las respuestas (para mostrarlo en la interfaz)."""
    if proveedor_chat() == "gemini":
        return f"{config.GEMINI_CHAT_MODEL} (nube)"
    if proveedor_chat() == "llamacpp":
        return f"{config.CHAT_MODEL} (llama.cpp)"
    return config.CHAT_MODEL if config.AI_MODE == "ollama" else "simulado (modo prueba)"


_memoria_modelos = (0.0, None)


def info_modelos() -> dict:
    """Qué modelos usa el sistema ahora y si Ollama los tiene instalados (se recuerda 15 s: se pide en cada pantalla)."""
    global _memoria_modelos
    if _memoria_modelos[1] is None or time.time() - _memoria_modelos[0] > 15 \
            or _memoria_modelos[1].get("chat") != nombre_modelo_chat():
        _memoria_modelos = (time.time(), _info_modelos())
    return _memoria_modelos[1]


def _info_modelos() -> dict:
    if config.AI_MODE != "ollama":
        return {
            "chat": "simulado (modo prueba)",
            "embeddings": "hashing trick (modo prueba)",
            "ollama_activo": None,
            "faltan": [],
        }
    try:
        r = clientes.LOCAL.get(f"{config.OLLAMA_URL}/api/tags", timeout=3)
        r.raise_for_status()
        instalados = {m["name"] for m in r.json()["models"]}
        activo = True
    except (httpx.HTTPError, KeyError, ValueError):
        instalados, activo = set(), False

    def esta(nombre: str) -> bool:
        return nombre in instalados or f"{nombre}:latest" in instalados

    necesarios = ((config.EMBED_MODEL,) if proveedor_chat() in ("gemini", "llamacpp")
                  else (config.CHAT_MODEL, config.EMBED_MODEL))
    faltan = [m for m in necesarios if activo and not esta(m)]
    return {
        "chat": nombre_modelo_chat(),
        "embeddings": config.EMBED_MODEL,
        "ollama_activo": activo,
        "faltan": faltan,
    }


def generar_respuesta(prompt: str) -> str:
    if proveedor_chat() == "gemini":
        return gemini.generar(prompt)
    if proveedor_chat() == "llamacpp":
        return llamacpp.generar(prompt)
    if config.AI_MODE == "ollama":
        return _respuesta_ollama(prompt)
    return _respuesta_mock(prompt)


def generar_respuesta_stream(prompt: str):
    """Igual que generar_respuesta, pero entrega la respuesta por pedazos."""
    if proveedor_chat() == "gemini":  # la nube responde de una vez; se entrega en pedazos pequeños
        texto = gemini.generar(prompt)
        for i in range(0, len(texto), 24):
            yield texto[i:i + 24]
        return
    if proveedor_chat() == "llamacpp":
        yield from llamacpp.generar_stream(prompt)
        return
    if config.AI_MODE == "ollama":
        yield from _respuesta_ollama_stream(prompt)
    else:
        yield _respuesta_mock(prompt)
