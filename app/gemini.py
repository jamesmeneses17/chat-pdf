"""Embeddings de Gemini (nube) para buscar dentro de las imágenes de los PDFs.

`gemini-embedding-2` convierte texto e imágenes en vectores del MISMO espacio, así que se puede
buscar una imagen con una frase ("un perro"). Solo se usa para las imágenes: el texto de los
documentos sigue con Ollama (local).

Hasta 3 claves (GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3 en el .env), usadas por orden:
si una llega a su límite de uso o es rechazada, se pasa sola a la siguiente y se avisa. Cada clave
se puede encender o apagar desde la aplicación. Las claves nunca se muestran ni se escriben en
los mensajes de error.
"""
import base64
import time
from datetime import datetime

import httpx
import numpy as np

from app import config, db

URL = "https://generativelanguage.googleapis.com/v1beta/models/{modelo}:embedContent"
PAUSA_LIMITE = 65  # segundos que descansa una clave que llegó a su límite por minuto
PAUSA_LARGA = 3600  # si llega al límite 3 veces seguidas, es probable que sea el cupo del día

_estado: dict[int, dict] = {}  # estado de cada clave mientras corre la API
_activas: dict[int, bool] | None = None  # qué claves están encendidas (se guarda en la base)
_guardadas: dict[int, str] | None = None  # claves pegadas desde la interfaz (tienen prioridad sobre el .env)
_avisos: list[str] = []  # lo que pasó con las claves, para decírselo al usuario


# ------------------------------------------------------------------ estado de las claves
def _claves() -> list[tuple[int, str]]:
    """Las claves configuradas (de la interfaz o del .env), como (número, clave)."""
    global _guardadas
    if _guardadas is None:
        try:
            _guardadas = db.claves_gemini_guardadas()
        except Exception:
            _guardadas = {}
    return [(n, _guardadas.get(n) or k) for n, k in config.GEMINI_CLAVES if _guardadas.get(n) or k]


def guardar_clave(numero: int, clave: str):
    """Guarda la clave que se pegó en la interfaz (vacía = borrarla; entonces vale la del .env, si hay)."""
    global _guardadas
    if numero not in (1, 2, 3):
        raise ValueError("La clave debe ser 1, 2 o 3.")
    clave = clave.strip()
    if clave and (len(clave) < 20 or any(c.isspace() for c in clave)):
        raise ValueError("Esa clave no parece válida: no debe tener espacios y suele tener unos 40 caracteres o más.")
    db.guardar_clave_gemini(numero, clave)
    _guardadas = None  # se vuelve a leer
    _est(numero).update(bloqueada_hasta=0.0, invalida=False, seguidos_429=0, motivo="")


def _est(numero: int) -> dict:
    return _estado.setdefault(numero, {"bloqueada_hasta": 0.0, "invalida": False,
                                       "seguidos_429": 0, "usos": 0, "motivo": ""})


def _activa(numero: int) -> bool:
    global _activas
    if _activas is None:
        try:
            _activas = db.claves_gemini_activas()
        except Exception:
            _activas = {}
    return _activas.get(numero, True)


def fijar_activa(numero: int, activa: bool):
    """Enciende o apaga una clave (queda guardado). Al encenderla se le da una nueva oportunidad."""
    if numero not in (1, 2, 3):
        raise ValueError("La clave debe ser 1, 2 o 3.")
    _activa(numero)  # asegura que el estado esté cargado
    _activas[numero] = activa
    db.fijar_clave_gemini(numero, activa)
    if activa:
        _est(numero).update(bloqueada_hasta=0.0, invalida=False, seguidos_429=0, motivo="")


def _usable(numero: int, ahora: float) -> bool:
    e = _est(numero)
    return _activa(numero) and not e["invalida"] and e["bloqueada_hasta"] <= ahora


def disponible() -> bool:
    """True si hay al menos una clave configurada y encendida."""
    return any(_activa(n) for n, _ in _claves())


def estado_claves() -> list[dict]:
    """Estado de las 3 claves para mostrarlo en la interfaz (sin revelar las claves)."""
    ahora, configuradas, salida = time.time(), dict(_claves()), []
    for n in (1, 2, 3):
        e = _est(n)
        if n not in configuradas:
            estado, texto = "sin_configurar", "sin configurar"
        elif not _activa(n):
            estado, texto = "desactivada", "desactivada"
        elif e["invalida"]:
            estado, texto = "invalida", "rechazada: revisa que sea válida"
        elif e["bloqueada_hasta"] > ahora:
            hasta = datetime.fromtimestamp(e["bloqueada_hasta"]).strftime("%H:%M")
            estado, texto = "en_pausa", f"en pausa hasta las {hasta} ({e['motivo']})"
        else:
            estado, texto = "lista", "lista" + (f" ({e['usos']} usos)" if e["usos"] else "")
        salida.append({"numero": n, "configurada": n in configuradas, "activa": _activa(n),
                       "estado": estado, "texto": texto})
    return salida


def mensaje_sin_claves() -> str:
    """Por qué no se puede usar Gemini ahora mismo."""
    if not _claves():
        return "Falta la clave de Gemini: pégala en GEMINI_API_KEY del archivo .env y reinicia uvicorn."
    partes = [f"clave {c['numero']}: {c['texto']}" for c in estado_claves() if c["configurada"]]
    return "No hay ninguna clave de Gemini disponible (" + "; ".join(partes) + ")."


def pop_avisos() -> list[str]:
    """Lo que pasó con las claves desde la última vez (cambios de clave, límites...), sin repetir."""
    avisos = list(dict.fromkeys(_avisos))
    _avisos.clear()
    return avisos


# ------------------------------------------------------------------------ las peticiones
def _error_de_estado(codigo: int) -> str:
    if codigo == 400:
        return "Gemini rechazó el contenido enviado (código 400); la imagen o la frase no es válida."
    return f"Gemini respondió con un error (código {codigo})."


def _es_problema_de_la_clave(respuesta) -> bool:
    """Una clave inválida o sin permiso. Un 400 por el contenido enviado NO es culpa de la clave."""
    try:
        error = respuesta.json().get("error", {})
    except ValueError:
        error = {}
    texto = f"{error.get('message', '')} {error.get('status', '')}".lower()
    return respuesta.status_code in (401, 403) or "api key" in texto or "api_key" in texto


def _pedir(metodo: str, cuerpo: dict, modelo: str | None = None) -> dict:
    """Llama a Gemini probando las claves por orden. Devuelve el JSON de la respuesta."""
    claves = _claves()
    if not claves:
        raise RuntimeError(mensaje_sin_claves())
    url = URL.format(modelo=modelo or config.GEMINI_EMBED_MODEL).replace(":embedContent", f":{metodo}")
    fallos: list[tuple[int, str]] = []  # claves que no sirvieron en esta petición
    for posicion, (numero, clave) in enumerate(claves):
        if not _usable(numero, time.time()):
            continue
        quedan_otras = any(_usable(m, time.time()) for m, _ in claves[posicion + 1:])
        cabeceras = {"x-goog-api-key": clave, "Content-Type": "application/json"}
        for intento in range(3):
            try:
                respuesta = httpx.post(url, headers=cabeceras, json=cuerpo, timeout=120)
            except httpx.HTTPError:
                raise RuntimeError("No se pudo conectar con Gemini. ¿Hay conexión a internet?") from None
            # Con una sola clave disponible se espera un poco (límite por minuto); con otra de respaldo, se cambia
            if respuesta.status_code == 429 and not quedan_otras and intento < 2:
                time.sleep(3 * (intento + 1))
                continue
            break

        e = _est(numero)
        if respuesta.status_code == 429:  # límite de uso: la clave descansa
            e["seguidos_429"] += 1
            larga = e["seguidos_429"] >= 3
            e["bloqueada_hasta"] = time.time() + (PAUSA_LARGA if larga else PAUSA_LIMITE)
            e["motivo"] = "cupo agotado" if larga else "límite de uso"
            fallos.append((numero, "llegó a su límite de uso"))
            continue
        if respuesta.status_code >= 500:  # falla pasajera de Gemini: no se castiga a la clave
            fallos.append((numero, "no obtuvo respuesta de Gemini"))
            continue
        if respuesta.status_code >= 400:
            if _es_problema_de_la_clave(respuesta):
                e["invalida"], e["motivo"] = True, "rechazada"
                fallos.append((numero, "fue rechazada por Gemini (inválida o sin permiso)"))
                continue
            raise RuntimeError(_error_de_estado(respuesta.status_code))

        e["seguidos_429"] = 0
        e["usos"] += 1
        for fallida, motivo in fallos:  # si hubo que cambiar de clave, se avisa
            _avisos.append(f"La clave {fallida} {motivo}; se usó la clave {numero}.")
        return respuesta.json()

    for fallida, motivo in fallos:
        _avisos.append(f"La clave {fallida} {motivo}.")
    if fallos and all("no obtuvo respuesta" in motivo for _, motivo in fallos):
        raise RuntimeError("Gemini no respondió. Inténtalo de nuevo en un momento.")
    raise RuntimeError(mensaje_sin_claves())


def _normalizado(vector) -> list[float]:
    if not vector or len(vector) != config.GEMINI_DIM:
        raise RuntimeError("Gemini devolvió un vector inesperado (tamaño distinto al de GEMINI_DIM).")
    v = np.array(vector, dtype=float)
    return (v / np.linalg.norm(v)).tolist()


def _embedding(partes: list[dict]) -> list[float]:
    """Vector de un contenido (texto o imagen)."""
    datos = _pedir("embedContent", {
        "model": f"models/{config.GEMINI_EMBED_MODEL}",
        "content": {"parts": partes},
        "output_dimensionality": config.GEMINI_DIM,
    })
    # La respuesta trae el vector en 'embedding.values' (o 'embeddings[0].values')
    return _normalizado((datos.get("embedding") or {}).get("values")
                        or ((datos.get("embeddings") or [{}])[0]).get("values"))


def embeddings_documentos(textos: list[str], lote: int = 50) -> list[list[float]]:
    """Vectores de muchos fragmentos de documento, varios por petición (mucho más rápido que uno a uno)."""
    salida = []
    for i in range(0, len(textos), lote):
        pedidos = [{"model": f"models/{config.GEMINI_EMBED_MODEL}",
                    "content": {"parts": [{"text": f"title: none | text: {t}"}]},
                    "output_dimensionality": config.GEMINI_DIM} for t in textos[i:i + lote]]
        datos = _pedir("batchEmbedContents", {"requests": pedidos})
        vectores = [e.get("values") for e in datos.get("embeddings", [])]
        if len(vectores) != len(pedidos):
            raise RuntimeError("Gemini devolvió menos vectores de los pedidos.")
        salida.extend(_normalizado(v) for v in vectores)
    return salida


def generar(prompt: str, imagenes: list[bytes] | None = None, maximo: int = 1024) -> str:
    """Respuesta de un modelo de Gemini (nube). Puede recibir imágenes (PNG) para que las vea."""
    partes = [{"text": prompt}] + [
        {"inline_data": {"mime_type": "image/png", "data": base64.b64encode(png).decode("ascii")}}
        for png in imagenes or []]
    datos = _pedir("generateContent", {
        "contents": [{"role": "user", "parts": partes}],
        "generationConfig": {"temperature": config.TEMPERATURE, "maxOutputTokens": maximo},
    }, modelo=config.GEMINI_CHAT_MODEL)
    candidatos = datos.get("candidates") or []
    texto = "".join(p.get("text", "") for p in ((candidatos[0].get("content") or {}).get("parts") or [])) \
        if candidatos else ""
    if not texto.strip():
        raise RuntimeError("Gemini no devolvió texto (puede que sus filtros bloquearan la respuesta).")
    return texto.strip()


def embedding_consulta(texto: str) -> list[float]:
    """Vector de una frase que se usa para BUSCAR ("un perro")."""
    return _embedding([{"text": f"task: search result | query: {texto}"}])


def embedding_imagen(png: bytes) -> list[float]:
    """Vector de una imagen (PNG)."""
    datos = base64.b64encode(png).decode("ascii")
    return _embedding([{"inline_data": {"mime_type": "image/png", "data": datos}}])
