"""Memoria de conversación: entender preguntas de seguimiento como "¿y eso qué significa?".

Una pregunta suelta ("¿y eso qué significa?") no sirve para buscar en los PDFs: no dice de qué
se habla. Por eso, si parece un seguimiento, se reescribe con ayuda del historial para que se
entienda sola, y esa versión es la que se busca. Además, el historial reciente se incluye en
el prompt para que el modelo mantenga el hilo.
"""
import re
import unicodedata

from app.llm import generar_respuesta

MAX_MENSAJES = 6  # últimos mensajes que se recuerdan (3 intercambios)
MAX_CARACTERES = 400  # cuánto se recuerda de cada mensaje

# Palabras que delatan una pregunta que depende de lo anterior
_MARCADORES = {
    "eso", "esto", "esa", "ese", "esos", "esas", "ello", "ellos", "ellas", "aquello",
    "anterior", "mismo", "misma", "mismos", "mismas", "ahi", "alli", "dicho", "dicha",
}
# Palabras con que suele empezar una pregunta que continúa la anterior ("¿y la fecha?")
_INICIOS = {"y", "e", "pero", "entonces", "ademas", "tambien"}
# Palabras de pregunta que no nombran ningún tema por sí solas
_GENERICAS = {
    "sirve", "significa", "funciona", "quiere", "decir", "explica", "explicame", "detalla",
    "detalle", "ejemplo", "ejemplos", "cuesta", "vale", "cuanto", "cuantos", "cuanta",
    "cuantas", "como", "cual", "cuales", "quien", "quienes", "donde", "cuando", "porque",
    "entonces", "ademas", "tambien", "hace", "hacen", "tiene", "tienen", "puede", "pueden",
    "seria", "serian", "dice", "dicen", "mas", "menos", "pero", "algo", "cosa", "cosas",
    "para", "sobre", "otro", "otra", "otros", "otras", "puedes", "podrias", "podria", "dame",
    "dime", "explicar", "profundizar", "resumir",
}


def _sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                   if unicodedata.category(c) != "Mn")


def recortar_historial(historial: list[dict]) -> list[dict]:
    """Se queda con los últimos mensajes y los acorta, para no llenar el contexto del modelo."""
    utiles = [m for m in historial if m.get("rol") in ("user", "assistant") and m.get("texto")]
    recientes = utiles[-MAX_MENSAJES:]
    return [{"rol": m["rol"], "texto": m["texto"][:MAX_CARACTERES]} for m in recientes]


def texto_historial(historial: list[dict]) -> str:
    """El historial como texto para el prompt."""
    nombres = {"user": "Usuario", "assistant": "Asistente"}
    return "\n".join(f"{nombres[m['rol']]}: {m['texto']}" for m in historial)


def parece_seguimiento(pregunta: str) -> bool:
    """True si la pregunta probablemente depende de lo anterior.

    Lo es si empieza con 'y...', si dice 'eso', 'esto'..., o si no nombra ningún tema propio
    (como '¿para qué sirve?'). Una pregunta con su propio tema ('¿Qué es el SARLAFT?') se
    entiende sola y no se mezcla con la conversación anterior.
    """
    palabras = re.findall(r"\w+", _sin_tildes(pregunta))
    if not palabras:
        return False
    if palabras[0] in _INICIOS or _MARCADORES & set(palabras):
        return True
    con_tema = [p for p in palabras if len(p) >= 4 and p not in _GENERICAS]
    return not con_tema


def _normalizada(texto: str) -> str:
    return " ".join(re.findall(r"\w+", _sin_tildes(texto)))


def reformular_pregunta(pregunta: str, historial: list[dict]) -> str:
    """Reescribe la pregunta para que se entienda sola.

    Primero se le pide al modelo. Los modelos pequeños a veces copian la pregunta sin cambiarla;
    en ese caso se le antepone la pregunta anterior del usuario, que casi siempre aporta el tema.
    """
    if not historial or not parece_seguimiento(pregunta):
        return pregunta

    prompt = (
        "Reescribe la ÚLTIMA PREGUNTA del usuario para que se entienda por sí sola, sin ver la "
        "conversación. Reemplaza palabras como 'eso', 'esto' o 'ese' por aquello a lo que se "
        "refieren, según el historial.\n"
        "- Responde ÚNICAMENTE con la pregunta reescrita, en una línea, sin explicaciones.\n\n"
        "Ejemplo:\n"
        "HISTORIAL:\nUsuario: ¿Cuál es el plazo del crédito?\nAsistente: El plazo es de 24 meses.\n"
        "ÚLTIMA PREGUNTA: ¿y eso es mucho?\n"
        "PREGUNTA REESCRITA: ¿Es mucho un plazo de 24 meses para el crédito?\n\n"
        f"HISTORIAL:\n{texto_historial(historial)}\n"
        f"ÚLTIMA PREGUNTA: {pregunta}\n"
        "PREGUNTA REESCRITA:"
    )
    lineas = generar_respuesta(prompt).strip().splitlines()
    nueva = lineas[0].strip().strip('"“”«»') if lineas else ""
    valida = nueva and len(nueva) <= 300 and "MODO PRUEBA" not in nueva
    if valida and _normalizada(nueva) != _normalizada(pregunta):
        return nueva

    # Respaldo: la pregunta anterior del usuario aporta el tema
    anteriores = [m["texto"] for m in historial if m["rol"] == "user"]
    return f"{anteriores[-1]} {pregunta}" if anteriores else pregunta
