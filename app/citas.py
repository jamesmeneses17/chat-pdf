"""Saber de qué fragmento(s) salió una respuesta, para citar documento y página.

El modelo pequeño casi nunca escribe [1], [2] en su respuesta. Por eso, si no las escribe, se
comparan las palabras de la respuesta con las de cada fragmento y se marcan los que la respaldan.
"""
import re
import unicodedata

_VACIAS = {
    "segun", "contexto", "informacion", "proporcionada", "proporciona", "documento",
    "documentos", "fragmento", "tambien", "ademas", "entre", "para", "desde", "hasta",
    "esta", "este", "estos", "estas", "tiene", "tienen", "puede", "pueden", "sobre",
    "como", "cual", "cuales", "donde", "cuando", "porque", "pero", "esos", "esas",
}


def _palabras(texto: str) -> set[str]:
    sin_tildes = "".join(c for c in unicodedata.normalize("NFD", texto.lower())
                         if unicodedata.category(c) != "Mn")
    return {w for w in re.findall(r"\w+", sin_tildes)
            if (len(w) >= 4 or w.isdigit()) and w not in _VACIAS}


def fragmentos_citados(respuesta: str, fragmentos: list[dict]) -> list[int]:
    """Números (n) de los fragmentos que respaldan la respuesta (como máximo 2)."""
    if not fragmentos or "no encontré esa información" in respuesta.lower():
        return []

    # 1. Si el modelo citó con [n], se respeta
    explicitas = sorted({int(n) for n in re.findall(r"\[(\d+)\]", respuesta)
                         if 1 <= int(n) <= len(fragmentos)})
    if explicitas:
        return explicitas[:3]

    # 2. Si no, los fragmentos que más comparten palabras con la respuesta
    de_la_respuesta = _palabras(respuesta)
    if not de_la_respuesta:
        return []
    puntajes = [len(de_la_respuesta & _palabras(f["contenido"])) / len(de_la_respuesta)
                for f in fragmentos]
    mejor = max(puntajes)
    if mejor < 0.3:
        return []
    return [f["n"] for f, p in zip(fragmentos, puntajes) if p >= max(0.3, mejor * 0.7)][:2]
