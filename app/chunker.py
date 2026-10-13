"""Divide un texto largo en fragmentos (chunks) con solape."""
from app import config


def dividir_texto(texto: str) -> list[str]:
    """Parte el texto en fragmentos de ~CHUNK_SIZE caracteres con solape CHUNK_OVERLAP.

    Intenta cortar al final de una oración (punto) o en un espacio,
    para no partir palabras.
    """
    tamano = config.CHUNK_SIZE
    solape = min(config.CHUNK_OVERLAP, tamano // 2)  # el solape nunca debe ser enorme
    texto = " ".join(texto.split())  # limpia saltos de línea y espacios repetidos

    fragmentos = []
    inicio = 0
    while inicio < len(texto):
        fin = min(inicio + tamano, len(texto))

        if fin < len(texto):
            # Buscar un buen punto de corte en la segunda mitad del fragmento
            mitad = inicio + tamano // 2
            corte = max(texto.rfind(". ", mitad, fin), texto.rfind("? ", mitad, fin),
                        texto.rfind("! ", mitad, fin))
            if corte != -1:
                fin = corte + 1  # incluye el punto
            else:
                espacio = texto.rfind(" ", mitad, fin)
                if espacio != -1:
                    fin = espacio

        pedazo = texto[inicio:fin].strip()
        if pedazo:
            fragmentos.append(pedazo)

        if fin >= len(texto):
            break

        # El siguiente fragmento empieza un poco antes (solape), sin partir palabra
        inicio = max(fin - solape, inicio + 1)
        if inicio < len(texto) and texto[inicio - 1] != " ":
            siguiente = texto.find(" ", inicio)
            if siguiente != -1 and siguiente < fin:
                inicio = siguiente + 1
    return fragmentos
