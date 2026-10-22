"""Buscar dentro de las imágenes, audios y videos con una frase ("¿hay un perro?", "un audio que hable de energía solar").

Al subir un PDF se sacan sus imágenes y sus archivos de audio y video adjuntos; cada uno se convierte en un
vector con EmbeddingGemma 2 (local; también sirve Gemini para imágenes) y se guarda. Al preguntar, la frase
también se convierte en vector y se muestran los más parecidos.

Los audios y videos se parten en segmentos, así que cada resultado dice desde qué segundo empieza.

Importante: el parecido es una ESTIMACIÓN. Siempre habrá algo "más parecido", aunque nada trate de lo que se
busca; por eso se aplican umbrales por modelo y tipo (config.UMBRALES) y se muestra el porcentaje.
"""
import re

import numpy as np

from app import config, db, embeddings, gemini, gemma2, medios, transcribir
from app.ingest import ingerir_transcripcion
from app.imagenes import extraer_imagenes

_ARTICULO = r"(?:un|una|unos|unas|el|la|los|las)"
_MEDIO = r"(?:im[aá]gen(?:es)?|fotos?|fotograf[ií]as?|figuras?|ilustraci[oó]n(?:es)?|gr[aá]ficos?|audios?|videos?|v[ií]deos?|grabaci[oó]n(?:es)?|sonidos?)"
_TEMA = re.compile(
    rf"{_MEDIO}\s+(?:de|del|con|donde|que)\s+(?:(?:se\s+)?(?:habl\w+|trat\w+|mencion\w+|diga\w*|oiga\w*|escuch\w+|muestr\w+)\s+(?:de|del|sobre|acerca\s+de)?\s*)?"
    rf"(?:{_ARTICULO}\s+)?(?P<tema>[^?¿!.]+)", re.I)
_TIPOS = {"imagen": re.compile(r"im[aá]gen|foto|figura|ilustraci|gr[aá]fico", re.I),
          "audio": re.compile(r"\baudios?\b|grabaci[oó]n|sonido|pista", re.I),
          "video": re.compile(r"\bv[ií]deos?\b", re.I)}


def tema_de_la_pregunta(pregunta: str) -> str:
    """Lo que se busca: 'hay una imagen de un perro?' -> 'perro'. Si no se reconoce, la pregunta entera."""
    m = _TEMA.search(pregunta)
    return m.group("tema").strip() if m else re.sub(r"[¿?¡!]", "", pregunta).strip()


def tipos_pedidos(pregunta: str) -> tuple:
    """Qué tipos de medio menciona la pregunta (imagen, audio, video). Si no dice cuál, todos."""
    pedidos = tuple(t for t, patron in _TIPOS.items() if patron.search(pregunta))
    return pedidos or ("imagen", "audio", "video")


def _vector_de_consulta(proveedor: str, frase: str) -> np.ndarray:
    if proveedor == "gemma2":
        return np.array(gemma2.embedding_consulta(frase))
    return np.array(gemini.embedding_consulta(frase))


# ------------------------------------------------------------------------------ indexar
def _fila(tipo: str, proveedor: str, vector, **resto) -> dict:
    return {"tipo": tipo, "proveedor": proveedor, "embedding": np.array(vector), **resto}


def indexar(usuario: str, nombre: str, contenido_pdf: bytes) -> dict:
    """Indexa las imágenes del PDF y, con EmbeddingGemma 2, los audios y videos que trae adjuntos.

    Devuelve {"indexadas", "total", "audios", "videos", "error", "activo", "proveedor"}.
    Si falla a mitad (límite de uso, sin internet, memoria), se guardan los que ya se hicieron.
    """
    proveedor = embeddings.proveedor_medios()
    resultado = {"indexadas": 0, "total": 0, "audios": 0, "videos": 0, "error": None, "activo": True,
                 "proveedor": proveedor}
    if proveedor == "gemini" and not gemini.disponible():
        return {**resultado, "activo": False}
    filas, adjuntos, error = [], [], None
    imagenes = extraer_imagenes(contenido_pdf, config.IMAGENES_INCLUIR_ESCANEADAS, config.IMAGENES_MAX_POR_DOC)
    resultado["total"] = len(imagenes)
    for imagen in imagenes:
        try:
            vector = (gemma2.embedding_imagen(imagen["png"]) if proveedor == "gemma2"
                      else gemini.embedding_imagen(imagen["png"]))
        except RuntimeError as e:
            error = str(e)
            break
        filas.append(_fila("imagen", proveedor, vector, pagina=imagen["pagina"], ancho=imagen["ancho"],
                           alto=imagen["alto"], png=imagen["png"]))
    resultado["indexadas"] = len(filas)

    if proveedor == "gemma2" and not error:  # audios y videos adjuntos al PDF
        for adjunto in medios.adjuntos_del_pdf(contenido_pdf):
            try:
                nuevas = _segmentos_indexados(adjunto)
            except RuntimeError as e:
                error = str(e)
                break
            filas += nuevas
            adjuntos.append(medios.version_reproducible(adjunto))
            resultado["audios" if adjunto["tipo"] == "audio" else "videos"] += 1
    db.guardar_imagenes(usuario, nombre, filas, adjuntos)
    return {**resultado, "error": error}


def _segmentos_indexados(adjunto: dict, transcripciones: list | None = None) -> list[dict]:
    """Parte un archivo de audio o video en segmentos y calcula el vector de cada uno.

    Si se pasa una lista en `transcripciones`, se llena con lo que se dice en cada segmento (Whisper):
    [{"numero", "inicio", "texto"}]. En un video se transcribe su pista de audio.
    """
    filas, errores = [], []
    with medios.con_directorio_temporal() as tmp:
        if adjunto["tipo"] == "audio":
            for seg in medios.segmentos_de_audio(adjunto["datos"], tmp):
                filas.append(_fila("audio", "gemma2", gemma2.embedding_audio(seg["ruta"]), pagina=0,
                                   inicio=seg["inicio"], fin=seg["fin"], nombre=adjunto["nombre"]))
                _transcribir(seg, transcripciones)
        else:
            for seg in medios.segmentos_de_video(adjunto["datos"], tmp):
                filas.append(_fila("video", "gemma2", gemma2.embedding_video(seg["fotos"]), pagina=0,
                                   inicio=seg["inicio"], fin=seg["fin"], nombre=adjunto["nombre"],
                                   png=seg["png"], ancho=seg["ancho"], alto=seg["alto"]))
            if transcripciones is not None:  # lo que se dice en el video: su pista de audio
                for seg in medios.segmentos_de_audio(adjunto["datos"], tmp):
                    _transcribir(seg, transcripciones)
    return filas


def _transcribir(seg: dict, transcripciones: list | None):
    if transcripciones is None:
        return
    texto = transcribir.texto_de_wav(seg["ruta"])
    if texto:
        transcripciones.append({"numero": len(transcripciones) + 1, "inicio": seg["inicio"], "texto": texto})


def indexar_archivo_suelto(usuario: str, nombre: str, datos: bytes) -> dict:
    """Indexa un audio o video subido solo (no dentro de un PDF). Siempre con EmbeddingGemma 2."""
    tipo = medios.tipo_de_archivo(nombre)
    if tipo is None:
        raise ValueError("Ese archivo no es un audio ni un video conocido.")
    adjunto = {"nombre": nombre, "tipo": tipo, "mime": medios.mime_de(nombre), "datos": datos}
    transcripciones: list[dict] = []
    filas = _segmentos_indexados(adjunto, transcripciones)
    if not filas:
        raise ValueError("No se pudo leer ningún sonido o imagen de ese archivo.")
    db.guardar_imagenes(usuario, nombre, filas, [medios.version_reproducible(adjunto)])
    # El texto de lo que se dice entra como fragmentos: así se puede conversar, resumir, citar con el minuto...
    texto = ingerir_transcripcion(usuario, nombre, transcripciones)
    return {"tipo": tipo, "segmentos": len(filas), "fragmentos": texto["fragmentos"],
            "palabras": sum(len(t["texto"].split()) for t in transcripciones),
            "aviso_embeddings": texto["aviso_embeddings"]}


# ------------------------------------------------------------------------------ buscar
def buscar(usuario: str, consulta: str, documentos: list[str] | None = None,
           tipos: tuple = ("imagen",)) -> tuple[list[dict], list[dict]]:
    """Busca imágenes / audios / videos que coinciden con la frase. Devuelve (coinciden, candidatas).

    `candidatas` son los más parecidos, de mayor a menor, con {id, documento, pagina, tipo, inicio, similitud...}.
    `coinciden` son solo los que pasan los dos filtros (por modelo y tipo, ver config.UMBRALES):
      1. parecido al menos el mínimo (si ninguno llega, no hay coincidencia), y
      2. estar a menos del margen del mejor de su grupo (si el gato gana por mucho, no salen el perro ni la
         lombriz; si la pregunta es general, "un animal", salen todos los animales).
    """
    candidatas, coinciden = [], []
    for proveedor in db.proveedores_de_medios(usuario, tipos):
        grupo = db.buscar_imagenes(usuario, _vector_de_consulta(proveedor, consulta), documentos, 20,
                                   proveedor=proveedor, tipos=tipos)
        mejores = {}
        for c in grupo:  # el mejor de cada tipo, para aplicar el margen dentro de su tipo
            mejores.setdefault(c["tipo"], c["similitud"])
        for c in grupo:
            minimo, margen = config.UMBRALES.get(proveedor, {}).get(c["tipo"], (1.0, 0.0))
            c["proveedor"] = proveedor
            if c["similitud"] >= minimo and c["similitud"] >= mejores[c["tipo"]] - margen:
                c["minimo"] = minimo
                coinciden.append(c)
        candidatas += grupo
    candidatas.sort(key=lambda c: c["similitud"], reverse=True)
    coinciden.sort(key=lambda c: c["similitud"], reverse=True)
    return coinciden[:config.IMAGENES_TOP], candidatas
