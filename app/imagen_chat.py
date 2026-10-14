"""Leer el texto de una imagen que el usuario adjunta en el chat (captura, foto de un papel...).

Usa el mismo OCR (Tesseract) que los PDFs escaneados. Lee las letras que hay en la imagen; NO
describe lo que la imagen muestra (un perro, una planta...): para eso haría falta un modelo de visión.
"""
import pymupdf

from app import config

MAX_BYTES = 10 * 1024 * 1024  # 10 MB
LADO_MINIMO = 1400  # px: las imágenes más chicas se agrandan para que el OCR lea mejor
TIPOS = {"png", "jpg", "jpeg"}


def leer_texto_de_imagen(contenido: bytes, nombre: str = "imagen.png") -> str:
    """Devuelve el texto leído en la imagen, en una sola línea (vacío si no encontró letras)."""
    tipo = nombre.rsplit(".", 1)[-1].lower() if "." in nombre else "png"
    if tipo not in TIPOS:
        tipo = "png"
    with pymupdf.open(stream=contenido, filetype=tipo) as doc:
        pagina = doc[0]
        # Una imagen se abre como una página de 1 punto por píxel: se elige el dpi para que el
        # OCR trabaje con unos LADO_MINIMO píxeles en el lado largo (sin reducir las grandes)
        lado = max(pagina.rect.width, pagina.rect.height)
        dpi = max(72, round(72 * LADO_MINIMO / lado)) if lado < LADO_MINIMO else 72
        ocr = pagina.get_textpage_ocr(language=config.OCR_LANG, dpi=dpi, full=True,
                                      tessdata=config.TESSDATA_DIR or None)
        texto = pagina.get_text(textpage=ocr)
    return " ".join(texto.split())
